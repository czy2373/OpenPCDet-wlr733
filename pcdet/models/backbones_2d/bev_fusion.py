import torch
import torch.nn as nn
import torch.nn.functional as F


class BEVFusion(nn.Module):
    """
    Robust BEV fusion with:
      - 1x1 projection (img -> same channels)
      - ROI-aware normalization
      - per-pixel soft gating (sigmoid) with learnable bias
      - alpha scaling (learnable by default)

    Call:
      fused = bev_fusion(lidar_bev, img_bev, roi_mask=roi_mask)
    where:
      lidar_bev: (B, C, H, W)
      img_bev:   (B, C, H, W)
      roi_mask:  (B, 1, H, W) or (B, H, W) or None
    """

    def __init__(self, model_cfg=None):
        super().__init__()
        # -------- defaults --------
        img_ch = 64
        alpha_init = 0.005
        gate_temp = 1.0
        gate_bias_init = 2.0
        debug = True
        use_proj = True
        use_alpha_map = False
        alpha_map_hidden = 16

        # -------- read cfg --------
        if model_cfg is not None:
            img_ch = int(getattr(model_cfg, "IMG_CHANNELS", img_ch))
            alpha_init = float(getattr(model_cfg, "ALPHA_INIT", alpha_init))
            gate_temp = float(getattr(model_cfg, "GATE_TEMP", gate_temp))
            gate_bias_init = float(getattr(model_cfg, "GATE_BIAS_INIT", gate_bias_init))
            debug = bool(getattr(model_cfg, "DEBUG", debug))
            use_proj = bool(getattr(model_cfg, "USE_PROJ", use_proj))
            use_alpha_map = bool(getattr(model_cfg, "USE_ALPHA_MAP", use_alpha_map))
            alpha_map_hidden = int(getattr(model_cfg, "ALPHA_MAP_HIDDEN", alpha_map_hidden))

        self.debug = debug
        self.gate_temp = gate_temp
        self.use_alpha_map = use_alpha_map

        # 1x1 proj for image bev features
        if use_proj:
            self.proj = nn.Sequential(
                nn.Conv2d(img_ch, img_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(img_ch),
                nn.ReLU(inplace=True),
            )
        else:
            self.proj = nn.Identity()

        # learnable alpha
        self.alpha = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))

        # learnable gate bias (crucial to avoid gate always near 0 at init)
        self.gate_bias = nn.Parameter(torch.tensor(gate_bias_init, dtype=torch.float32))

        if self.use_alpha_map:
            self.alpha_head = nn.Sequential(
                nn.Conv2d(2, alpha_map_hidden, kernel_size=3, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(alpha_map_hidden, 1, kernel_size=1, bias=True),
            )
            nn.init.zeros_(self.alpha_head[-1].weight)
            nn.init.zeros_(self.alpha_head[-1].bias)
        else:
            self.alpha_head = None

    @staticmethod
    def _ensure_roi_mask(roi_mask, ref_tensor):
        """Return roi mask as (B,1,H,W) float tensor on same device/dtype as ref_tensor."""
        if roi_mask is None:
            return None
        if roi_mask.dim() == 3:
            roi_mask = roi_mask.unsqueeze(1)  # (B,1,H,W)
        if roi_mask.dim() != 4:
            raise ValueError(f"roi_mask must be (B,H,W) or (B,1,H,W), got {roi_mask.shape}")
        return roi_mask.to(device=ref_tensor.device, dtype=ref_tensor.dtype)

    @staticmethod
    def _roi_mean_abs(x, m):
        """
        ROI-aware mean(|x|) over all dims, returning shape (B,1,1,1).
        x: (B,C,H,W), m: (B,1,H,W)
        """
        # broadcast m to channels
        mc = m
        if x.size(1) != 1:
            mc = m.expand(-1, x.size(1), -1, -1)
        num = (x.abs() * mc).sum(dim=(1, 2, 3), keepdim=True)
        den = mc.sum(dim=(1, 2, 3), keepdim=True) + 1e-6
        return num / den

    def forward(self, lidar_bev, img_bev, roi_mask=None):
        """
        lidar_bev: (B,C,H,W)
        img_bev: (B,C,H,W)
        roi_mask: (B,1,H,W) or (B,H,W) or None
        """
        assert lidar_bev.dim() == 4 and img_bev.dim() == 4, \
            f"Expect 4D tensors, got lidar {lidar_bev.shape}, img {img_bev.shape}"
        assert lidar_bev.shape == img_bev.shape, \
            f"Shape mismatch: lidar {lidar_bev.shape} vs img {img_bev.shape}"

        m = self._ensure_roi_mask(roi_mask, lidar_bev)  # (B,1,H,W) or None

        # project image features
        img = self.proj(img_bev)

        # ROI-aware normalization on img to stabilize scale when BEV is sparse
        if m is not None:
            denom = self._roi_mean_abs(img, m)  # (B,1,1,1)
        else:
            denom = img.abs().mean(dim=(1, 2, 3), keepdim=True) + 1e-6
        img = img / (denom + 1e-6)

        # -------- per-pixel soft gating --------
        # energy maps: (B,1,H,W)
        img_e = img.abs().mean(dim=1, keepdim=True)
        lidar_e = lidar_bev.abs().mean(dim=1, keepdim=True)

        # gate: (B,1,H,W)
        score = (img_e - lidar_e) + self.gate_bias
        gate = torch.sigmoid(score * self.gate_temp)
        gate = 0.05 + 0.95 * gate


        if m is not None:
            gate = gate * m  # enforce ROI

        # apply gate to image features (broadcast to channels)
        img_g = img * gate

        # （建议）如果有 ROI mask，定义一个只在 ROI 内注入的 img_g_roi
        if m is not None:
            img_g_roi = img_g * m
        else:
            img_g_roi = img_g

        # alpha (keep positive-ish; allow learn, but avoid crazy negatives)
        alpha = torch.clamp(self.alpha, min=0.0)

        if self.use_alpha_map:
            alpha_in = torch.cat([lidar_e, img_e], dim=1)
            alpha_map = torch.sigmoid(self.alpha_head(alpha_in))
            if m is not None:
                alpha_map = alpha_map * m
            alpha_eff = alpha * alpha_map
        else:
            alpha_map = None
            alpha_eff = alpha

        fused = lidar_bev + alpha_eff * img_g_roi

        # ---- debug (print every N iters) ----
        if self.debug:
            if not hasattr(self, "_dbg_i"):
                self._dbg_i = 0
            self._dbg_i += 1

            N = 100  # 每 N 次打印一次
            if (self._dbg_i % N) == 0:
                with torch.no_grad():
                    lidar_mean = lidar_bev.abs().mean().item()
                    img_mean = img.abs().mean().item()          # 注意：这里用的是投影+归一化后的 img
                    img_g_mean = img_g.abs().mean().item()      # 门控后的 img_g
                    fused_minus = (fused - lidar_bev).abs().mean().item()

                    roi_mean = m.mean().item() if m is not None else -1.0
                    gate_mean = gate.mean().item()
                    gate_min = gate.min().item()
                    gate_max = gate.max().item()

                    if alpha_map is not None:
                        alpha_map_mean = alpha_map.mean().item()
                        alpha_map_min = alpha_map.min().item()
                        alpha_map_max = alpha_map.max().item()
                        alpha_eff_mean = alpha_eff.mean().item()
                    else:
                        alpha_map_mean = -1.0
                        alpha_map_min = -1.0
                        alpha_map_max = -1.0
                        alpha_eff_mean = alpha.item()

                    if m is not None:
                        # ROI 内 gate 的均值（避免被非ROI大量0稀释）
                        gate_roi_mean = (gate * m).sum().item() / (m.sum().item() + 1e-6)
                    else:
                        gate_roi_mean = -1.0

                print(f"[BEV_FUSION] alpha= {alpha.item()}")
                print(f"[BEV_FUSION] lidar_mean= {lidar_mean}")
                print(f"[BEV_FUSION] img_mean= {img_mean}")
                print(f"[BEV_FUSION] img_g_mean= {img_g_mean}")
                print(f"[BEV_FUSION] fused_minus_lidar_mean= {fused_minus}")
                print(f"[BEV_FUSION] roi_mean= {roi_mean}")
                print(f"[BEV_FUSION] gate_roi_mean={gate_roi_mean:.4f} gate_mean={gate_mean:.4f} gate_min={gate_min:.4f} gate_max={gate_max:.4f}")
                if alpha_map is not None:
                    print(f"[BEV_FUSION] alpha_map_mean={alpha_map_mean:.4f} alpha_map_min={alpha_map_min:.4f} alpha_map_max={alpha_map_max:.4f} alpha_eff_mean={alpha_eff_mean:.6f}")


        return fused
