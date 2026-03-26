import dpkt
import numpy as np
import struct

# WLR-733 单回波数据包解析
# 每个数据块 260 字节：2字节帧头 + 2字节方位角 + 64*(2+1+1)字节通道数据

# 64通道垂直角（从手册 Table 1.2 读取，或从 VanJeeView 参数配置里读取实际值）
VERT_ANGLES = np.array([
    -1.5,-1.7,-1.9,-2.0,-2.1,-2.2,-2.3,-2.4,
    -2.5,-2.6,-2.7,-2.8,-2.9,-3.0,-3.1,-3.2,
    -3.3,-3.4,-3.5,-3.6,-3.7,-3.8,-3.9,-4.0,
    -4.1,-4.2,-4.3,-4.4,-4.5,-4.6,-4.8,-5.0,
    -5.2,-5.4,-5.6,-5.8,-6.1,-6.4,-6.7,-7.0,
    -7.4,-7.8,-8.2,-8.6,-9.2,-9.8,-10.4,-11.0,
    -12.,-13.,-14.,-15.,-16.,-17.,-19.,-21.,
    -23.,-25.,-27.,-29.,-32.,-35.,-38.,-42.
], dtype=np.float32)

def parse_wlr733_packet(udp_data):
    """解析单个 UDP 包，返回点列表 [(x,y,z,intensity), ...]"""
    points = []
    offset = 0
    for block in range(4):  # 单回波每包4个数据块
        if offset + 260 > len(udp_data):
            break
        header = struct.unpack_from('<H', udp_data, offset)[0]
        if header != 0xFFEE:
            offset += 260
            continue
        azimuth_raw = struct.unpack_from('<H', udp_data, offset + 2)[0]
        azimuth = azimuth_raw / 100.0  # 转换为度
        
        for ch in range(64):
            ch_offset = offset + 4 + ch * 4
            dist_raw = struct.unpack_from('<H', udp_data, ch_offset)[0]
            intensity = udp_data[ch_offset + 2]
            # reflectance = udp_data[ch_offset + 3]  # 反射率
            
            if dist_raw in [4,16,17,18,19,20]:  # 无效距离码（手册3.2.1）
                continue
            dist_m = dist_raw * 4 / 1000.0  # 毫米→米
            if dist_m < 0.1:
                continue
            
            omega = np.radians(VERT_ANGLES[ch])
            alpha = np.radians(azimuth)
            x = dist_m * np.cos(omega) * np.sin(alpha)
            y = -dist_m * np.cos(omega) * np.cos(alpha)
            z = dist_m * np.sin(omega)
            
            intensity_norm = intensity / 255.0  # 归一化到 [0,1]
            points.append([x, y, z, intensity_norm])
        
        offset += 260
    return points

def pcap_to_bin(pcap_path, output_bin_path):
    """把 pcap 转为带 intensity 的 bin 文件（KITTI格式）"""
    all_points = []
    with open(pcap_path, 'rb') as f:
        pcap = dpkt.pcap.Reader(f)
        for ts, buf in pcap:
            try:
                eth = dpkt.ethernet.Ethernet(buf)
                if not isinstance(eth.data, dpkt.ip.IP):
                    continue
                ip = eth.data
                if not isinstance(ip.data, dpkt.udp.UDP):
                    continue
                udp = ip.data
                if udp.sport == 3333:  # WLR-733 雷达端口
                    pts = parse_wlr733_packet(bytes(udp.data))
                    all_points.extend(pts)
            except:
                continue
    
    if all_points:
        arr = np.array(all_points, dtype=np.float32)
        arr.tofile(output_bin_path)
        print(f"保存 {len(arr)} 个点到 {output_bin_path}")
        print(f"intensity 范围: {arr[:,3].min():.3f} ~ {arr[:,3].max():.3f}")
    return all_points