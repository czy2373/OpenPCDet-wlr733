import numpy as np

class Track:
    def __init__(self, track_id, box, label, score):
        self.id = track_id
        self.box = box        # (7,) x,y,z,dx,dy,dz,yaw
        self.label = label
        self.score = score
        self.age = 0          # 存活帧数
        self.miss = 0         # 连续没匹配上的次数

class IoUTracker:
    def __init__(self, iou_thresh=0.3, max_age=5):
        """
        iou_thresh: IoU 阈值，大于此才认为是同一个目标
        max_age: 连续多少帧没匹配上就删除 track
        """
        self.iou_thresh = iou_thresh
        self.max_age = max_age
        self.tracks = []
        self.next_id = 1

    @staticmethod
    def boxes_to_bev(boxes):
        """
        boxes: (N,7) [x,y,z,dx,dy,dz,yaw]
        这里简单粗暴地忽略 yaw，当成轴对齐框
        返回: (N,4) [x1,y1,x2,y2]
        """
        x, y, z, dx, dy, dz, yaw = boxes.T
        x1 = x - dx / 2.0
        x2 = x + dx / 2.0
        y1 = y - dy / 2.0
        y2 = y + dy / 2.0
        return np.stack([x1, y1, x2, y2], axis=1)

    @staticmethod
    def iou_bev(boxes1, boxes2):
        """
        boxes1: (M,7) or (M,4)
        boxes2: (N,7) or (N,4)
        返回: (M,N) IoU 矩阵
        """
        if boxes1.shape[1] == 7:
            b1 = IoUTracker.boxes_to_bev(boxes1)
        else:
            b1 = boxes1
        if boxes2.shape[1] == 7:
            b2 = IoUTracker.boxes_to_bev(boxes2)
        else:
            b2 = boxes2

        M = b1.shape[0]
        N = b2.shape[0]
        ious = np.zeros((M, N), dtype=np.float32)
        if M == 0 or N == 0:
            return ious

        for i in range(M):
            x1, y1, x2, y2 = b1[i]
            xx1 = np.maximum(x1, b2[:, 0])
            yy1 = np.maximum(y1, b2[:, 1])
            xx2 = np.minimum(x2, b2[:, 2])
            yy2 = np.minimum(y2, b2[:, 3])

            inter_w = np.maximum(0.0, xx2 - xx1)
            inter_h = np.maximum(0.0, yy2 - yy1)
            inter = inter_w * inter_h

            area1 = (x2 - x1) * (y2 - y1)
            area2 = (b2[:, 2] - b2[:, 0]) * (b2[:, 3] - b2[:, 1])
            union = area1 + area2 - inter + 1e-6

            ious[i] = inter / union

        return ious

    def update(self, det_boxes, det_labels, det_scores):
        """
        det_boxes: (N,7)
        det_labels: (N,)
        det_scores: (N,)
        返回: list(dict)，每个检测的 {track_id, box, label, score}
        """
        det_boxes = det_boxes.astype(np.float32)
        N = det_boxes.shape[0]

        # 先默认每个 detection 的 track_id = -1（未匹配）
        det_track_ids = np.full(N, -1, dtype=np.int32)

        # 当前 tracks 信息
        M = len(self.tracks)
        if M > 0 and N > 0:
            track_boxes = np.stack([t.box for t in self.tracks], axis=0)
            iou_mat = self.iou_bev(track_boxes, det_boxes)  # (M,N)

            # 每个 track 找到 IoU 最大的 detection
            for ti in range(M):
                # 限制类别相同再匹配（可选）
                same_cls = (det_labels == self.tracks[ti].label)
                valid_idx = np.where(same_cls)[0]
                if len(valid_idx) == 0:
                    continue

                ious = iou_mat[ti, valid_idx]
                di = valid_idx[np.argmax(ious)]
                best_iou = iou_mat[ti, di]

                if best_iou >= self.iou_thresh:
                    # 若这个 det 还没被占用，就直接用
                    if det_track_ids[di] == -1:
                        det_track_ids[di] = self.tracks[ti].id
                        # 更新 track 状态
                        self.tracks[ti].box = det_boxes[di]
                        self.tracks[ti].score = det_scores[di]
                        self.tracks[ti].age += 1
                        self.tracks[ti].miss = 0

        # 对于未匹配的 detection，新建 track
        for i in range(N):
            if det_track_ids[i] == -1:
                new_id = self.next_id
                self.next_id += 1
                det_track_ids[i] = new_id
                self.tracks.append(
                    Track(new_id, det_boxes[i], det_labels[i], det_scores[i])
                )

        # 没匹配到的老 track，miss+1
        alive_tracks = []
        for t in self.tracks:
            # 如果这个 track 在本帧里被分配过，就一定已经更新过 miss=0
            # 没匹配到的，我们在这里 +1
            # 简单做法：再扫一遍看它 id 是否出现在 det_track_ids 里
            if t.id in det_track_ids:
                alive_tracks.append(t)
            else:
                t.miss += 1
                if t.miss <= self.max_age:
                    alive_tracks.append(t)
                # 否则就丢弃

        self.tracks = alive_tracks

        # 返回当前帧每一个 detection 对应的 track 信息
        results = []
        for i in range(N):
            results.append(
                dict(
                    track_id=int(det_track_ids[i]),
                    box=det_boxes[i],
                    label=int(det_labels[i]),
                    score=float(det_scores[i]),
                )
            )
        return results
