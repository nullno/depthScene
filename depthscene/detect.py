"""目标检测：ONNX 模型封装、输出解码、去重合并、时序平滑。

一个检测器就是一个固定输入/固定词表的 ONNX：YOLO-World（开放词表）或 YOLOv10（COCO 80）。
多个检测器的结果会合并去重；检测可以丢到后台线程，和下一帧的深度推理重叠。
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, replace

import cv2
import numpy as np

from .runtime import make_session

COCO_LABELS = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
    'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
    'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush',
]


@dataclass
class Detection:
    """归一化坐标 [0,1] 的检测框；distance 由深度图补上。"""
    label: str
    score: float
    x: float
    y: float
    w: float
    h: float
    src: str = ''
    zh: str = ''
    distance: float | None = None
    hits: int = 1
    misses: int = 0

    def as_dict(self) -> dict:
        out = {
            'label': self.label,
            'labelZh': self.zh,
            'score': self.score,
            'x': self.x, 'y': self.y, 'w': self.w, 'h': self.h,
            'src': self.src,
        }
        if self.distance is not None:
            out['distance'] = self.distance
        return out


# ---------------- 通用几何 / 解码 ----------------

def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def load_meta(model_path: str) -> dict:
    """模型目录下的 labels.json：vocab（类别名）/ zh（中文名）/ threshold。"""
    meta_path = os.path.join(os.path.dirname(str(model_path)), 'labels.json')
    if not os.path.exists(meta_path):
        return {}
    try:
        with open(meta_path, encoding='utf-8') as f:
            return json.load(f) or {}
    except Exception:  # noqa: BLE001
        return {}


def detect_kind(out_shape, n_classes: int) -> str:
    """
      e2e  [1, 300, 6]      (x1,y1,x2,y2,score,cls) 已含 NMS —— YOLOv10
      raw  [1, 4+N, 8400]   (cx,cy,w,h,cls...)      需要自己解码 —— YOLOv8 / YOLO-World
    类别很少时 4+N 也可能等于 6，所以先按类别数排除歧义。
    """
    if len(out_shape) != 3:
        return 'raw'
    a, b = out_shape[1], out_shape[2]
    if not isinstance(a, int) or not isinstance(b, int) or a <= 0 or b <= 0:
        return 'raw'
    if n_classes and (a == n_classes + 4 or b == n_classes + 4):
        return 'raw'
    return 'e2e' if (a == 6 and b > 6) or (b == 6 and a > 6) else 'raw'


def nms_xyxy(boxes, scores, classes, iou_thr: float = 0.45) -> list[int]:
    """按类别做非极大值抑制，返回保留的下标。"""
    order = np.argsort(-scores)
    keep: list[int] = []

    while order.size:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break

        rest = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])

        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_r = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        iou = inter / np.maximum(1e-9, area_i + area_r - inter)

        order = rest[~((iou > iou_thr) & (classes[rest] == classes[i]))]

    return keep


def decode(pred, threshold: float, kind: str = 'raw') -> list[tuple]:
    """网络输出 → [(x1, y1, x2, y2, score, cls)]，坐标是模型输入像素。"""
    d = np.squeeze(pred)
    if d.ndim != 2:
        return []
    if d.shape[0] > d.shape[1]:      # [anchors, C] → [C, anchors]
        d = d.T

    if kind == 'e2e' and d.shape[0] == 6:
        keep = d[4] >= threshold
        t = d[:, keep]
        return [(float(x1), float(y1), float(x2), float(y2), float(s), int(round(float(c))))
                for x1, y1, x2, y2, s, c in t.T]

    if d.shape[0] < 5:
        return []

    scores = d[4:]
    cls = np.argmax(scores, axis=0)
    conf = scores[cls, np.arange(scores.shape[1])]

    idx = np.flatnonzero(conf >= threshold)
    if not idx.size:
        return []
    # 大词表下候选很多，只留置信度最高的 600 个，NMS 才不至于卡住
    if idx.size > 600:
        idx = idx[np.argsort(-conf[idx])[:600]]

    conf, cls = conf[idx], cls[idx]
    cx, cy, bw, bh = d[0][idx], d[1][idx], d[2][idx], d[3][idx]
    boxes = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)

    keep = nms_xyxy(boxes, conf, cls, 0.45)
    return [(float(boxes[i, 0]), float(boxes[i, 1]), float(boxes[i, 2]), float(boxes[i, 3]),
             float(conf[i]), int(cls[i])) for i in keep]


def iou(a: Detection, b: Detection) -> float:
    iw = max(0.0, min(a.x + a.w, b.x + b.w) - max(a.x, b.x))
    ih = max(0.0, min(a.y + a.h, b.y + b.h) - max(a.y, b.y))
    inter = iw * ih
    if inter <= 0:
        return 0.0
    return inter / max(1e-9, a.w * a.h + b.w * b.h - inter)


def is_duplicate(a: Detection, b: Detection) -> bool:
    """
    同类框 IoU ≥ 0.5 视为同一个；
    不同类（cup vs mug、tv vs monitor 这类近义）要求更严，IoU ≥ 0.7，
    免得把「人手里的杯子」当成人的重复框抹掉。
    """
    o = iou(a, b)
    return o >= 0.5 if a.label == b.label else o >= 0.7


def merge(dets: list[Detection], max_det: int) -> list[Detection]:
    """置信度优先的贪心去重，最多 max_det 个。"""
    pool = sorted(dets, key=lambda d: -d.score)
    kept: list[Detection] = []
    for cand in pool:
        if len(kept) >= max_det:
            break
        if any(is_duplicate(cand, k) for k in kept):
            continue
        kept.append(cand)
    return kept


# ---------------- 单个检测模型 ----------------

class Detector:
    """一个 ONNX 检测模型 + 它的词表。"""

    def __init__(self, path, providers: list[str], default_threshold: float = 0.30, threads: int = 0):
        self.path = str(path)
        self.name = os.path.basename(os.path.dirname(self.path)) or 'detector'

        meta = load_meta(self.path)
        self.sess, self.ep = make_session(self.path, providers, threads, f'检测 {self.name}')

        inp = self.sess.get_inputs()[0]
        self.input = inp.name
        self.size = int(inp.shape[-1]) if isinstance(inp.shape[-1], int) and inp.shape[-1] > 0 else 640

        self.labels = [str(w) for w in (meta.get('vocab') or [])] or list(COCO_LABELS)
        self.zh: dict[str, str] = meta.get('zh') or {}
        self.kind = detect_kind(self.sess.get_outputs()[0].shape, len(self.labels))
        self.threshold = float(meta.get('threshold') or default_threshold)
        self.ms = 0.0

    def infer(self, frame: np.ndarray) -> list[Detection]:
        h, w = frame.shape[:2]
        size = self.size
        r = min(size / w, size / h)
        nw, nh = int(round(w * r)), int(round(h * r))
        dx, dy = (size - nw) // 2, (size - nh) // 2

        canvas = np.full((size, size, 3), 114, np.uint8)
        canvas[dy:dy + nh, dx:dx + nw] = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        x = canvas[:, :, ::-1].astype(np.float32)
        x *= 1.0 / 255.0
        x = np.ascontiguousarray(np.transpose(x, (2, 0, 1))[None])

        t0 = time.perf_counter()
        out = self.sess.run(None, {self.input: x})[0]
        self.ms = (time.perf_counter() - t0) * 1000.0

        items: list[Detection] = []
        for x1, y1, x2, y2, score, cls in decode(out, self.threshold, self.kind):
            nx1 = float(clamp((x1 - dx) / r / w, 0.0, 1.0))
            ny1 = float(clamp((y1 - dy) / r / h, 0.0, 1.0))
            nx2 = float(clamp((x2 - dx) / r / w, 0.0, 1.0))
            ny2 = float(clamp((y2 - dy) / r / h, 0.0, 1.0))
            if nx2 - nx1 < 0.01 or ny2 - ny1 < 0.01:
                continue

            label = self.labels[cls] if 0 <= cls < len(self.labels) else f'class{cls}'
            items.append(Detection(
                label=label, score=round(float(score), 3),
                x=round(nx1, 4), y=round(ny1, 4),
                w=round(nx2 - nx1, 4), h=round(ny2 - ny1, 4),
                src=self.name, zh=self.zh.get(label, ''),
            ))
        return items


# ---------------- 多模型 + 时序平滑 ----------------

class Tracker:
    """
    按 IoU 关联相邻两次检测，对框做 EMA 平滑。
    检测是隔帧跑的，不关联的话框会一跳一跳；漏检时保留一两次，
    远处目标也不会闪一下就消失。
    """

    def __init__(self, alpha: float = 0.45, keep: int = 2, match_iou: float = 0.3):
        self.alpha = min(1.0, max(0.0, alpha))
        self.keep = max(0, keep)
        self.match_iou = match_iou
        self._tracks: list[Detection] = []

    def reset(self) -> None:
        self._tracks = []

    def update(self, dets: list[Detection]) -> list[Detection]:
        prev = self._tracks
        used = [False] * len(prev)
        out: list[Detection] = []
        k = self.alpha

        for d in dets:
            best, best_iou = -1, self.match_iou
            for i, t in enumerate(prev):
                if used[i] or t.label != d.label:
                    continue
                o = iou(d, t)
                if o > best_iou:
                    best, best_iou = i, o

            if best < 0:
                out.append(d)
                continue

            used[best] = True
            t = prev[best]
            out.append(replace(
                d,
                x=round(t.x + k * (d.x - t.x), 4),
                y=round(t.y + k * (d.y - t.y), 4),
                w=round(t.w + k * (d.w - t.w), 4),
                h=round(t.h + k * (d.h - t.h), 4),
                score=round(t.score + k * (d.score - t.score), 3),
                hits=t.hits + 1,
            ))

        # 这一轮没被匹配上的旧框：保留 keep 次，分数衰减
        for i, t in enumerate(prev):
            if used[i] or t.misses >= self.keep:
                continue
            out.append(replace(t, score=round(t.score * 0.85, 3), misses=t.misses + 1))

        out.sort(key=lambda d: -d.score)
        self._tracks = out
        return out


class DetectorSet:
    """一组检测器：合并、去重、平滑。"""

    def __init__(self, detectors: list[Detector], opt):
        self.detectors = detectors
        self.opt = opt
        self.tracker = Tracker(alpha=opt.box_smooth, keep=opt.box_keep)
        self.ms = 0.0

    @property
    def classes(self) -> list[str]:
        seen: dict[str, None] = {}
        for det in self.detectors:
            for name in det.labels:
                seen.setdefault(name, None)
        return list(seen)

    def infer(self, frame: np.ndarray) -> list[Detection]:
        t0 = time.perf_counter()
        pool: list[Detection] = []
        for det in self.detectors:
            pool.extend(det.infer(frame))
        self.ms = (time.perf_counter() - t0) * 1000.0
        return self.tracker.update(merge(pool, self.opt.max_det))

    def describe(self) -> list[dict]:
        return [{'name': d.name, 'classes': len(d.labels), 'size': d.size, 'ms': round(d.ms, 1)}
                for d in self.detectors]


class AsyncDetector:
    """
    后台线程跑检测：主循环只等深度，框异步刷新。

    调度是「最新一帧覆盖旧的」——检测跟不上就丢帧，不会积压。
    """

    def __init__(self, workers: DetectorSet):
        self.workers = workers
        self._cond = threading.Condition()
        self._job = None
        self._items: list[Detection] = []
        self._stop = False
        self._fail = 0
        self._thread = threading.Thread(target=self._loop, name='depthscene-detect', daemon=True)
        self._thread.start()

    def submit(self, frame: np.ndarray) -> None:
        with self._cond:
            self._job = frame
            self._cond.notify()

    def latest(self) -> list[Detection]:
        with self._cond:
            return list(self._items)

    def close(self) -> None:
        with self._cond:
            self._stop = True
            self._cond.notify()
        self._thread.join(timeout=1.0)

    def _loop(self) -> None:
        while True:
            with self._cond:
                while self._job is None and not self._stop:
                    self._cond.wait()
                if self._stop:
                    return
                frame, self._job = self._job, None

            try:
                items = self.workers.infer(frame)
            except Exception as err:  # noqa: BLE001
                self._fail += 1
                if self._fail <= 3:
                    print('检测线程出错：', err)
                continue

            with self._cond:
                self._items = items
