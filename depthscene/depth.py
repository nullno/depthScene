"""单目深度估计（Depth-Anything-V2-Small）。

输出是一张归一化深度网格（1 = 近，0 = 远），
既是避障分析的数据源，也用来给检测框补距离。
"""

from __future__ import annotations

import time

import cv2
import numpy as np

from .runtime import make_session

# ImageNet 归一化，折叠成一次乘加：x / (255*std) - mean/std
_ALPHA = (1.0 / (255.0 * np.array([0.229, 0.224, 0.225], np.float32))).reshape(1, 1, 3)
_BETA = (-np.array([0.485, 0.456, 0.406], np.float32) / np.array([0.229, 0.224, 0.225], np.float32)).reshape(1, 1, 3)


class DepthEstimator:
    def __init__(self, path, providers: list[str], opt):
        self.opt = opt
        self.sess, self.ep = make_session(path, providers, opt.depth_threads, '深度模型')
        self.input = self.sess.get_inputs()[0].name
        self.path = str(path)
        self.ms = 0.0
        self._stat = None      # 归一化统计的滑动值 (lo, hi)
        self._ema = None       # 耗时滑动平均
        self._hold = 0         # 连续偏出目标的次数
        self._calls = 0        # 推理次数（用于 warmup / 校准期）
        self._locked = False   # 校准结束后锁定分辨率

    # ---------------- 尺寸 ----------------

    def _shape(self, w: int, h: int) -> tuple[int, int]:
        """Depth-Anything 要求边长是 14 的倍数。"""
        short, p = self.opt.depth_input, 14
        if w >= h:
            th = max(p, round(short / p) * p)
            tw = max(p, round(short * w / h / p) * p)
        else:
            tw = max(p, round(short / p) * p)
            th = max(p, round(short * h / w / p) * p)
        return tw, th

    def _tune(self, ms: float) -> None:
        """
        校准阶段按实测耗时挑一个跟得上的输入尺寸，挑完就锁死。

        分辨率一动，相对深度的刻度就整体平移（同一个物体的读数会差一截），
        所以只在开头十几帧校准；前两帧是 warmup，耗时不代表稳态，跳过。
        死区 + 连续两次确认，避免被偶发抖动带到最低分辨率。
        """
        self._calls += 1
        if not self.opt.adaptive or self._locked or ms <= 5:
            return
        if self._calls <= 2:
            return
        if self.opt.adaptive_settle and self._calls > self.opt.adaptive_settle:
            self._locked = True
            return

        self._ema = ms if self._ema is None else self._ema * 0.65 + ms * 0.35

        raw = (self.opt.target_ms / self._ema) ** 0.5
        if 0.9 <= raw <= 1.1:
            self._hold = 0
            return

        self._hold += 1
        if self._hold < 2:
            return
        self._hold = 0

        cur = self.opt.depth_input
        nxt = int(round(cur * min(1.18, max(0.85, raw)) / 14) * 14)
        nxt = int(min(self.opt.depth_max, max(self.opt.depth_min, nxt)))
        if abs(nxt - cur) >= 14:
            self.opt.depth_input = nxt

    # ---------------- 推理 ----------------

    def infer(self, frame: np.ndarray) -> np.ndarray:
        """BGR 帧 → 归一化深度网格 (gw, gh)，1 = 近。"""
        h, w = frame.shape[:2]
        tw, th = self._shape(w, h)

        x = cv2.resize(frame, (tw, th), interpolation=cv2.INTER_AREA)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB).astype(np.float32)
        x *= _ALPHA
        x += _BETA
        x = np.ascontiguousarray(np.transpose(x, (2, 0, 1))[None])

        t0 = time.perf_counter()
        out = self.sess.run(None, {self.input: x})[0]
        self.ms = (time.perf_counter() - t0) * 1000.0
        self._tune(self.ms)

        return self.to_grid(np.squeeze(out), w, h)

    def to_grid(self, depth: np.ndarray, w: int, h: int) -> np.ndarray:
        """
        深度图 → [0,1] 网格，1 = 近。

        分位数统计做时序平滑：单帧分位数会被前景占比带着跳，
        平滑之后同一面墙的距离读数不会每秒抖半米，扇区等级也不会闪。
        """
        d = depth.astype(np.float32)
        lo, hi = (float(v) for v in np.percentile(d, (2.0, 98.0)))

        k = min(1.0, max(0.0, self.opt.depth_smooth))
        if self._stat is None or k <= 0:
            self._stat = (lo, hi)
        else:
            self._stat = (self._stat[0] + k * (lo - self._stat[0]),
                          self._stat[1] + k * (hi - self._stat[1]))
        lo, hi = self._stat

        center = (lo + hi) * 0.5
        span = max(hi - lo, abs(center) * 0.06, 1e-6)
        dn = np.clip((d - center) / span + 0.5, 0.0, 1.0)

        gw = self.opt.grid_width
        gh = max(16, int(round(gw * h / w)))
        return cv2.resize(dn, (gw, gh), interpolation=cv2.INTER_AREA)
