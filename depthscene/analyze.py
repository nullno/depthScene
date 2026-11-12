"""避障分析：扇区距离、通行建议、障碍连通域、框内测距。

深度是相对量，`near`/`far` 只是一条标定映射，用来把 0–1 的深度网格
换成「米」量纲的读数，方便和画面上的框对得上，不是真实测距。
"""

from __future__ import annotations

import cv2
import numpy as np

from .detect import Detection


def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


class Analyzer:
    def __init__(self, opt):
        self.opt = opt

    # ---------------- 主入口 ----------------

    def analyze(self, grid: np.ndarray) -> dict:
        gh, gw = grid.shape
        near, far = self.opt.near, self.opt.far

        # 只看画面中段：最上面是天花板/天空，最下面是脚边，都不该算障碍
        band = grid[int(gh * 0.22):int(gh * 0.85)]
        dist = near + (far - near) * (1.0 - band)

        sectors = self._sectors(dist)
        left, center, right = self._thirds(sectors)
        action, text, code = self._decide(left, center, right)

        return {
            'sectors': sectors,
            'recommendation': {
                'action': action, 'text': text, 'code': code,
                'targetIndex': self._target(sectors, action),
                'clearance': {'left': round(left, 2), 'center': round(center, 2), 'right': round(right, 2)},
            },
            'threats': self._threats(dist),
            'nearest': {'distance': round(float(np.percentile(dist, 5)), 2)},
            'near': float(near),
            'far': float(far),
            'grid': [int(gw), int(gh)],
        }

    # ---------------- 扇区 ----------------

    def _sectors(self, dist: np.ndarray) -> list[dict]:
        gw = dist.shape[1]
        n = max(3, self.opt.sectors)
        edges = np.linspace(0, gw, n + 1).astype(int)

        sectors = []
        for i in range(n):
            seg = dist[:, edges[i]:edges[i + 1]]
            # 取最近 12% 像素，避免个别噪点把整段判成危险
            dmin = float(np.percentile(seg, 12)) if seg.size else float(self.opt.far)
            level = 'danger' if dmin <= self.opt.critical else 'warn' if dmin <= self.opt.warn else 'safe'
            sectors.append({
                'index': i,
                'center': round(float((edges[i] + edges[i + 1]) * 0.5 / gw), 4),
                'distance': round(dmin, 2),
                'clearance': round(float(clamp((dmin - self.opt.critical) /
                                               max(0.01, self.opt.warn - self.opt.critical), 0.0, 1.0)), 3),
                'level': level,
            })
        return sectors

    def _thirds(self, sectors: list[dict]) -> tuple[float, float, float]:
        n = len(sectors)
        third = max(1, n // 3)
        left = float(min(s['distance'] for s in sectors[:third]))
        center = float(min(s['distance'] for s in sectors[third:2 * third]))
        right = float(min(s['distance'] for s in sectors[2 * third:])) if n > 2 * third else center
        return left, center, right

    def _decide(self, left: float, center: float, right: float) -> tuple[str, str, str]:
        if center >= self.opt.warn:
            return 'forward', '前方通畅，可以直行', 'CLEAR'
        if center <= self.opt.critical and max(left, right) <= self.opt.critical:
            return 'stop', '前方严重受阻，建议停止', 'BLOCKED'
        if left > right:
            return 'left', '左侧空间更大，建议向左绕行', 'VEER_LEFT'
        return 'right', '右侧空间更大，建议向右绕行', 'VEER_RIGHT'

    def _target(self, sectors: list[dict], action: str) -> int:
        """指向该方向上最空的那一扇区，雷达图箭头按它画。"""
        n = len(sectors)
        third = max(1, n // 3)
        if action == 'left':
            pool = range(0, third)
        elif action == 'right':
            pool = range(2 * third, n)
        else:
            pool = range(third, 2 * third)
        pool = list(pool) or list(range(n))
        return int(max(pool, key=lambda i: sectors[i]['distance']))

    # ---------------- 障碍连通域 ----------------

    def _threats(self, dist: np.ndarray) -> list[dict]:
        mask = (dist < self.opt.warn).astype(np.uint8)
        if not mask.any():
            return []

        dh, dw = dist.shape
        sh = max(8, int(72 * dh / dw))
        small = cv2.resize(mask * 255, (72, sh), interpolation=cv2.INTER_AREA)
        small = (small > 90).astype(np.uint8)

        count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(small, connectivity=8)
        min_area = 0.012 * sh * 72

        out = []
        for i in range(1, count):
            x, y, bw, bh, area = (int(v) for v in stats[i])
            if area < min_area:
                continue

            sub = dist[int(y * dh / sh):max(int(y * dh / sh) + 1, int((y + bh) * dh / sh)),
                       int(x * dw / 72):max(int(x * dw / 72) + 1, int((x + bw) * dw / 72))]
            dmin = float(np.percentile(sub, 15)) if sub.size else float(self.opt.far)

            out.append({
                'x': round(x / 72, 4), 'y': round(y / sh, 4),
                'w': round(bw / 72, 4), 'h': round(bh / sh, 4),
                'area': area,
                'distance': round(dmin, 2),
            })

        out.sort(key=lambda t: -(t['w'] * t['h']))
        return out[:6]


def attach_distances(dets: list[Detection], grid: np.ndarray, near: float, far: float) -> None:
    """用同一帧的深度网格给检测框补距离（取框内较近的一批像素）。"""
    if not dets:
        return
    gh, gw = grid.shape

    for d in dets:
        x0 = int(clamp(d.x * gw, 0, gw - 1))
        x1 = max(x0 + 1, int(clamp((d.x + d.w) * gw, 1, gw)))
        y0 = int(clamp(d.y * gh, 0, gh - 1))
        y1 = max(y0 + 1, int(clamp((d.y + d.h) * gh, 1, gh)))

        # 远处的目标可能只占一两个格子，邻域扩一点再取分位数，免得单点噪声说了算
        if x1 - x0 < 3:
            x0, x1 = max(0, x0 - 1), min(gw, x1 + 2)
        if y1 - y0 < 3:
            y0, y1 = max(0, y0 - 1), min(gh, y1 + 2)

        sub = grid[y0:y1, x0:x1]
        if not sub.size:
            continue
        dn = float(np.percentile(sub, 85))
        d.distance = round(float(near + (far - near) * (1.0 - dn)), 2)
