"""对外门面：一帧进，结构化结果出。"""

from __future__ import annotations

import base64
import os
import threading

import cv2
import numpy as np

from . import models
from .analyze import Analyzer, attach_distances
from .depth import DepthEstimator
from .detect import AsyncDetector, Detector, DetectorSet
from .options import Options
from .runtime import resolve_providers, short_name


class DepthScene:
    """
    深度 + 检测 + 避障的完整管线。

        ds = DepthScene(Options(detect_every=2))
        state = ds.process(jpeg_bytes)      # 或 ds.process_image(bgr)

    process 是线程安全的：上一次还在算时，后来的调用直接拿到上次结果，
    所以客户端可以放心按 10fps 推帧，服务端不会堆任务。
    """

    def __init__(self, options: Options | None = None, depth_model=None, detect_models=None,
                 provider: str = 'auto'):
        self.opt = options or Options()
        self.providers = resolve_providers(provider)

        depth_path = models.resolve(depth_model) if depth_model else models.find_depth_model()
        if not depth_path or not depth_path.exists():
            raise FileNotFoundError(
                f'缺少深度模型：{depth_path or models.models_dir() / "depth" / "model.onnx"}\n'
                '运行 python -m depthscene（会自动下载），或调 depthscene.models.download()')

        det_paths = models.find_detect_models(detect_models)
        if not det_paths:
            raise FileNotFoundError(
                '缺少检测模型\n运行 python -m depthscene（会自动下载），或调 depthscene.models.download()')

        self.depth = DepthEstimator(depth_path, self.providers, self.opt)

        # 异步时两个模型同时在跑：给检测留 1/4 的核，
        # 免得它把深度的线程全抢走——深度一被拖慢，自适应会降分辨率，测距就飘了
        threads = self.opt.detect_threads
        if not threads and self.opt.async_detect:
            threads = max(1, (os.cpu_count() or 4) // 4)

        detectors = [Detector(p, self.providers, self.opt.det_threshold, threads) for p in det_paths]
        self.detectors = DetectorSet(detectors, self.opt)
        self.async_detector = AsyncDetector(self.detectors) if self.opt.async_detect else None
        self.analyzer = Analyzer(self.opt)

        eps = [self.depth.ep] + [d.ep for d in detectors]
        self.device = '+'.join(dict.fromkeys(short_name(e) for e in eps))
        self.labels = self.detectors.classes

        self._lock = threading.Lock()
        self._tick = 0
        self._dets = []
        self.last_state: dict = {}
        self.count = 0

    # ---------------- 对外 ----------------

    def process(self, jpeg: bytes) -> dict:
        """JPEG 字节 → 结果字典。解析失败返回上一次结果。"""
        frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return self.last_state
        return self.process_image(frame)

    def process_image(self, frame: np.ndarray) -> dict:
        """BGR ndarray → 结果字典。"""
        if not self._lock.acquire(blocking=False):
            return self.last_state

        try:
            if self.opt.mirror:
                frame = cv2.flip(frame, 1)
            self.last_state = self._infer(frame)
            self.count += 1
            return self.last_state
        except Exception as err:  # noqa: BLE001
            print('推理失败：', err)
            return self.last_state
        finally:
            self._lock.release()

    def close(self) -> None:
        if self.async_detector:
            self.async_detector.close()
            self.async_detector = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ---------------- 内部 ----------------

    def _infer(self, frame: np.ndarray) -> dict:
        grid = self.depth.infer(frame)
        dets = self._detect(frame)
        attach_distances(dets, grid, self.opt.near, self.opt.far)

        state = self.analyzer.analyze(grid)
        state['detections'] = [d.as_dict() for d in dets]
        state['depthMs'] = round(float(self.depth.ms), 1)
        state['detectMs'] = round(float(self.detectors.ms), 1)
        state['depthInput'] = int(self.opt.depth_input)
        state['device'] = self.device
        state['count'] = self.count + 1
        state['models'] = self.detectors.describe()
        state['classes'] = len(self.labels)
        if self.opt.preview:
            state['depth'] = _preview(grid)
        return state

    def _detect(self, frame: np.ndarray):
        self._tick += 1
        every = max(1, self.opt.detect_every)

        if self.async_detector:
            if (self._tick - 1) % every == 0:
                self.async_detector.submit(frame)
            return self.async_detector.latest()

        if (self._tick - 1) % every == 0 or not self._dets:
            self._dets = self.detectors.infer(frame)
        return self._dets


def _preview(grid: np.ndarray) -> str:
    """深度缩略图（1 = 近），前端直接当 <img src> 用。

    用 JPEG 而不是 PNG：同一张渐变图 6KB vs 24KB，每帧都要发，省带宽也省编码时间。
    """
    small = cv2.resize(1.0 - grid, (160, 90), interpolation=cv2.INTER_AREA)
    colored = cv2.applyColorMap((small * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    ok, buf = cv2.imencode('.jpg', colored, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
    if not ok:
        return ''
    return 'data:image/jpeg;base64,' + base64.b64encode(buf.tobytes()).decode()
