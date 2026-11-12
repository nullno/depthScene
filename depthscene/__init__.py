"""
DepthScene —— 单目深度视觉避障
====================================================
一帧画面进，结构化结果出：深度估计 → 目标检测 → 避障分析。

    from depthscene import DepthScene

    ds = DepthScene()                       # 自动找 models/ 下的权重
    result = ds.process(jpeg_bytes)         # 或 ds.process_image(bgr_ndarray)
    result['recommendation']                # {'action': 'left', ...}

命令行（一条命令搞定，缺权重会自己下）：

    python -m depthscene                    # 启动 http://localhost:8000
    python -m depthscene --build-detect     # 可选：换成 189 类开放词表

`DepthScene` / `Detection` 是惰性导入：只 import 包本身不会拉起 OpenCV 和 ONNX Runtime，
导出脚本这类只用 `depthscene.vocab` 的地方不会平白多几百毫秒启动时间。
库调用不会自动下载权重（行为可预期），自动下载只发生在命令行入口。
"""

from .options import Options

__version__ = '0.3.0'
__all__ = ['DepthScene', 'Options', 'Detection', '__version__']


def __getattr__(name: str):
    if name == 'DepthScene':
        from .pipeline import DepthScene

        return DepthScene
    if name == 'Detection':
        from .detect import Detection

        return Detection
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
