"""运行时配置：库调用和命令行共用同一份 Options。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Options:
    # ---------------- 深度 ----------------
    depth_input: int = 224        # 推理短边，越小越快
    adaptive: bool = True         # 按实测耗时自动调 depth_input
    target_ms: float = 220.0      # 自适应目标：单帧深度耗时
    depth_min: int = 168          # 再小深度就糊成一块，测出来的距离不能信
    depth_max: int = 336
    grid_width: int = 160         # 分析用深度网格宽度
    depth_smooth: float = 0.0     # 归一化统计的时序平滑；0 = 逐帧（刻度对分辨率不敏感，最稳）
    adaptive_settle: int = 12     # 前 N 帧校准分辨率，之后锁定（中途改尺寸会让读数整体平移）
    depth_threads: int = 0        # 0 = 由 ORT 决定

    # ---------------- 检测 ----------------
    detect_every: int = 2         # 每 N 帧跑一次检测（深度仍每帧算）
    det_threshold: float = 0.30   # 大词表下 0.30 是召回/误检的折中
    max_det: int = 30             # 合并后最多返回多少个框
    async_detect: bool = True     # 检测放后台线程，和下一帧的深度重叠
    detect_threads: int = 0       # 0 = 异步时自动留 1/4 核给检测，同步时用全部
    box_smooth: float = 0.45      # 框的时序平滑系数（0 = 关）
    box_keep: int = 2             # 漏检后最多保留几次，避免框闪烁

    # ---------------- 避障 ----------------
    near: float = 0.4             # 最近距离映射（相对深度，不是真实测距）
    far: float = 6.0
    sectors: int = 24
    warn: float = 2.6             # <= 它算 warn
    critical: float = 1.0         # <= 它算 danger

    # ---------------- 其它 ----------------
    preview: bool = True          # 是否返回深度缩略图（关闭可省编码时间）
    mirror: bool = False          # 浏览器侧已翻转，这里兜底给非浏览器调用
