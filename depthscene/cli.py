"""命令行入口。

    python -m depthscene                 # 启动 http://localhost:8000
    python -m depthscene --port 8080
    python -m depthscene --download      # 下载权重（深度 + YOLOv10）
    python -m depthscene --build-detect  # 导出 189 类开放词表模型
"""

from __future__ import annotations

import argparse
import sys

from . import models
from .options import Options


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog='depthscene', description='单目深度视觉避障（服务端推理）')

    g = ap.add_argument_group('权重')
    g.add_argument('--download', action='store_true',
                   help='只下载权重后退出（正常情况下启动时会自动下载）')
    g.add_argument('--no-download', action='store_true',
                   help='缺权重也不联网，直接报错退出')
    g.add_argument('--build-detect', action='store_true',
                   help='导出 YOLO-World 开放词表模型后退出（需要 pip install -e ".[export]"）')
    g.add_argument('--vocab', default='full', help='词表预设（full/core）、逗号列表或词表文件')
    g.add_argument('--imgsz', type=int, default=640, help='检测模型输入边长，800 更利于小目标')
    g.add_argument('--opset', type=int, default=12)
    g.add_argument('--weights', default=None, help='yolov8s-worldv2.pt；默认找 weights/ 或自动下载')
    g.add_argument('--out-dir', default=None, help='导出目录，默认 models/yoloworld')
    g.add_argument('--force', action='store_true', help='权重已存在也重新下载')
    g.add_argument('--only', choices=['depth', 'detect'], help='只下其中一个')
    g.add_argument('--mirror-host', default=models.HF_MIRROR, help='下载源')
    g.add_argument('--depth-variant', choices=['quantized', 'fp32'], default='quantized',
                   help='quantized 适合 CPU；GPU 上 fp32 通常更快')
    g.add_argument('--detect-variant', choices=['n', 's', 'm', 'b', 'x'], default='n',
                   help='YOLOv10 规模：n 最快，s/m 更准也更慢')

    s = ap.add_argument_group('服务')
    s.add_argument('--host', default='0.0.0.0')
    s.add_argument('--port', type=int, default=8000)
    s.add_argument('--provider', default='auto', choices=['auto', 'cpu', 'cuda', 'dml'],
                   help='执行后端；auto 优先 CUDA 再 DirectML，失败自动回退 CPU')

    m = ap.add_argument_group('模型')
    m.add_argument('--depth-model', default=None, help='默认 models/depth/model.onnx')
    m.add_argument('--detect-model', default=None,
                   help='逗号分隔的检测模型；默认 YOLO-World，没有才用最大的 YOLOv10')

    d = ap.add_argument_group('深度 / 避障')
    d.add_argument('--depth-input', type=int, default=224, help='深度推理短边，越小越快')
    d.add_argument('--target-ms', type=float, default=200.0, help='自适应调节的目标单帧耗时')
    d.add_argument('--no-adaptive', action='store_true', help='固定 --depth-input，不自动调节')
    d.add_argument('--near', type=float, default=0.4, help='最近距离映射')
    d.add_argument('--far', type=float, default=6.0, help='最远距离映射')
    d.add_argument('--sectors', type=int, default=24, help='扇区数')
    d.add_argument('--warn', type=float, default=2.6, help='小于它算 warn')
    d.add_argument('--critical', type=float, default=1.0, help='小于它算 danger')

    t = ap.add_argument_group('检测')
    t.add_argument('--detect-every', type=int, default=2, help='每 N 帧跑一次检测（1 = 每帧都跑）')
    t.add_argument('--detect-threshold', type=float, default=0.30, help='置信度阈值，调低召回更多')
    t.add_argument('--max-det', type=int, default=30, help='最多返回多少个框')
    t.add_argument('--no-async', action='store_true', help='检测不并行（默认放后台线程）')
    t.add_argument('--no-smooth', action='store_true', help='关闭框的时序平滑')
    t.add_argument('--threads', type=int, default=0, help='每个模型的计算线程数，0 = 自动')

    o = ap.add_argument_group('其它')
    o.add_argument('--no-preview', action='store_true', help='不返回深度缩略图')
    o.add_argument('--mirror', action='store_true', help='服务端水平翻转（浏览器侧已翻转时不要开）')
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.download:
        failed = models.download(
            host=args.mirror_host, only=args.only, force=args.force,
            depth_variant=args.depth_variant, detect_variant=args.detect_variant,
        )
        return 1 if failed else 0

    if args.build_detect:
        from .export import build_world

        build_world(vocab=args.vocab, imgsz=args.imgsz, opset=args.opset,
                    weights=args.weights, out_dir=args.out_dir)
        return 0

    opt = Options(
        depth_input=args.depth_input,
        adaptive=not args.no_adaptive,
        target_ms=args.target_ms,
        detect_every=max(1, args.detect_every),
        det_threshold=args.detect_threshold,
        max_det=max(1, args.max_det),
        async_detect=not args.no_async,
        box_smooth=0.0 if args.no_smooth else 0.45,
        depth_threads=args.threads,
        detect_threads=args.threads,
        near=args.near,
        far=args.far,
        sectors=max(3, args.sectors),
        warn=args.warn,
        critical=args.critical,
        preview=not args.no_preview,
        mirror=args.mirror,
    )

    from .pipeline import DepthScene
    from .server import serve

    try:
        models.ensure(allow_download=not args.no_download, host=args.mirror_host)
        print('加载模型…')
        scene = DepthScene(opt, args.depth_model, args.detect_model, args.provider)
    except FileNotFoundError as err:
        print(err)
        return 1

    for info in scene.detectors.describe():
        print(f"  检测模型：{info['name']}（{info['classes']} 类，{info['size']}px）")
    print(f'  推理后端：{scene.device.upper()}')
    print(f'  可识别类别：{len(scene.labels)} 类')
    print(f"  检测并行：{'后台线程' if scene.async_detector else '同步'} · 每 {opt.detect_every} 帧一次")

    if not models.models_dir().joinpath('yoloworld', 'model.onnx').exists():
        print('  提示：现在是 COCO 80 类；想要 189 类开放词表 → python -m depthscene --build-detect')

    try:
        serve(scene, args.host, args.port)
    finally:
        scene.close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
