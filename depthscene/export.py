"""把 YOLO-World 导出成固定词表的 ONNX（可选步骤）。

YOLO-World 的文本嵌入在导出时会固化（reparameterize）进网络，产物是一个
**单输入的标准 YOLO**：运行时不需要 CLIP、tokenizer，也不需要联网。
代价是词表在导出时定死，换词表要重新导。

只有这一步需要 ultralytics：`pip install -e ".[export]"`。
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from . import models
from .vocab import resolve, zh_map


def build_world(vocab: str = 'full', imgsz: int = 640, opset: int = 12,
                weights: str | None = None, out_dir: str | None = None) -> Path:
    """导出到 models/yoloworld/model.onnx + labels.json，返回模型路径。"""
    try:
        from ultralytics import YOLOWorld
    except ImportError as err:
        raise SystemExit('需要先安装导出依赖：pip install -e ".[export]"\n' + str(err))

    words = resolve(vocab)
    if not words:
        raise SystemExit('词表为空')
    if len(words) > 300:
        print(f'⚠ {len(words)} 类偏多，单类精度会下滑、导出也更慢；建议 ≤200')

    print(f'词表（{len(words)} 类）：{", ".join(words)}')

    if not weights:
        local = models.ROOT / 'weights' / 'yolov8s-worldv2.pt'
        weights = str(local) if local.exists() else 'yolov8s-worldv2.pt'
        if not Path(weights).exists():
            print(f'  {weights} 不存在，交给 ultralytics 自动下载（约 25MB）')

    model = YOLOWorld(weights)

    print('写入文本嵌入（首次会下载 CLIP 文本编码器，约 340MB）…')
    model.set_classes(words)

    print(f'导出 ONNX（imgsz={imgsz}，opset={opset}）…')
    onnx_path = model.export(format='onnx', imgsz=imgsz, opset=opset, simplify=True)

    out = Path(out_dir) if out_dir else models.models_dir() / 'yoloworld'
    out.mkdir(parents=True, exist_ok=True)

    dst = out / 'model.onnx'
    shutil.move(str(onnx_path), dst)

    meta = {
        'vocab': words,
        'zh': zh_map(words),
        'imgsz': imgsz,
        'source': Path(str(weights)).name,
        'kind': 'yolo-world',
    }
    with open(out / 'labels.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f'\n✅ {dst}  ({dst.stat().st_size / 1048576:.0f}MB)')
    print(f'   {out / "labels.json"}  共 {len(words)} 类')
    print('   启动： python -m depthscene   （有这个词表就不再加载 YOLOv10）')
    return dst
