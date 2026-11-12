"""权重查找与下载。

查找顺序：`DEPTHSCENE_MODELS` 环境变量 → 当前目录的 `models/` → 项目根目录的 `models/`。
下载走 hf-mirror（可用 `--host` 换源），因为浏览器直接拉会被 CORS 拦住。
"""

from __future__ import annotations

import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HF_MIRROR = 'https://hf-mirror.com'

DEPTH_REL = ('depth', 'model.onnx')
WORLD_REL = ('yoloworld', 'model.onnx')
V10_ORDER = ('yolov10x', 'yolov10m', 'yolov10b', 'yolov10s', 'yolov10n')


def models_dir() -> Path:
    env = os.environ.get('DEPTHSCENE_MODELS')
    if env:
        return Path(env).expanduser()
    for base in (Path.cwd(), ROOT):
        cand = base / 'models'
        if cand.is_dir():
            return cand
    return ROOT / 'models'


def _search_bases() -> list[Path]:
    """DEPTHSCENE_MODELS 一旦设置就以它为准，不再回退到项目里的 models/。"""
    if os.environ.get('DEPTHSCENE_MODELS'):
        return [models_dir()]
    return list(dict.fromkeys([models_dir(), ROOT / 'models']))


def resolve(path) -> Path:
    """'models/depth/model.onnx' / 绝对路径 / 相对路径 → 实际存在的路径。"""
    p = Path(path).expanduser()
    if p.is_absolute():
        return p

    cands = [Path.cwd() / p, ROOT / p]
    if p.parts and p.parts[0] == 'models':
        cands.append(models_dir().joinpath(*p.parts[1:]))
    for c in cands:
        if c.exists():
            return c
    return cands[0]


def find_depth_model() -> Path | None:
    for base in _search_bases():
        p = base.joinpath(*DEPTH_REL)
        if p.exists():
            return p
    return None


def find_detect_models(explicit=None) -> list[Path]:
    """
    默认优先 YOLO-World（开放词表，已覆盖 COCO 80 类），没有才退回最大的 YOLOv10。
    explicit 可以是逗号分隔的字符串、单个路径或路径列表。
    """
    if explicit:
        parts = explicit.split(',') if isinstance(explicit, str) else list(explicit)
        out = []
        for part in parts:
            part = str(part).strip()
            if not part:
                continue
            p = resolve(part)
            if p.exists():
                out.append(p)
            else:
                print(f'  忽略不存在的检测模型：{part}')
        return out

    bases = _search_bases()
    for base in bases:
        world = base.joinpath(*WORLD_REL)
        if world.exists():
            return [world]

    for base in bases:
        for name in V10_ORDER:
            p = base / name / 'model.onnx'
            if p.exists():
                return [p]

    return []


# ---------------- 下载 ----------------

TARGETS = {
    'depth': ('onnx-community/depth-anything-v2-small', 'onnx/model_quantized.onnx', DEPTH_REL),
    'detect': ('onnx-community/yolov10n', 'onnx/model.onnx', ('yolov10n', 'model.onnx')),
}


def _download(url: str, out: Path) -> int:
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + '.part')

    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=60) as res:
        total = int(res.headers.get('Content-Length') or 0)
        loaded = 0
        with open(tmp, 'wb') as f:
            while True:
                chunk = res.read(262144)
                if not chunk:
                    break
                f.write(chunk)
                loaded += len(chunk)
                if total:
                    sys.stdout.write(f'\r  下载中 {int(loaded * 100 / total):3d}%  {loaded / 1048576:.1f}MB')
                    sys.stdout.flush()
    print()
    os.replace(tmp, out)
    return loaded


def download(host: str = HF_MIRROR, only: str | None = None, force: bool = False,
             depth_variant: str = 'quantized', detect_variant: str = 'n') -> int:
    """下载深度 / 检测权重，返回失败个数。"""
    host = host.rstrip('/')
    targets = dict(TARGETS)

    if depth_variant == 'fp32':
        targets['depth'] = (TARGETS['depth'][0], 'onnx/model.onnx', DEPTH_REL)
    if detect_variant != 'n':
        targets['detect'] = (
            f'onnx-community/yolov10{detect_variant}',
            'onnx/model.onnx',
            (f'yolov10{detect_variant}', 'model.onnx'),
        )

    keys = [only] if only else ['depth', 'detect']
    failed = 0

    for key in keys:
        repo, file, rel = targets[key]
        out = models_dir().joinpath(*rel)
        if out.exists() and not force:
            print(f'✓ {out} 已存在（{out.stat().st_size / 1048576:.1f}MB），跳过')
            continue

        print(f'↓ {repo}/{file}')
        try:
            size = _download(f'{host}/{repo}/resolve/main/{file}', out)
            print(f'✓ {out}  ({size / 1048576:.1f}MB)')
        except Exception as err:  # noqa: BLE001
            failed += 1
            print(f'✗ 失败：{err}')
            print('  可换源：python -m depthscene --download --host https://huggingface.co')

    return failed


def missing() -> list[str]:
    """缺哪些必需权重：'depth' / 'detect'。"""
    need = []
    if not find_depth_model():
        need.append('depth')
    if not find_detect_models():
        need.append('detect')
    return need


def ensure(allow_download: bool = True, host: str = HF_MIRROR,
           depth_variant: str = 'quantized', detect_variant: str = 'n') -> None:
    """
    一条命令能直接跑起来的关键：缺权重就补，补不上再说人话。

    只补最小集合（深度 + YOLOv10n，共 35MB）——开放词表的 189 类模型要本地导出，
    那是可选步骤，不在启动路径上偷偷下载几百 MB。
    """
    need = missing()
    if not need:
        return

    if not allow_download:
        raise FileNotFoundError(
            f'缺少权重：{" / ".join(need)}（{models_dir()}）\n'
            '去掉 --no-download 让程序自动下载，或先运行 python -m depthscene --download')

    print(f'首次运行：缺少 {" / ".join(need)} 权重，开始自动下载…')
    failed = 0
    for key in need:
        failed += download(host=host, only=key,
                           depth_variant=depth_variant, detect_variant=detect_variant)

    left = missing()
    if failed or left:
        raise SystemExit(f'权重下载失败（缺 {" / ".join(left)}），检查网络后重试，'
                         f'或换源：python -m depthscene --download --mirror-host https://huggingface.co')

