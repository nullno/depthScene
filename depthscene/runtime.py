"""ONNX Runtime 会话：后端挑选、线程预算、失败回退。"""

from __future__ import annotations

import onnxruntime as ort


def short_name(provider: str) -> str:
    return provider.replace('ExecutionProvider', '').lower()


def resolve_providers(pref: str = 'auto') -> list[str]:
    """按偏好挑执行后端；不支持的自动跳过，末尾带 CPU 兜底。"""
    avail = ort.get_available_providers()
    order = {
        'cpu': ['CPUExecutionProvider'],
        'cuda': ['CUDAExecutionProvider'],
        'dml': ['DmlExecutionProvider'],
        'auto': ['CUDAExecutionProvider', 'DmlExecutionProvider'],
    }.get(pref, [])

    picked = [p for p in order if p in avail]
    if not picked:
        if pref not in ('auto', 'cpu'):
            print(f'  未找到 {pref.upper()} 后端（可用：{", ".join(avail)}），使用 CPU')
        return ['CPUExecutionProvider']

    return list(dict.fromkeys([*picked, 'CPUExecutionProvider']))


def make_session(path, providers: list[str], threads: int = 0, label: str = ''):
    """
    建 session。目标 EP 建不起来（GPU 上 int8 算子很常见）就回退 CPU，
    保证换个机器、换个权重也不会直接起不来。
    """
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.intra_op_num_threads = max(0, int(threads))
    so.inter_op_num_threads = 1

    try:
        sess = ort.InferenceSession(path, so, providers=providers)
        used = sess.get_providers()[0]
        print(f'  {label}：{short_name(used).upper()}')
        return sess, used
    except Exception as err:  # noqa: BLE001
        if providers[0] == 'CPUExecutionProvider':
            raise
        print(f'  {label} 在 {short_name(providers[0]).upper()} 上建图失败：{err}')
        print('  → 回退 CPU（GPU 上 int8 算子常不支持，换 fp32 权重再试）')

    sess = ort.InferenceSession(path, so, providers=['CPUExecutionProvider'])
    return sess, 'CPUExecutionProvider'
