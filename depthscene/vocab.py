"""
开放词表预设
====================================================
YOLO-World 的类别在导出时固化（见 depthscene/export.py），这里准备两套词表：

    core  高频、识别稳、耗时几乎无感（29 类）
    full  覆盖人/身体部位/办公/家居/厨房/工具/街道（189 类）

用法：
    python -m depthscene --build-detect --vocab full
    python -m depthscene --build-detect --vocab "hand,pen,cup"
    python -m depthscene --build-detect --vocab my_words.txt   # 一行一个，或用逗号分隔

词表越大召回越广，但同义/近义类别会互相抢框（mug/cup、tv/monitor）；
DetectorSet 合并时会按 IoU 去重，剩一两个不影响观看。
超过 ~300 类后单类精度会明显下滑，不是越多越好。
"""

from __future__ import annotations

import os

# 英文类别 -> 中文名；同时作为 full 词表的唯一定义源
ZH: dict[str, str] = {
    # ---- 人 / 身体部位 ----
    'person': '人',
    'hand': '手',
    'finger': '手指',
    'head': '头',
    'face': '脸',
    'eye': '眼睛',
    'ear': '耳朵',
    'mouth': '嘴',
    'nose': '鼻子',
    'arm': '手臂',
    'leg': '腿',
    'foot': '脚',
    'hair': '头发',

    # ---- COCO 80 ----
    'bicycle': '自行车',
    'car': '汽车',
    'motorcycle': '摩托车',
    'airplane': '飞机',
    'bus': '公交车',
    'train': '火车',
    'truck': '卡车',
    'boat': '船',
    'traffic light': '红绿灯',
    'fire hydrant': '消防栓',
    'stop sign': '停车标志',
    'parking meter': '停车计时器',
    'bench': '长椅',
    'bird': '鸟',
    'cat': '猫',
    'dog': '狗',
    'horse': '马',
    'sheep': '羊',
    'cow': '牛',
    'elephant': '大象',
    'bear': '熊',
    'zebra': '斑马',
    'giraffe': '长颈鹿',
    'backpack': '背包',
    'umbrella': '雨伞',
    'handbag': '手提包',
    'tie': '领带',
    'suitcase': '行李箱',
    'frisbee': '飞盘',
    'skis': '双板滑雪板',
    'snowboard': '单板滑雪板',
    'sports ball': '球',
    'kite': '风筝',
    'baseball bat': '棒球棒',
    'baseball glove': '棒球手套',
    'skateboard': '滑板',
    'surfboard': '冲浪板',
    'tennis racket': '网球拍',
    'bottle': '瓶子',
    'wine glass': '高脚杯',
    'cup': '杯子',
    'fork': '叉子',
    'knife': '刀',
    'spoon': '勺子',
    'bowl': '碗',
    'banana': '香蕉',
    'apple': '苹果',
    'sandwich': '三明治',
    'orange': '橙子',
    'broccoli': '西兰花',
    'carrot': '胡萝卜',
    'hot dog': '热狗',
    'pizza': '披萨',
    'donut': '甜甜圈',
    'cake': '蛋糕',
    'chair': '椅子',
    'couch': '沙发',
    'potted plant': '盆栽',
    'bed': '床',
    'dining table': '餐桌',
    'toilet': '马桶',
    'tv': '电视',
    'laptop': '笔记本电脑',
    'mouse': '鼠标',
    'remote': '遥控器',
    'keyboard': '键盘',
    'cell phone': '手机',
    'microwave': '微波炉',
    'oven': '烤箱',
    'toaster': '烤面包机',
    'sink': '水槽',
    'refrigerator': '冰箱',
    'book': '书',
    'clock': '时钟',
    'vase': '花瓶',
    'scissors': '剪刀',
    'teddy bear': '毛绒熊',
    'hair drier': '吹风机',
    'toothbrush': '牙刷',

    # ---- 办公 / 电子 ----
    'monitor': '显示器',
    'tablet': '平板电脑',
    'printer': '打印机',
    'speaker': '音箱',
    'headphones': '耳机',
    'microphone': '麦克风',
    'camera': '相机',
    'desk': '书桌',
    'lamp': '台灯',
    'pen': '笔',
    'pencil': '铅笔',
    'marker': '马克笔',
    'notebook': '笔记本',
    'paper': '纸张',
    'stapler': '订书机',
    'tape': '胶带',
    'charger': '充电器',
    'cable': '数据线',
    'router': '路由器',
    'calculator': '计算器',
    'whiteboard': '白板',
    'envelope': '信封',
    'wallet': '钱包',
    'key': '钥匙',
    'coin': '硬币',
    'mug': '马克杯',
    'thermos': '保温杯',
    'tissue box': '纸巾盒',
    'trash can': '垃圾桶',
    'battery': '电池',
    'power strip': '插排',
    'light switch': '开关',
    'socket': '插座',

    # ---- 家居 / 厨房 ----
    'shelf': '架子',
    'cabinet': '柜子',
    'drawer': '抽屉',
    'curtain': '窗帘',
    'door': '门',
    'door handle': '门把手',
    'window': '窗户',
    'wall': '墙',
    'picture frame': '相框',
    'mirror': '镜子',
    'plate': '盘子',
    'teapot': '茶壶',
    'kettle': '水壶',
    'pot': '锅',
    'pan': '平底锅',
    'cutting board': '砧板',
    'stove': '灶台',
    'blender': '搅拌机',
    'rice cooker': '电饭煲',
    'chopsticks': '筷子',
    'jar': '罐子',
    'tomato': '番茄',
    'potato': '土豆',
    'onion': '洋葱',
    'garlic': '大蒜',
    'egg': '鸡蛋',
    'bread': '面包',
    'milk': '牛奶',
    'towel': '毛巾',
    'soap': '肥皂',
    'sponge': '海绵',
    'broom': '扫把',
    'mop': '拖把',
    'bucket': '水桶',
    'hanger': '衣架',
    'box': '盒子',
    'cardboard box': '纸箱',
    'package': '快递包裹',
    'plastic bag': '塑料袋',
    'shopping bag': '购物袋',
    'basket': '篮子',

    # ---- 工具 ----
    'screwdriver': '螺丝刀',
    'hammer': '锤子',
    'wrench': '扳手',
    'pliers': '钳子',
    'drill': '电钻',
    'ladder': '梯子',
    'rope': '绳子',

    # ---- 街道 / 户外 ----
    'traffic cone': '路锥',
    'barrier': '护栏',
    'pole': '杆子',
    'street light': '路灯',
    'stairs': '楼梯',
    'railing': '栏杆',
    'fence': '围栏',
    'poster': '海报',
    'scooter': '滑板车',
    'baby stroller': '婴儿车',
    'wheelchair': '轮椅',
    'shopping cart': '购物车',
    'balloon': '气球',
    'tree': '树',
    'flower': '花',
    'rock': '石头',
}

# 原版小词表：几乎不增加耗时，适合只想多认几种手边小物的场景
CORE = [
    'person', 'hand', 'head', 'face', 'pen', 'cup', 'mug', 'bottle', 'cell phone',
    'keyboard', 'mouse', 'laptop', 'monitor', 'book', 'scissors', 'chair', 'door',
    'window', 'shelf', 'backpack', 'wallet', 'key', 'notebook', 'paper', 'camera',
    'headphones', 'marker', 'pencil', 'toothbrush',
]

FULL = list(ZH)

PRESETS = {'core': CORE, 'full': FULL}


def resolve(spec: str) -> list[str]:
    """预设名 / 逗号列表 / 词表文件路径 → 去重后的类别列表。"""
    spec = (spec or '').strip()

    if spec.lower() in PRESETS:
        words = PRESETS[spec.lower()]
    elif os.path.isfile(spec):
        with open(spec, encoding='utf-8') as f:
            words = [w for line in f for w in line.replace('\n', ',').split(',')]
    else:
        words = spec.split(',')

    out, seen = [], set()
    for w in words:
        w = ' '.join(str(w).strip().lower().split())
        if not w or w in seen:
            continue
        seen.add(w)
        out.append(w)
    return out


def zh_map(words: list[str]) -> dict[str, str]:
    return {w: ZH[w] for w in words if w in ZH}
