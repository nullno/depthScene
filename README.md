# DepthScene

单目深度视觉避障。**摄像头在浏览器采集**（`getUserMedia`），**推理在 Python 侧**（ONNX Runtime + OpenCV），
核心是一个可以 `import` 的包，页面只负责画。

```
浏览器                                      Python（depthscene）
┌─────────────────────┐   每 150ms 一帧   ┌──────────────────────────────┐
│ getUserMedia        │ ──── JPEG ────▶  │ 深度 Depth-Anything-V2  ┐     │
│ <canvas> 60fps 预览 │ ◀─── JSON ─────  │                        ├ 并行 │
│ 叠加框/扇区/深度图   │                   │ 检测 YOLO-World / v10   ┘     │
└─────────────────────┘                   │ 避障分析（扇区/连通域）        │
                                          └──────────────────────────────┘
```

画面由浏览器自己渲染（60fps，不受推理影响），服务端只回结构化数据；推理慢了只是叠加层滞后。

## 开始

```bash
pip install -r requirements.txt
python -m depthscene
```

就这样。第一次运行发现没有权重会自己下（深度 26MB + 检测 9MB），下完直接启动 → `http://localhost:8000`，
浏览器打开允许摄像头即可（镜像按钮默认开启）。

权重不进仓库（`models/`、`weights/` 都在 `.gitignore` 里），所以换台机器、换个目录，都还是这一条命令。
离线环境想跳过下载就加 `--no-download`，缺权重会直接告诉你缺什么。

### 可选：换成 189 类开放词表

默认检测器是 YOLOv10n（COCO 80 类）。想要手、笔、门、垃圾桶这些 COCO 之外的类别，跑一次导出：

```bash
pip install -e ".[export]"          # ultralytics + onnx + onnxslim，约 1 分钟
python -m depthscene --build-detect # 导出 189 类到 models/yoloworld，约 1–2 分钟
python -m depthscene                # 之后启动会自动优先用它
```

| 动作 | 体积 | 是否必需 |
| --- | --- | --- |
| 首次启动自动下载 | 35MB | 是 |
| `pip install -e ".[export]"` | ≈400MB | 否，只在导出词表时用 |
| `--build-detect` | 49MB | 否，换来 189 类而不是 80 类 |

## 作为库使用

```python
from depthscene import DepthScene, Options

ds = DepthScene(Options(detect_every=2))

state = ds.process(jpeg_bytes)          # 或 ds.process_image(bgr_ndarray)
state['recommendation']                 # {'action': 'left', 'clearance': {...}, 'targetIndex': 7}
state['detections']                     # [{'label': 'cup', 'labelZh': '杯子', 'distance': 0.8, ...}]
ds.close()                              # 或 with DepthScene() as ds: ...
```

常用参数（完整列表见 `depthscene/options.py`）：

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `depth_input` / `adaptive` | `224` / `True` | 深度推理短边；自适应只在前 12 帧校准，之后锁定 |
| `detect_every` | `2` | 每 N 帧跑一次检测（深度仍每帧算） |
| `det_threshold` | `0.30` | 置信度阈值，调低召回更多 |
| `async_detect` | `True` | 检测放后台线程，和下一帧深度重叠 |
| `box_smooth` / `box_keep` | `0.45` / `2` | 框的时序平滑与漏检保留 |
| `near` / `far` | `0.4` / `6.0` | 相对深度到「米」的标定映射 |

权重默认按 `DEPTHSCENE_MODELS` → `./models` → 项目根 `models/` 的顺序查找（设了环境变量就只认它）。
库调用**不会**偷偷下载，缺权重会抛 `FileNotFoundError`；要自动补齐就显式调一次：

```python
from depthscene import models
models.ensure()                 # 缺什么下什么，下不动会给出可读的提示
```

## 命令行

```bash
python -m depthscene                        # 启动服务（缺权重自动下）
python -m depthscene --port 8080 --provider dml
python -m depthscene --build-detect         # 换词表：导出检测模型后退出
python -m depthscene --detect-every 1       # 检测跑得更勤（框更跟手，CPU 更忙）
python -m depthscene --no-async             # 回到「深度→检测」串行
```

`pip install -e .` 之后还可以直接用 `depthscene` 命令。

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--host` / `--port` | `0.0.0.0` / `8000` | 监听地址 |
| `--provider` | `auto` | `auto` / `cpu` / `cuda` / `dml`，失败自动回退 CPU |
| `--no-download` | 关 | 缺权重时不联网，直接报错（离线用） |
| `--download` | 关 | 只预下载权重后退出（`--only` / `--force` / `--mirror-host`） |
| `--build-detect` | 关 | 只导出开放词表模型后退出，需 `.[export]` |
| `--vocab` / `--imgsz` / `--weights` | `full` / `640` / 自动 | 导出用：词表、输入边长、原始 .pt |
| `--depth-model` / `--detect-model` | 自动 | 逗号分隔可挂多个检测模型 |
| `--depth-input` / `--target-ms` / `--no-adaptive` | `224` / `220` / 关 | 深度分辨率与自适应 |
| `--detect-every` / `--detect-threshold` / `--max-det` | `2` / `0.30` / `30` | 检测节流、阈值、框上限 |
| `--no-async` / `--no-smooth` / `--threads` | 关 / 关 / `0` | 串行检测、关闭框平滑、线程数 |
| `--near` / `--far` / `--sectors` / `--warn` / `--critical` | `0.4` / `6.0` / `24` / `2.6` / `1.0` | 避障映射与分级 |
| `--no-preview` / `--mirror` | 关 / 关 | 不返回深度缩略图、服务端翻转 |

## 接口

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/` | 控制台页面 |
| POST | `/frame` | 请求体是一帧 JPEG，返回结构化结果 |
| GET | `/state` | 最近一次结果（兜底用） |

```json
{
  "sectors": [{ "index": 0, "distance": 4.27, "clearance": 1, "level": "safe" }],
  "recommendation": {
    "action": "right", "text": "右侧空间更大，建议向右绕行", "code": "VEER_RIGHT",
    "targetIndex": 22,
    "clearance": { "left": 0.76, "center": 0.41, "right": 1.25 }
  },
  "threats": [{ "x": 0.19, "y": 0.15, "w": 0.18, "h": 0.76, "distance": 0.42 }],
  "detections": [{
    "label": "cat", "labelZh": "猫", "score": 0.769,
    "x": 0.19, "y": 0.22, "w": 0.73, "h": 0.65, "src": "yoloworld", "distance": 0.79
  }],
  "nearest": { "distance": 0.49 },
  "classes": 189,
  "models": [{ "name": "yoloworld", "classes": 189, "size": 640, "ms": 306.4 }],
  "depth": "data:image/jpeg;base64,...",
  "depthMs": 217.0, "detectMs": 306.4, "depthInput": 168, "device": "cpu", "count": 12
}
```

`level`：`safe` / `warn` / `danger`；`action`：`forward` / `left` / `right` / `stop`。
`targetIndex` 是建议方向上最空的那一扇区，雷达图箭头按它画。
同一时刻只跑一次推理：上一次还在算时，`/frame` 直接返回上次结果。

## 算法

### 深度

- **自适应分辨率，校准完就锁死**：开头 12 帧（前 2 帧是 warmup）按实测耗时把短边调到 168–336 之间，
  之后固定不变。分辨率一动，相对深度的刻度就整体平移（同一物体读数能差一倍），
  中途反复调会让「距离」自己漂，所以宁可只校准一次。校准期还有 ±20% 死区和「连续两次偏才动」。
- **逐帧分位数归一化**（2% / 98%）：把 Depth-Anything 的仿射输出压到 0–1，`1 = 近`。
  逐帧算而不是做时序平滑：刻度自洽，同一画面连续帧的距离抖动实测 σ≈0.02m。
- **下限 168px**：再小深度就糊成一块，测出来的距离不能信；宁可慢一点。
  想要更精细的深度可以用 `--target-ms 400`，让它往 224–280 校准。

### 检测

- **开放词表 + 固定 ONNX**：词表在导出时由 CLIP 文本嵌入 reparameterize 进检测头，运行时不需要 CLIP，
  189 类覆盖人/身体部位/办公/家居/厨房/工具/街道，COCO 80 类全在里面。
- **两种输出头都能吃**：`[1,300,6]`（YOLOv10 端到端，已含 NMS）和 `[1,4+N,8400]`（原始头，自己解 cxcywh + NMS）。
- **合并去重**：同类 IoU ≥ 0.5、异类（`mug`/`cup` 这种近义）IoU ≥ 0.7 才算重复，避免把「人手里的杯子」抹掉。
- **时序平滑**：按 IoU 关联上一轮结果，坐标/分数做 EMA；漏检保留两轮再丢，远处目标不会闪一下就没。
- **后台线程**：检测在独立线程跑，主循环只等深度，第 N 帧送到最新一帧（覆盖式，不排队）。
  代价是检测只能分到 1/4 的核，所以耗时变长；换来的是每帧响应快一倍、尾延迟收敛。

### 避障

- 只看画面中段（22%–85% 高度），顶部天空/底部脚边不算障碍。
- 每个扇区取最近 12% 像素的距离，避免个别噪点把整段判成危险。
- 三分区（左/中/右）取各自最近扇区，中区通畅就直行，两侧选更空的一侧绕行，都堵住就停。
- 障碍连通域在 72×N 粗网格上做 8 邻域标记，输出包围盒 + 框内距离。
- 检测框的距离取框内 85% 分位（偏近的一批像素），框太小时邻域扩到 3×3 再算。

## 性能（i5-10210U，4 核 8 线程，纯 CPU）

| 环节 | 单独跑 | 与检测并行 |
| --- | --- | --- |
| 深度 168px | 148ms | ≈270ms |
| 深度 224px | 273ms | — |
| 深度 336px | 704ms | — |
| 检测（640，189 类，4 线程） | 250–350ms | — |
| 检测（640，189 类，异步限 2 线程） | — | ≈600ms |

两个模型抢同一批核，并行时各自都变慢，但**响应只等深度**，所以整体更快
（同一张 960×686 帧，30 次平均）：

| 模式 | 平均 | P90 |
| --- | --- | --- |
| 串行（`--no-async`） | 450ms | 758ms |
| **异步（默认）** | **322ms** | **402ms** |

HTTP 端到端（含 JPEG 上传/解码和 JSON 序列化）实测 282ms / P90 355ms。
避障层每帧刷新，检测框刷新约 0.6–1.2s 一次。
想更快：`--depth-input 168 --no-adaptive`（深度固定 148ms）、`--no-preview`。

## 换词表

词表在导出时固化，换词要重新导一次：

```bash
pip install -e ".[export]"
python -m depthscene --build-detect --vocab full            # 189 类（默认）
python -m depthscene --build-detect --vocab core            # 29 类，最省时
python -m depthscene --build-detect --vocab "hand,pen,cup"
python -m depthscene --build-detect --vocab my_words.txt    # 一行一个，或逗号分隔
python -m depthscene --build-detect --imgsz 800             # 更利于小目标，耗时 ×1.5
```

导出会把中文名写进 `models/yoloworld/labels.json`，页面直接显示「马克杯 / 门 / 垃圾桶」。
词表不是越大越好：150–200 类稳定，超过 ~300 类单类精度会下滑。

## GPU

| 设备 | 装什么 |
| --- | --- |
| NVIDIA / AMD / Intel 核显，不想装 CUDA | `pip install onnxruntime-directml`，然后用 `--provider dml` |
| NVIDIA + CUDA 12 + cuDNN | `pip install onnxruntime-gpu`，用 `--provider cuda` |

```bash
python -m depthscene --download --only depth --depth-variant fp32 --force   # GPU 上 fp32 更快
python -m depthscene --provider dml
```

启动会打印每个模型实际用的后端；GPU 上建图失败会自动回退 CPU。
笔记本双显卡记得在 NVIDIA 控制面板把 `python.exe` 指到独显。

## 目录

```
depthscene/                可 import 的包（也是服务本体）
  __init__.py              公开 API：DepthScene / Options / Detection
  pipeline.py              组合：帧 → 深度 → 检测 → 避障 → dict
  depth.py                 深度估计（自适应分辨率、逐帧归一化）
  detect.py                检测（解码、去重、时序平滑、后台线程）
  analyze.py               避障分析（扇区、连通域、框内测距）
  options.py               所有可调参数
  models.py                权重查找与下载
  runtime.py               ONNX Runtime 会话 / 后端回退
  vocab.py                 开放词表预设（189 类 + 中文名）
  server.py                HTTP 服务（纯标准库）
  export.py                导出固定词表 ONNX（唯一需要 ultralytics 的地方）
  cli.py / __main__.py     命令行
  web/                     页面（live.html / live.js / styles.css）
models/                    ← 首次启动自动生成：depth / yolov10n /（可选）yoloworld
weights/                   ← 导出词表时的 .pt 缓存（可删）
```

## 限制

- 单目深度是**相对深度**，`near`/`far` 只是标定映射，不是真实测距；要米制精度请接双目 / LiDAR。
- 检测框比画面滞后 0.3–1s（取决于检测耗时），画面本身不滞后。
- 每帧都要 JPEG 编码 + 上传，1080p 以上建议在浏览器侧先缩到 640 宽（`live.js` 已默认这么做）。
- YOLO-World / YOLOv10 权重遵循 AGPL-3.0（Ultralytics），商用前请确认许可。
