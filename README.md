# CLIP Runner

这是一个独立的 CLIP 推理脚本，用来消费服务端推入 Redis 的图片处理任务，生成 embedding 并通过内部接口写回数据库。

## 准备环境

1) 安装依赖

```bash
cd ml
python -m venv .venv
source .venv/bin/activate  # Windows 使用 .venv\\Scripts\\activate
pip install -r requirements.txt
```

> 首次运行会从 HuggingFace 下载 `openai/clip-vit-large-patch14`，如需离线可预先下载到本地 cache（设置 `HF_HOME`）。

2) 配置环境变量（可在 `.env` 中设置）：

```
REDIS_URL=redis://127.0.0.1:6379/0
REDIS_QUEUE_NAME=photo_tasks
REDIS_DLQ_NAME=photo_tasks_dlq
API_BASE_URL=http://127.0.0.1:5800
PROCESSING_INTERNAL_TOKEN=与服务端一致的 token
PROCESSING_EMBEDDING_DIM=768
# 多模型：label=hf_repo 逗号分隔；留空则使用 PROCESSING_DEFAULT_MODEL + CLIP_MODEL_NAME
# 例：CLIP_MODELS=clip14=openai/clip-vit-large-patch14,laion=laion/CLIP-ViT-H-14-laion2B-s32B-b79K
CLIP_MODELS=clip14=openai/clip-vit-large-patch14
PROCESSING_DEFAULT_MODEL=clip-ViT-L-14
# 文本侧：后端会调用 embed_service，无需前端生成
PROCESSING_TEXT_EMBED_URL=http://127.0.0.1:8001/embed_text
# Caption 生成：默认开启；如需关闭 CAPTION_ENABLED=0
CAPTION_ENABLED=1
CAPTION_MODEL_NAME=Salesforce/blip-image-captioning-base
CAPTION_MAX_LENGTH=64
# 人脸相关
FACE_MODEL_NAME=buffalo_l
PROCESSING_FACE_EMBEDDING_DIM=512
PROCESSING_DEFAULT_FACE_MODEL=arcface-buffalo_l
# 选填：为解决相对路径，runner 会尝试用仓库根目录；如路径仍找不到，可显式指定
PROJECT_ROOT=/home/xxx/mindsight-gallery-server
# 或者只给上传目录前缀
UPLOAD_BASE_DIR=/home/xxx/mindsight-gallery-server
```

## 运行

```bash
cd ml
python clip_runner.py
```

脚本会阻塞等待队列任务（服务端使用 `lpush` 入队）。当任务包含 `"clip"` 时，将读取 `path` 指向的图片，计算归一化后的 embedding，并调用：

```
POST {API_BASE_URL}/api/internal/photos/{photo_id}/ml_result
Header: x-internal-token: {PROCESSING_INTERNAL_TOKEN}
```

回写字段包含：`model`、`modality=image`、`embedding`、`ml_result`（记录 runner 来源）以及 `processing_status=done`。

当任务包含 `"face"` 或 `"faces"` 时，runner 会调用：

```
POST {API_BASE_URL}/api/internal/photos/{photo_id}/faces
Header: x-internal-token: {PROCESSING_INTERNAL_TOKEN}
Body: {"faces": [{"embedding": [...], "model": "arcface-buffalo_l", "bbox": {...}}] }
```

人脸特征使用 insightface `FACE_MODEL_NAME` 生成（默认 `buffalo_l`），embedding 长度默认为 512，写回时 model 字段为 `PROCESSING_DEFAULT_FACE_MODEL`。

## 任务格式

Redis 队列元素是 JSON 字符串，对应结构：

```json
{
  "photo_id": 1,
  "user_id": 1,
  "path": "./uploads/original/xxx.jpg",
  "tasks": ["thumbnail", "clip", "face"]
}
```

脚本处理包含 `"clip"` 或 `"face"` 的任务，失败会写入 `REDIS_DLQ_NAME` 方便排查。

## 额外说明

- 会自动选择 `cuda` 或 `cpu`，如需强制 CPU 可设置 `CUDA_VISIBLE_DEVICES=`。
- 如果服务端配置的 embedding 维度不是 768，请同时调整 `.env` 的 `PROCESSING_EMBEDDING_DIM` 与服务端环境变量保持一致。
- 任务里的图片路径如果是相对路径，默认会用仓库根目录（`ml` 的上一级）去拼；若 runner 不在仓库根运行，可通过 `PROJECT_ROOT` 或 `UPLOAD_BASE_DIR` 显式指定前缀。

---

## 文搜图（文本→图像检索）

脚本：`ml/text_search.py`

用途：输入一句文本，生成 CLIP 文本 embedding，然后调用服务端 `/api/search/vector` 返回最相似的照片。
如果希望完全由后端处理，可直接调用新增接口 `/api/search/text`，但仍需在调用前生成文本 embedding 并随请求提交（服务端不做文本编码）。

环境变量：

```
API_BASE_URL=http://127.0.0.1:5800
ACCESS_TOKEN=你的 JWT（登录获取）
CLIP_MODEL_NAME=openai/clip-vit-large-patch14
PROCESSING_EMBEDDING_DIM=768
PROCESSING_DEFAULT_MODEL=clip-ViT-L-14
# 服务器侧文本编码入口（推荐配合 /api/search/text 使用）
PROCESSING_TEXT_EMBED_URL=http://127.0.0.1:8001/embed_text
# 文搜图融合权重（后端读取环境变量）
TEXT_FUSION_IMAGE_WEIGHT=1.0
TEXT_FUSION_CAPTION_WEIGHT=0.7
PROCESSING_TEXT_EMBED_URL=http://127.0.0.1:8001/embed_text  # 若使用后端自动编码，需指向 embed_service
```

运行示例：

```bash
cd ml
python text_search.py "a cat on the beach" --limit 10
```

> 依赖已有的向量入库：确保你的图片已被 clip_runner 写入 embedding。
> 如果希望后端自动编码文本，可启动 `ml/embed_service.py`（见下）。

---

## 自动生成人物相册

脚本：`ml/people_album.py`

用途：本地读取已上传图片，使用 insightface 检测/提取人脸，按余弦距离聚类；对每个聚类调用 `/api/albums` 创建相册并把对应照片加入相册。

额外依赖：`insightface`, `opencv-python`, `numpy`（已写入 `requirements.txt`，首次需联网下载模型）。

环境变量：

```
API_BASE_URL=http://127.0.0.1:5800
ACCESS_TOKEN=你的 JWT（必填）
PROJECT_ROOT=/home/xxx/mindsight-gallery-server  # 用于拼接 /uploads/original/xxx 路径
# 或者 UPLOAD_BASE_DIR=/home/xxx/mindsight-gallery-server
FACE_DISTANCE_THRESHOLD=0.42   # 越小越严格
MIN_FACES_PER_PERSON=2         # 小于阈值的不建相册
MAX_PHOTOS=0                   # 可选，限制处理多少张最新照片
```

运行：

```bash
cd ml
python people_album.py
```

说明：
- 若路径无法命中本地文件，会在日志提示推断路径；请确认 `PROJECT_ROOT`/`UPLOAD_BASE_DIR` 是否正确。
- 聚类采用简单的中心向量+余弦距离规则，想要更严格或更宽松可调整 `FACE_DISTANCE_THRESHOLD`；`MIN_FACES_PER_PERSON` 防止噪声生成过多相册。
- 创建相册时会直接把聚类中涉及的照片 ID 一并提交，封面由后端按最新时间自动选取。

---

## 文本 embedding 服务（可选，供后端调用）

脚本：`ml/embed_service.py`（FastAPI）

用途：暴露 `/embed_text` 接口用于生成文本向量，后端可通过环境变量 `PROCESSING_TEXT_EMBED_URL` 调用，实现“文搜图”无需前端生成 embedding。

运行：

```bash
cd ml
uvicorn embed_service:app --host 0.0.0.0 --port 8001
# 或 python embed_service.py
```

接口示例：

```bash
curl -X POST http://127.0.0.1:8001/embed_text -H "Content-Type: application/json" \
  -d '{"text": "a cat on the beach"}'
```

返回 `{"embedding": [...]}`，长度通常为 768（默认 CLIP 文本模型）。

## 文本队列 Runner（异步模式）

脚本：`ml/text_runner.py`

用途：消费 Redis `text_tasks`（可通过 `TEXT_EMBED_QUEUE_NAME` 配置），为 `POST /api/search/text` 异步模式生成文本 embedding，并回调 `/api/internal/text_queries/{id}`。

环境变量：`REDIS_URL`、`TEXT_EMBED_QUEUE_NAME`（默认 text_tasks）、`TEXT_EMBED_DLQ_NAME`（默认 text_tasks_dlq）、`API_BASE_URL`、`PROCESSING_INTERNAL_TOKEN`、`CLIP_MODEL_NAME`、`PROCESSING_EMBEDDING_DIM`。

运行：
```bash
cd ml
python text_runner.py
```
