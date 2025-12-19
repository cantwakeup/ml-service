#!/usr/bin/env python3
"""
简易 CLIP runner：
- 从 Redis 队列（默认 photo_tasks）阻塞消费任务
- 读取任务中的图片路径，使用 HuggingFace CLIP 生成 embedding
- 调用后端内部接口 /api/internal/photos/{id}/ml_result 写回 embedding

环境变量（可用 .env 配置）：
- REDIS_URL: Redis 连接串，默认 redis://127.0.0.1:6379/0
- REDIS_QUEUE_NAME: 任务队列名，默认 photo_tasks
- REDIS_DLQ_NAME: 失败时的死信队列名，默认 photo_tasks_dlq
- API_BASE_URL: 服务地址，默认 http://127.0.0.1:5800
- PROCESSING_INTERNAL_TOKEN: 与服务端配置一致的内部 token
- CLIP_MODEL_NAME: HuggingFace 模型名，默认 openai/clip-vit-large-patch14
- PROCESSING_EMBEDDING_DIM: 预期 embedding 维度，默认 768
- PROCESSING_DEFAULT_MODEL: 写回数据库时记录的 model 名称，默认 config 中的 clip-ViT-L-14
"""

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Optional

from pathlib import Path

import redis
import requests
import torch
import cv2
from insightface.app import FaceAnalysis
from PIL import Image
from transformers import CLIPModel, CLIPProcessor, BlipForConditionalGeneration, BlipProcessor

try:
    from dotenv import load_dotenv

    load_dotenv()
except ModuleNotFoundError:
    # 允许没有 python-dotenv 也能运行
    pass


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
)


@dataclass
class Task:
    photo_id: int
    user_id: int
    path: str
    tasks: list

    @classmethod
    def from_raw(cls, raw: str) -> "Task":
        data = json.loads(raw)
        return cls(
            photo_id=int(data["photo_id"]),
            user_id=int(data["user_id"]),
            path=data["path"],
            tasks=data.get("tasks", []),
        )


class ClipRunner:
    def __init__(self) -> None:
        self.redis_url = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")
        self.queue_name = os.getenv("REDIS_QUEUE_NAME", "photo_tasks")
        self.dlq_name = os.getenv("REDIS_DLQ_NAME", f"{self.queue_name}_dlq")
        self.api_base = os.getenv("API_BASE_URL", "http://127.0.0.1:5800")
        self.internal_token = os.getenv("PROCESSING_INTERNAL_TOKEN", "change-me")
        self.embedding_dim = int(os.getenv("PROCESSING_EMBEDDING_DIM", "768"))
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info("Using device: %s", self.device)
        self.models = self._load_models()

        # Face embedding settings
        self.face_model_name = os.getenv("FACE_MODEL_NAME", "buffalo_l")
        self.face_embedding_dim = int(os.getenv("PROCESSING_FACE_EMBEDDING_DIM", "512"))
        self.default_face_model_label = os.getenv(
            "PROCESSING_DEFAULT_FACE_MODEL", "arcface-buffalo_l"
        )
        self.caption_enabled = os.getenv("CAPTION_ENABLED", "1") != "0"
        self.caption_model_name = os.getenv(
            "CAPTION_MODEL_NAME", "Salesforce/blip-image-captioning-base"
        )
        self.caption_max_length = int(os.getenv("CAPTION_MAX_LENGTH", "64"))
        self.caption_processor = None
        self.caption_model = None

        # 解决任务里的相对路径：默认取仓库根目录（ml 的上级），或使用 PROJECT_ROOT/UPLOAD_BASE_DIR 覆盖
        self.base_dir = Path(
            os.getenv("PROJECT_ROOT")
            or os.getenv("UPLOAD_BASE_DIR")
            or Path(__file__).resolve().parents[1]
        )

        # 预加载模型，避免每个任务重复开销
        for m in self.models:
            logging.info("Loaded CLIP model label=%s hf=%s", m["label"], m["name"])

        if self.caption_enabled:
            self.caption_processor = BlipProcessor.from_pretrained(self.caption_model_name)
            self.caption_model = BlipForConditionalGeneration.from_pretrained(
                self.caption_model_name
            ).to(self.device)
            self.caption_model.eval()
            logging.info("Loaded caption model %s", self.caption_model_name)

        # 人脸模型（insightface），优先 GPU
        self.face_app = FaceAnalysis(name=self.face_model_name, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        self.face_app.prepare(ctx_id=0 if torch.cuda.is_available() else -1, det_size=(640, 640))

        self.redis = redis.from_url(self.redis_url)

    def run_forever(self) -> None:
        logging.info("Runner started. Waiting for tasks on %s", self.queue_name)
        while True:
            task = self._pop_task()
            if not task:
                continue

            try:
                self._process_task(task)
            except Exception:
                logging.exception("Task %s failed", task)
                self._push_dlq(task)
                # 避免疯狂重试，稍作等待
                time.sleep(1)

    def _pop_task(self) -> Optional[Task]:
        item = self.redis.brpop(self.queue_name, timeout=5)
        if item is None:
            return None
        _, payload = item
        try:
            return Task.from_raw(payload.decode("utf-8"))
        except Exception:
            logging.exception("Failed to parse task payload: %s", payload)
            return None

    def _push_dlq(self, task: Task) -> None:
        try:
            self.redis.lpush(self.dlq_name, json.dumps(task.__dict__))
        except Exception:
            logging.exception("Failed to push task to DLQ")

    def _process_task(self, task: Task) -> None:
        img_path = self._resolve_path(task.path)
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")

        logging.info("Processing photo %s for user %s", task.photo_id, task.user_id)

        if "clip" in task.tasks:
            self._run_clip(task.photo_id, img_path)
            if self.caption_enabled:
                self._run_caption(task.photo_id, img_path)

        if "face" in task.tasks or "faces" in task.tasks:
            self._run_faces(task.photo_id, img_path)

    def _run_clip(self, photo_id: int, img_path: Path) -> None:
        image = Image.open(img_path).convert("RGB")
        for m in self.models:
            processor = m["processor"]
            model = m["model"]
            inputs = processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                feats = model.get_image_features(**inputs)
                feats = feats / feats.norm(p=2, dim=-1, keepdim=True)

            embedding = feats.squeeze(0).cpu().tolist()

            if len(embedding) != self.embedding_dim:
                logging.warning(
                    "Embedding dim mismatch: expected %s, got %s for model %s",
                    self.embedding_dim,
                    len(embedding),
                    m["label"],
                )

            payload = {
                "model": m["label"],
                "modality": "image",
                "embedding": embedding,
                "ml_result": {
                    "source": "clip_runner",
                    "model_name": m["name"],
                    "device": self.device,
                },
                "processing_status": "done",
            }

            url = f"{self.api_base}/api/internal/photos/{photo_id}/ml_result"
            headers = {"x-internal-token": self.internal_token}
            resp = requests.post(url, json=payload, headers=headers, timeout=30)
            if resp.status_code >= 300:
                raise RuntimeError(f"Callback failed: {resp.status_code} {resp.text}")

            logging.info(
                "Photo %s: clip embedding updated (model=%s)", photo_id, m["label"]
            )

    def _run_faces(self, photo_id: int, img_path: Path) -> None:
        img = cv2.imread(str(img_path))
        if img is None:
            raise RuntimeError(f"Failed to read image for faces: {img_path}")

        faces = self.face_app.get(img)
        if not faces:
            logging.info("Photo %s: no faces detected", photo_id)
            return

        payload_faces = []
        for face in faces:
            emb = getattr(face, "normed_embedding", None)
            if emb is None:
                emb = getattr(face, "embedding", None)
            if emb is None:
                continue
            emb_list = emb.tolist()
            if len(emb_list) != self.face_embedding_dim:
                logging.warning(
                    "Face embedding dim mismatch: expected %s, got %s",
                    self.face_embedding_dim,
                    len(emb_list),
                )
            bbox = getattr(face, "bbox", None)
            bbox_dict = None
            if bbox is not None:
                # bbox 格式 [x1, y1, x2, y2]
                bbox_dict = {
                    "x1": float(bbox[0]),
                    "y1": float(bbox[1]),
                    "x2": float(bbox[2]),
                    "y2": float(bbox[3]),
                }

            payload_faces.append(
                {
                    "embedding": emb_list,
                    "model": self.default_face_model_label,
                    "bbox": bbox_dict,
                }
            )

        if not payload_faces:
            logging.info("Photo %s: faces detected but no embeddings", photo_id)
            return

        url = f"{self.api_base}/api/internal/photos/{photo_id}/faces"
        headers = {"x-internal-token": self.internal_token}
        resp = requests.post(
            url, json={"faces": payload_faces}, headers=headers, timeout=30
        )
        if resp.status_code >= 300:
            raise RuntimeError(f"Face callback failed: {resp.status_code} {resp.text}")

        logging.info(
            "Photo %s: %s faces uploaded (model=%s)",
            photo_id,
            len(payload_faces),
            self.default_face_model_label,
        )

    def _run_caption(self, photo_id: int, img_path: Path) -> None:
        if not self.caption_processor or not self.caption_model or not self.models:
            return

        image = Image.open(img_path).convert("RGB")
        inputs = self.caption_processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.caption_model.generate(
                **inputs, max_new_tokens=self.caption_max_length
            )
        caption = self.caption_processor.decode(out[0], skip_special_tokens=True).strip()
        if not caption:
            logging.info("Photo %s: caption empty, skip caption embedding", photo_id)
            return

        # 复用首个 CLIP 模型编码 caption
        text_proc = self.models[0]["processor"]
        text_model = self.models[0]["model"]
        text_inputs = text_proc(text=[caption], return_tensors="pt", padding=True)
        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
        with torch.no_grad():
            feats = text_model.get_text_features(**text_inputs)
            feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
        embedding = feats.squeeze(0).cpu().tolist()

        payload = {
            "model": self.models[0]["label"],
            "modality": "caption",
            "embedding": embedding,
            "processing_status": "done",
        }

        url = f"{self.api_base}/api/internal/photos/{photo_id}/ml_result"
        headers = {"x-internal-token": self.internal_token}
        resp = requests.post(url, json=payload, headers=headers, timeout=30)
        if resp.status_code >= 300:
            raise RuntimeError(f"Caption callback failed: {resp.status_code} {resp.text}")

        logging.info("Photo %s: caption='%s' embedding updated", photo_id, caption)

    def _resolve_path(self, path: str) -> Path:
        candidate = Path(path)
        if candidate.is_absolute():
            return candidate

        combined = self.base_dir / candidate
        if combined.exists():
            return combined

        # 兜底返回绝对化的原路径（便于错误提示）
        return candidate.resolve()

    def _load_models(self):
        # 支持多模型：CLIP_MODELS 格式 label=hf_repo,label2=hf_repo2
        # 例如：clip14=openai/clip-vit-large-patch14,laion=laion/CLIP-ViT-H-14-laion2B-s32B-b79K
        # 若未配置则退回单模型（CLIP_MODEL_NAME + PROCESSING_DEFAULT_MODEL）
        raw = os.getenv("CLIP_MODELS")
        entries = []
        if raw:
            for item in raw.split(","):
                if "=" in item:
                    label, name = item.split("=", 1)
                    entries.append((label.strip(), name.strip()))
        if not entries:
            default_label = os.getenv("PROCESSING_DEFAULT_MODEL", "clip-ViT-L-14")
            default_name = os.getenv("CLIP_MODEL_NAME", "openai/clip-vit-large-patch14")
            entries.append((default_label, default_name))

        models = []
        for label, name in entries:
            processor = CLIPProcessor.from_pretrained(name)
            model = CLIPModel.from_pretrained(name).to(self.device)
            model.eval()
            models.append({"label": label, "name": name, "processor": processor, "model": model})
        return models


def main() -> None:
    runner = ClipRunner()
    runner.run_forever()


if __name__ == "__main__":
    main()
