#!/usr/bin/env python3
"""
轻量文本 embedding 服务：
- POST /embed_text {"text": "...", "model": "openai/clip-vit-large-patch14"}
- 返回 {"embedding": [...]}

用途：后端可通过 PROCESSING_TEXT_EMBED_URL 调用本服务生成文本向量，再用 /api/search/text 搜索，无需前端/客户端生成 embedding。

运行：
```bash
cd ml
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn embed_service:app --host 0.0.0.0 --port 8001
```

环境变量：
- CLIP_MODEL_NAME: 默认 openai/clip-vit-large-patch14
- PROCESSING_EMBEDDING_DIM: 默认 768，用于警告尺寸不符
"""

import os
from functools import lru_cache
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import CLIPModel, CLIPProcessor

try:
    from dotenv import load_dotenv

    load_dotenv()
except ModuleNotFoundError:
    pass


class EmbedRequest(BaseModel):
    text: str
    model: Optional[str] = None


class EmbedResponse(BaseModel):
    embedding: List[float]


@lru_cache(maxsize=1)
def get_model_and_processor(model_name: str):
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    return processor, model, device


def embed_text(text: str, model_name: str, expected_dim: int) -> List[float]:
    processor, model, device = get_model_and_processor(model_name)
    inputs = processor(text=[text], return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        feats = model.get_text_features(**inputs)
        feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
    vec = feats.squeeze(0).cpu().tolist()
    if expected_dim and len(vec) != expected_dim:
        # 仅记录尺寸不符，不直接报错
        print(
            f"[warn] embedding dim mismatch, expected {expected_dim}, got {len(vec)}"
        )
    return vec


default_model = os.getenv("CLIP_MODEL_NAME", "openai/clip-vit-large-patch14")
expected_dim = int(os.getenv("PROCESSING_EMBEDDING_DIM", "768"))

app = FastAPI(title="Text Embedding Service", version="0.1.0")


@app.post("/embed_text", response_model=EmbedResponse)
def embed(req: EmbedRequest):
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is required")

    model_name = req.model or default_model
    try:
        embedding = embed_text(text, model_name, expected_dim)
    except Exception as e:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"failed to embed text: {e}")

    return EmbedResponse(embedding=embedding)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
