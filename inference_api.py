"""
Inference API for ANLI R2 Natural Language Inference (NLI)

This script defines a FastAPI service that loads a fine-tuned DeBERTa-v3 model
from the Hugging Face Hub and exposes a REST API for real-time predictions.

Endpoints:
    GET  /health     -> Health check
    POST /predict    -> Run NLI inference (entailment / neutral / contradiction)

This file is used in Docker deployment.
"""

import os
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load Model & Tokenizer from Hugging Face
HF_MODEL_ID = "BakshiSan/deberta-v3-anli-r2"
print(f"Loading model from Hugging Face Hub: {HF_MODEL_ID}")
tok = AutoTokenizer.from_pretrained(HF_MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(HF_MODEL_ID)
model.eval()

# Extract the mapping
id2label = model.config.id2label


# Initialize FastAPI 
app = FastAPI(
    title="ANLI R2 NLI Service",
    description="DeBERTa-v3-base fine-tuned on ANLI Round 2 (entailment / neutral / contradiction).",
    version="1.0",
)

# Define input JSON
class NLIRequest(BaseModel):
    premise: str
    hypothesis: str

# Health Check Endpoint
@app.get("/health")
def health():
    return {"status": "ok"}

# Prediction Endpoint
@app.post("/predict")
def predict(req: NLIRequest):
    inputs = tok(
        req.premise,
        req.hypothesis,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256,
    )

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1).squeeze().tolist()
        pred_id = int(torch.argmax(logits, dim=1).item())

    # Build output JSON
    return {
        "label_id": pred_id,
        "label": id2label[pred_id],
        "probabilities": {id2label[i]: float(p) for i, p in enumerate(probs)},
    }