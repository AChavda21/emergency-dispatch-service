# Shared Configuration
# Single source of truth for model hyperparameters, preprocessing constants, routing rules and paths.
# Both the training job and the live API import from here, so they can never drift apart.

from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# GPT-2 small (124M) architecture
MODEL_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": True,
}

# Preprocessing
MAX_LENGTH = 64
PAD_TOKEN_ID = 50256  # <|endoftext|>

# Routing
ROUTING_MAP = {
    0: "MEDICAL",
    1: "FIRE",
    2: "VIOLENT CRIME",
    3: "NON-VIOLENT CRIME",
    4: "RESCUE & HAZMAT",
}
NUM_CLASSES = len(ROUTING_MAP)
HIGH_SEVERITY_CLASSES = {1, 2, 4}  # Fire, Violent, Hazmat are HIGH

# Predictions below this softmax probability are flagged for a human dispatcher
CONFIDENCE_THRESHOLD = 0.60

# Paths
TRAINING_CORPUS_PATH = PROJECT_ROOT / "offline_jobs" / "training_corpus.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "triage_v2.pth"
METRICS_PATH = PROJECT_ROOT / "models" / "triage_v2_metrics.json"
PRETRAINED_GPT2_PATH = PROJECT_ROOT / "models" / "pretrained" / "gpt2_pytorch_model.bin"
PRETRAINED_GPT2_URL = "https://huggingface.co/openai-community/gpt2/resolve/main/pytorch_model.bin"
DASHBOARD_PATH = PROJECT_ROOT / "dashboard" / "index.html"


def select_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
