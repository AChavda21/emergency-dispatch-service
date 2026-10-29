# The ML Training Pipeline
# This script reads the CSV, fine-tunes the model, and exports the .pth binary artifact to your models/ directory for the API to use.

import sys
import os
import argparse
import json
import torch
import torch.nn as nn
import pandas as pd
import tiktoken
import logging
from torch.utils.data import Dataset, DataLoader

# Append project root to path so we can import the core architecture
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.architecture import GPTModel
from core.config import (MODEL_CONFIG, NUM_CLASSES, ROUTING_MAP, TRAINING_CORPUS_PATH,
                         MODEL_PATH, METRICS_PATH, select_device)
from core.preprocessing import encode_transcript, select_last_token
from core.pretrained import load_pretrained_gpt2

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class DispatchDataset(Dataset):
    def __init__(self, data, tokenizer):
        encoded = [encode_transcript(tokenizer, text) for text in data["transcript"]]
        self.input_ids = [ids for ids, _ in encoded]
        self.lengths = [length for _, length in encoded]
        self.labels = data["label"].tolist()

    def __getitem__(self, i):
        return torch.tensor(self.input_ids[i]), torch.tensor(self.lengths[i]), torch.tensor(self.labels[i])

    def __len__(self):
        return len(self.labels)


def split_train_val(df, val_fraction=0.2, seed=42):
    """Stratified split so every department is represented in validation."""
    val = df.groupby("label").sample(frac=val_fraction, random_state=seed)
    train = df.drop(val.index)
    return train.reset_index(drop=True), val.reset_index(drop=True)


def configure_classification_head(base_model, num_classes=NUM_CLASSES):
    for param in base_model.parameters(): param.requires_grad = False
    base_model.out_head = nn.Linear(MODEL_CONFIG["emb_dim"], num_classes)
    for param in base_model.trf_blocks[-1].parameters(): param.requires_grad = True
    for param in base_model.final_norm.parameters(): param.requires_grad = True
    return base_model


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    confusion = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long)
    for x, lengths, y in dataloader:
        x, lengths = x.to(device), lengths.to(device)
        preds = select_last_token(model(x), lengths).argmax(dim=-1).cpu()
        for t, p in zip(y, preds):
            confusion[t, p] += 1

    true_pos = confusion.diag().float()
    precision = true_pos / confusion.sum(dim=0).clamp(min=1)
    recall = true_pos / confusion.sum(dim=1).clamp(min=1)
    f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-8)
    return {
        "accuracy": (true_pos.sum() / confusion.sum()).item(),
        "macro_f1": f1.mean().item(),
        "per_class_f1": {ROUTING_MAP[i]: round(f1[i].item(), 4) for i in range(NUM_CLASSES)},
        "confusion_matrix": confusion.tolist(),
    }


def execute_training_pipeline(epochs=5, lr=5e-5, use_pretrained=True):
    device = select_device()
    logging.info(f"Initializing training pipeline on {device}...")

    tokenizer = tiktoken.get_encoding("gpt2")
    train_df, val_df = split_train_val(pd.read_csv(TRAINING_CORPUS_PATH))
    train_loader = DataLoader(DispatchDataset(train_df, tokenizer), batch_size=16, shuffle=True)
    val_loader = DataLoader(DispatchDataset(val_df, tokenizer), batch_size=64)
    logging.info(f"Train records: {len(train_df)} | Validation records: {len(val_df)}")

    model = GPTModel(MODEL_CONFIG)
    if use_pretrained:
        model = load_pretrained_gpt2(model)
        logging.info("Loaded pretrained GPT-2 (124M) weights.")
    model = configure_classification_head(model).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Trainable parameters: {trainable:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)

    logging.info("Beginning gradient descent...")
    best_metrics = None
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, lengths, y in train_loader:
            x, lengths, y = x.to(device), lengths.to(device), y.to(device)
            optimizer.zero_grad()
            logits = select_last_token(model(x), lengths)
            loss = nn.functional.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        metrics = evaluate(model, val_loader, device)
        logging.info(f"Epoch {epoch+1}/{epochs} | Train Loss: {total_loss/len(train_loader):.4f} | "
                     f"Val Accuracy: {metrics['accuracy']:.2%} | Val Macro-F1: {metrics['macro_f1']:.4f}")

        if best_metrics is None or metrics["macro_f1"] > best_metrics["macro_f1"]:
            best_metrics = {**metrics, "epoch": epoch + 1}
            torch.save(model.state_dict(), MODEL_PATH)

    logging.info(f"Best epoch {best_metrics['epoch']} | Confusion matrix (rows = true, cols = predicted):")
    for i, row in enumerate(best_metrics["confusion_matrix"]):
        logging.info(f"  {ROUTING_MAP[i]:<18} {row}")

    with open(METRICS_PATH, "w") as f:
        json.dump(best_metrics, f, indent=2)
    logging.info(f"Artifact exported successfully to {MODEL_PATH} (metrics: {METRICS_PATH})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune the dispatch triage classifier.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--no-pretrained", action="store_true", help="Train from random weights instead of GPT-2")
    args = parser.parse_args()
    execute_training_pipeline(epochs=args.epochs, lr=args.lr, use_pretrained=not args.no_pretrained)
