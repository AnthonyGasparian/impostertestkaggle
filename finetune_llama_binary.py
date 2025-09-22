import os
import argparse
from typing import List, Dict
import math
import random
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AutoConfig, get_linear_schedule_with_warmup
from tqdm.auto import tqdm

# ---------------------------
# Dataset
# ---------------------------
class TextLabelDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        enc = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding="max_length"
        )
        # enc contains input_ids, attention_mask
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(label, dtype=torch.float)  # BCEWithLogitsLoss expects float targets
        return item
    
    
# ---------------------------
# Model wrapper
# ---------------------------
class LMForBinaryClassification(nn.Module):
    def __init__(self, base_model_name_or_path: str, freeze_base: bool = True, dropout: float = 0.1):
        """
        - Loads the base LM (causal or encoder) via AutoModel and attaches a binary classification head.
        - For causal LMs (like LLaMA), we use the last hidden state's final token representation.
        """
        super().__init__()
        # Load config/tokenizer/model
        self.config = AutoConfig.from_pretrained(base_model_name_or_path, trust_remote_code=False)
        # Use AutoModel (returns base model w/out LM head in many HF releases)
        # If the local model only exposes LlamaForCausalLM, AutoModel will still load the base model layers.
        self.base = AutoModel.from_pretrained(base_model_name_or_path, config=self.config, trust_remote_code=False)

        hidden_size = getattr(self.base.config, "hidden_size", None) or getattr(self.base.config, "dim", None)
        if hidden_size is None:
            raise RuntimeError("Couldn't infer hidden size from the base model config.")

        self.pooler = nn.Identity()  # For now we extract last token hidden state; user can change pooling.
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, 1)  # single logit for binary classification

        if freeze_base:
            for p in self.base.parameters():
                p.requires_grad = False
            # but keep classifier params trainable (they are by default)

    def forward(self, input_ids, attention_mask=None, labels=None):
        """
        Returns: dict with 'logits' and optional 'loss'
        For causal models: we take the hidden state of the last non-padded token.
        """
        outputs = self.base(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)

        # outputs.last_hidden_state shape: (batch, seq_len, hidden)
        last_hidden = outputs.last_hidden_state  # (B, L, H)

        # For classification with causal LM: use hidden state at the last non-masked position.
        if attention_mask is not None:
            # compute index of last token per example (sum of mask - 1)
            lengths = attention_mask.sum(dim=1) - 1  # shape (B,)
            batch_indices = torch.arange(last_hidden.size(0), device=last_hidden.device)
            pooled = last_hidden[batch_indices, lengths, :]  # (B, H)
        else:
            # fallback: take last token
            pooled = last_hidden[:, -1, :]

        pooled = self.dropout(pooled)
        logits = self.classifier(pooled).squeeze(-1)  # (B,)

        out = {"logits": logits}
        if labels is not None:
            # BCEWithLogitsLoss expects float labels
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels.float())
            out["loss"] = loss
        return out


# ---------------------------
# Helpers: metrics
# ---------------------------
def binary_metrics_from_logits(logits: torch.Tensor, labels: torch.Tensor, threshold: float = 0.5):
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).long()
    labels_long = labels.long()
    correct = (preds == labels_long).sum().item()
    acc = correct / labels.size(0)
    # simple precision/recall/f1
    tp = ((preds == 1) & (labels_long == 1)).sum().item()
    fp = ((preds == 1) & (labels_long == 0)).sum().item()
    fn = ((preds == 0) & (labels_long == 1)).sum().item()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


# ---------------------------
# Training loop
# ---------------------------
def train(
    model: nn.Module,
    tokenizer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    output_dir: str,
    gradient_accumulation_steps: int = 1,
    max_grad_norm: float = 1.0,
    warmup_steps: int = 0,
    use_fp16: bool = True,
):
    model.to(device)

    # Only parameters with requires_grad=True will be optimized
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    total_steps = (len(train_loader) // gradient_accumulation_steps) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=max(1, total_steps))

    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16 and device.type == "cuda")

    global_step = 0
    best_val_f1 = -1.0

    for epoch in range(epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Train Epoch {epoch+1}")
        running_loss = 0.0

        for step, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.cuda.amp.autocast(enabled=use_fp16 and device.type == "cuda"):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs["loss"] / gradient_accumulation_steps

            scaler.scale(loss).backward()
            running_loss += loss.item() * gradient_accumulation_steps

            if (step + 1) % gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

                pbar.set_postfix({"loss": f"{running_loss / (global_step + 1):.4f}"})

        # End epoch: evaluate
        val_metrics = evaluate(model, val_loader, device, use_fp16=use_fp16)
        val_f1 = val_metrics["f1"]
        print(f"Epoch {epoch+1} validation metrics: {val_metrics}")

        # Save best
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            os.makedirs(output_dir, exist_ok=True)
            save_path = os.path.join(output_dir, "best_model.pth")
            # Save model state dict + tokenizer
            torch.save({
                "model_state_dict": model.state_dict(),
                "tokenizer": tokenizer.__class__.__name__,
                "config": getattr(model, "config", None)
            }, save_path)
            # also save via huggingface methods if base supports it
            try:
                # if base model is HF-like, save with save_pretrained for tokenizer compatibility
                tokenizer.save_pretrained(output_dir)
                # attempt to save base model weights (only works if the underlying HF objects have save_pretrained)
                if hasattr(model.base, "save_pretrained"):
                    model.base.save_pretrained(os.path.join(output_dir, "base_model"))
            except Exception as e:
                print("Warning: couldn't save tokenizer/base via HF utilities:", e)

    print("Training complete. Best val f1:", best_val_f1)
    return model

# ---------------------------
# Evaluation
# ---------------------------
@torch.no_grad()
def evaluate(model: nn.Module, data_loader: DataLoader, device: torch.device, use_fp16: bool = True):
    model.eval()
    all_logits = []
    all_labels = []
    for batch in tqdm(data_loader, desc="Evaluating", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with torch.cuda.amp.autocast(enabled=use_fp16 and device.type == "cuda"):
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = out["logits"]

        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    metrics = binary_metrics_from_logits(all_logits, all_labels)
    return metrics

# ---------------------------
# Utility to load CSVs and build dataloaders
# ---------------------------
def build_dataloaders(train_csv: str, val_csv: str, tokenizer, batch_size: int, max_length: int, num_workers: int = 0):
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)

    assert "text" in train_df.columns and "label" in train_df.columns, "CSV must have 'text' and 'label' columns."

    train_ds = TextLabelDataset(train_df["text"].tolist(), train_df["label"].tolist(), tokenizer, max_length=max_length)
    val_ds = TextLabelDataset(val_df["text"].tolist(), val_df["label"].tolist(), tokenizer, max_length=max_length)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader
