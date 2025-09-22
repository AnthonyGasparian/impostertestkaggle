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
