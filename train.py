import glob
import math
import os
import socket
from typing import Any, Dict, List, Optional

import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchmetrics.classification import BinaryAUROC, BinaryAccuracy
from transformers import AutoModel, AutoTokenizer

DATA_DIR = "data/train"
CSV_PATH = "data/train.csv"
BERT_NAME = "bert-base-uncased"
MAX_LENGTH = 256
BATCH_SIZE = 8
LR = 1e-3  # as requested
EPOCHS = 50
NUM_WORKERS = 4
SEED = 0xDEADBEEF
FREEZE_BERT = True  # baseline: freeze encoder for a fast MLP head


def get_localhost_ip() -> str:
    return "0.0.0.0"


def build_df() -> pd.DataFrame:
    article_dirs = sorted(glob.glob(os.path.join(DATA_DIR, "article_*")))
    fs_df = pd.DataFrame(
        {"id": list(range(len(article_dirs))), "article_dir": article_dirs}
    )
    csv = pd.read_csv(CSV_PATH)
    csv = csv.dropna(subset=["real_text_id"])
    csv["real_text_id"] = csv["real_text_id"].astype(int)
    df = pd.merge(fs_df, csv, on="id", how="inner")
    if df.empty:
        raise RuntimeError(
            "No overlap between filesystem articles and CSV ids. "
            "Check the id mapping assumption (sorted article_* order)."
        )
    df["target"] = (df["real_text_id"] == 1).astype(int)
    return df


class PairDataset(Dataset):
    """
    Loads file_1.txt and file_2.txt per article directory.
    Emits both raw texts + target (1 if file_1 is real, else 0).

    Modified: for each sample, both (file_1, file_2, target) and (file_2, file_1, 1-target) are added as augmented examples.
    """

    def __init__(self, df: pd.DataFrame, augment: bool = True):
        self.examples = []
        for idx, row in df.reset_index(drop=True).iterrows():
            d = row["article_dir"]
            t1_path = os.path.join(d, "file_1.txt")
            t2_path = os.path.join(d, "file_2.txt")
            t1 = self._read_txt(t1_path)
            t2 = self._read_txt(t2_path)
            y = int(row["target"]) if "target" in row else None
            id_ = int(row["id"])
            dir_ = d
            # If y exists, add with augmentation (for train/val). Test y=None.
            self.examples.append(
                {
                    "text1": t1,
                    "text2": t2,
                    "y": y,
                    "id": id_,
                    "dir": dir_,
                }
            )
            if "target" in row and augment:
                self.examples.append(
                    {
                        "text1": t2,
                        "text2": t1,
                        "y": 1 - y,
                        "id": id_,
                        "dir": dir_,
                    }
                )

    def __len__(self):
        return len(self.examples)

    @staticmethod
    def _read_txt(path: str) -> str:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.examples[idx]


class PairDataModule(pl.LightningDataModule):
    def __init__(self, df: pd.DataFrame, tokenizer_name: str = BERT_NAME):
        super().__init__()
        self.df = df
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def setup(self, stage: Optional[str] = None):
        train_df, val_df = train_test_split(
            self.df,
            test_size=0.2,
            random_state=SEED,
            stratify=self.df["target"],
        )
        # Augmentation only for train set
        self.train_ds = PairDataset(train_df, augment=True)
        self.val_ds = PairDataset(val_df, augment=False)

    def _collate(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        texts1 = [b["text1"] for b in batch]
        texts2 = [b["text2"] for b in batch]
        y = None
        if batch[0]["y"] is not None:
            y = torch.tensor([b["y"] for b in batch], dtype=torch.float32)
        enc1 = self.tokenizer(
            texts1,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        enc2 = self.tokenizer(
            texts2,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        meta = {
            "id": [b["id"] for b in batch],
            "dir": [b["dir"] for b in batch],
        }
        out = {"enc1": enc1, "enc2": enc2, "meta": meta}
        if y is not None:
            out["y"] = y
        return out

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            collate_fn=self._collate,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            collate_fn=self._collate,
        )


class BertPairMLP(pl.LightningModule):
    def __init__(
        self,
        bert_name: str = BERT_NAME,
        freeze_bert: bool = FREEZE_BERT,
        lr: float = LR,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.bert = AutoModel.from_pretrained(bert_name)
        hs = self.bert.config.hidden_size

        if freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = False
        self.bert.train()

        feat_dim = hs * 2
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
        )

        self.criterion = nn.BCEWithLogitsLoss()
        self.train_auc = BinaryAUROC()
        self.train_acc = BinaryAccuracy()
        self.val_auc = BinaryAUROC()
        self.val_acc = BinaryAccuracy()

    def _cls(self, enc: Dict[str, torch.Tensor]) -> torch.Tensor:
        out = self.bert(
            input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]
        )
        x = out.last_hidden_state[:, 0, :]
        return x

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        h1 = self._cls(batch["enc1"])
        h2 = self._cls(batch["enc2"])
        feats = torch.cat([h1, h2], dim=1)
        logit = self.mlp(feats).squeeze(1)
        return logit

    def training_step(self, batch, batch_idx):
        logit = self(batch)
        y = batch["y"]
        loss = self.criterion(logit, y)
        prob = torch.sigmoid(logit).detach()
        self.train_auc.update(prob, y.int())
        self.train_acc.update((prob >= 0.5).int(), y.int())
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=y.size(0),
        )
        return loss

    def on_train_epoch_end(self):
        auc = self.train_auc.compute()
        acc = self.train_acc.compute()
        self.log("train_auc", auc, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        self.train_auc.reset()
        self.train_acc.reset()

    def validation_step(self, batch, batch_idx):
        logit = self(batch)
        y = batch["y"]
        loss = self.criterion(logit, y)
        prob = torch.sigmoid(logit).detach()
        self.val_auc.update(prob, y.int())
        self.val_acc.update((prob >= 0.5).int(), y.int())
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=y.size(0),
        )

    def on_validation_epoch_end(self):
        auc = self.val_auc.compute()
        acc = self.val_acc.compute()
        self.log("val_auc", auc, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        self.val_auc.reset()
        self.val_acc.reset()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


def main():
    pl.seed_everything(SEED, workers=True)

    df = build_df()
    dm = PairDataModule(df)
    model = BertPairMLP()

    logger = TensorBoardLogger(save_dir="tb_logs", name="imposter_baseline")
    logdir = os.path.abspath(logger.log_dir)
    ip = get_localhost_ip()
    print(f"\nTensorBoard logdir: {logdir}")
    print(f"TensorBoard URL:    http://{ip}:6006/?logdir={logdir}\n")

    ckpt_cb = ModelCheckpoint(
        monitor="val_auc",
        mode="max",
        save_top_k=1,
        filename="epoch{epoch:02d}-valauc{val_auc:.4f}",
    )
    lr_cb = LearningRateMonitor(logging_interval="step")

    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        accelerator="auto",
        devices="auto",
        logger=logger,
        log_every_n_steps=10,
        callbacks=[ckpt_cb, lr_cb],
    )

    trainer.fit(model, datamodule=dm)

    # ----------- Create Test Data and Make Predictions ------------

    test_data_dir = "data/test"
    article_dirs = sorted(glob.glob(os.path.join(test_data_dir, "article_*")))
    test_df = pd.DataFrame(
        {
            "id": list(range(len(article_dirs))),
            "article_dir": article_dirs,
        }
    )

    class TestPairDataset(Dataset):
        def __init__(self, df):
            self.examples = []
            for idx, row in df.reset_index(drop=True).iterrows():
                d = row["article_dir"]
                t1_path = os.path.join(d, "file_1.txt")
                t2_path = os.path.join(d, "file_2.txt")
                t1 = self._read_txt(t1_path)
                t2 = self._read_txt(t2_path)
                id_ = int(row["id"])
                dir_ = d
                self.examples.append(
                    {
                        "text1": t1,
                        "text2": t2,
                        "id": id_,
                        "dir": dir_,
                    }
                )

        def __len__(self):
            return len(self.examples)

        @staticmethod
        def _read_txt(path: str) -> str:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()

        def __getitem__(self, idx: int) -> Dict[str, Any]:
            return self.examples[idx]

    class TestPairCollator:
        def __init__(self, tokenizer, max_length: int):
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
            texts1 = [b["text1"] for b in batch]
            texts2 = [b["text2"] for b in batch]
            enc1 = self.tokenizer(
                texts1,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            enc2 = self.tokenizer(
                texts2,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            meta = {
                "id": [b["id"] for b in batch],
                "dir": [b["dir"] for b in batch],
            }
            return {"enc1": enc1, "enc2": enc2, "meta": meta}

    # Load best checkpoint (if available)
    best_ckpt_path = ckpt_cb.best_model_path
    if best_ckpt_path and os.path.exists(best_ckpt_path):
        model = BertPairMLP.load_from_checkpoint(best_ckpt_path)

    model.eval()
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(BERT_NAME)
    test_ds = TestPairDataset(test_df)
    test_dl = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=TestPairCollator(tokenizer, MAX_LENGTH),
    )

    all_ids = []
    all_real_text_ids = []

    with torch.no_grad():
        for batch in test_dl:
            enc1 = {k: v.to(model.device) for k, v in batch["enc1"].items()}
            enc2 = {k: v.to(model.device) for k, v in batch["enc2"].items()}

            # Forward for (file_1, file_2)
            logit_1 = model({"enc1": enc1, "enc2": enc2})
            prob_1 = torch.sigmoid(logit_1)

            # Forward for (file_2, file_1)
            logit_2 = model({"enc1": enc2, "enc2": enc1})
            prob_2 = torch.sigmoid(logit_2)

            # Use the average of both directions to decide
            avg_probs_file1_real = (prob_1 + (1 - prob_2)) / 2

            preds = (avg_probs_file1_real >= 0.5).long()

            for i in range(len(preds)):
                all_ids.append(batch["meta"]["id"][i])
                real_text_id = 1 if preds[i].item() == 1 else 2
                all_real_text_ids.append(real_text_id)

    submission = pd.DataFrame(
        {
            "id": all_ids,
            "real_text_id": all_real_text_ids,
        }
    )
    submission.sort_values("id", inplace=True)
    submission.reset_index(drop=True, inplace=True)
    print("Test predictions head:")
    print(submission.head())
    submission.to_csv("test_predictions.csv", index=False)
    submission.to_csv("submission/test_predictions.csv.gz", index=False)
    print("Saved test predictions to test_predictions.csv")


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    main()
