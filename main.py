import argparse
import torch
import torch.nn as nn
import torch.optim as optim

import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from ice.model import get_model
from ice.data import get_dataloaders


class Ice(L.LightningModule):
    def __init__(self, hparams):
        super().__init__()

        self.save_hyperparameters(vars(hparams))

        self.model = get_model(hparams)
        self.criterion = nn.CrossEntropyLoss()

        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

    def setup(self, stage=None):
        if self.train_loader is None:
            self.train_loader, self.val_loader, self.test_loader = get_dataloaders(self.hparams)

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask)

    def training_step(self, batch, batch_idx):
        logits = self(batch["input_ids"], batch["attention_mask"])

        loss = self.criterion(logits, batch["label"])

        preds = torch.argmax(logits, dim=1)
        acc = (preds == batch["label"]).float().mean()

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch["input_ids"], batch["attention_mask"])

        loss = self.criterion(logits, batch["label"])

        preds = torch.argmax(logits, dim=1)
        acc = (preds == batch["label"]).float().mean()

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        logits = self(batch["input_ids"], batch["attention_mask"])

        loss = self.criterion(logits, batch["label"])

        preds = torch.argmax(logits, dim=1)
        acc = (preds == batch["label"]).float().mean()

        self.log("test_loss", loss)
        self.log("test_acc", acc)

    def configure_optimizers(self):
        return optim.AdamW(
            self.parameters(),
            lr=self.hparams["lr"]
        )

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--model_dir", type=str, default="./checkpoints")
    parser.add_argument("--num_classes", type=int, default=2)

    return parser.parse_args()


def main():
    hparams = parse_args()

    model = Ice(hparams)

    logger = TensorBoardLogger(
        save_dir=hparams.model_dir,
        name="logs"
    )

    checkpoint = ModelCheckpoint(
        dirpath=hparams.model_dir,
        monitor="val_loss",
        mode="min",
        save_top_k=1
    )

    trainer = L.Trainer(
        max_epochs=hparams.epochs,
        logger=logger,
        callbacks=[checkpoint],
        accelerator="auto",
        devices="auto",
        precision="16-mixed" if torch.cuda.is_available() else 32
    )

    trainer.fit(model)
    trainer.test(model)


if __name__ == "__main__":
    main()