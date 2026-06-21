import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
import torchmetrics

from ice.model import VulnerabilityClassifier
from ice.data import DevignDataModule

class IceModule(L.LightningModule):
    def __init__(self, hparams_arg):
        super().__init__()
        self.save_hyperparameters(vars(hparams_arg))
        
        self.model = VulnerabilityClassifier(
            num_classes=self.hparams.num_classes,
            freeze_encoder=self.hparams.freeze_encoder
        )
        self.criterion = nn.CrossEntropyLoss()

        # Initialisation des métriques avancées (Précision, Rappel, F1-Score)
        metrics = torchmetrics.MetricCollection([
            torchmetrics.classification.BinaryAccuracy(),
            torchmetrics.classification.BinaryPrecision(),
            torchmetrics.classification.BinaryRecall(),
            torchmetrics.classification.BinaryF1Score()
        ])
        
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask)

    def training_step(self, batch, batch_idx):
        logits = self(batch["input_ids"], batch["attention_mask"])
        loss = self.criterion(logits, batch["label"])
        preds = torch.argmax(logits, dim=1)
        
        # Calcul et log des métriques d'entraînement
        output_metrics = self.train_metrics(preds, batch["label"])
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        self.log_dict(output_metrics, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch["input_ids"], batch["attention_mask"])
        loss = self.criterion(logits, batch["label"])
        preds = torch.argmax(logits, dim=1)
        
        output_metrics = self.val_metrics(preds, batch["label"])
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        self.log_dict(output_metrics, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        logits = self(batch["input_ids"], batch["attention_mask"])
        loss = self.criterion(logits, batch["label"])
        preds = torch.argmax(logits, dim=1)
        
        output_metrics = self.test_metrics(preds, batch["label"])
        self.log("test_loss", loss, on_epoch=True)
        self.log_dict(output_metrics, on_epoch=True)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=0.01)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=1
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }

def parse_args():
    parser = argparse.ArgumentParser(description="Ice Code Analyser")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--model_dir", type=str, default="./checkpoints")
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--freeze_encoder", type=bool, default=False)
    return parser.parse_args()

def main():
    args = parse_args()
    
    dm = DevignDataModule(batch_size=args.batch_size)
    model = IceModule(args)

    logger = TensorBoardLogger(save_dir=args.model_dir, name="logs")

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.model_dir,
        filename="best-checkpoint-{epoch:02d}-{val_loss:.2f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1
    )
    
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=2,
        mode="min"
    )

    trainer = L.Trainer(
        max_epochs=args.epochs,
        logger=logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        accelerator="auto",
        devices=1,
        precision="16-mixed" if torch.cuda.is_available() else 32
    )

    trainer.fit(model, datamodule=dm)
    trainer.test(datamodule=dm, ckpt_path="best")

if __name__ == "__main__":
    main()