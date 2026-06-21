from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
import torch


class CodeDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length=512):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        code = str(item["code"])
        label = int(item["label"])

        encoding = self.tokenizer(
            code,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long)
        }


def get_dataloaders(hparams):

    dataset = load_dataset("DetectVul/devign")

    def preprocess(ex):
        return {
            "code": ex["func"],
            "label": ex["target"]
        }

    dataset = dataset.map(preprocess)

    tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/codebert-base"
    )

    train_ds = CodeDataset(dataset["train"], tokenizer)
    val_ds = CodeDataset(dataset["validation"], tokenizer)
    test_ds = CodeDataset(dataset["test"], tokenizer)

    train_loader = DataLoader(
        train_ds,
        batch_size=hparams.batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=hparams.batch_size,
        shuffle=False
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=hparams.batch_size,
        shuffle=False
    )

    return train_loader, val_loader, test_loader