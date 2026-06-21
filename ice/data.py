import os
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from datasets import load_dataset
import lightning as L

class CodeDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length=512):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        encoding = self.tokenizer(
            str(item["func"]),
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(int(item["target"]), dtype=torch.long)
        }

class DevignDataModule(L.LightningDataModule):
    def __init__(self, model_name="microsoft/codebert-base", batch_size=8, max_length=512):
        super().__init__()
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.tokenizer = None
        self.datasets = {}

    def prepare_data(self):
        load_dataset("DetectVul/devign")

    def setup(self, stage=None):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        raw_dataset = load_dataset("DetectVul/devign")
        
        if stage == "fit" or stage is None:
            self.datasets["train"] = CodeDataset(raw_dataset["train"], self.tokenizer, self.max_length)
            self.datasets["val"] = CodeDataset(raw_dataset["validation"], self.tokenizer, self.max_length)
        if stage == "test" or stage is None:
            self.datasets["test"] = CodeDataset(raw_dataset["test"], self.tokenizer, self.max_length)

    def train_dataloader(self):
        return DataLoader(
            self.datasets["train"], 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=os.cpu_count() or 2, 
            pin_memory=torch.cuda.is_available()
        )

    def val_dataloader(self):
        return DataLoader(
            self.datasets["val"], 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=os.cpu_count() or 2, 
            pin_memory=torch.cuda.is_available()
        )

    def test_dataloader(self):
        return DataLoader(
            self.datasets["test"], 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=os.cpu_count() or 2, 
            pin_memory=torch.cuda.is_available()
        )