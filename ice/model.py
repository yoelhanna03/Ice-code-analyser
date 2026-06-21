import torch
import torch.nn as nn
from transformers import AutoModel


class VulnerabilityClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(
            "microsoft/codebert-base"
        )

        hidden = self.encoder.config.hidden_size

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(hidden, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        cls = outputs.last_hidden_state[:, 0]

        return self.classifier(cls)


def get_model(hparams):
    return VulnerabilityClassifier(
        num_classes=getattr(hparams, "num_classes", 2)
    )