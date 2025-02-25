from transformers import AutoModel, AutoConfig
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple
import omegaconf

class EncoderForClassification(nn.Module):
    def __init__(self, model_config: omegaconf.DictConfig):
        super().__init__()
        if model_config.model_name == 'bert-base-uncased':
            self.encoder = AutoModel.from_pretrained(model_config.model_name, attn_implementation='eager')
        elif model_config.model_name == 'answerdotai/ModernBERT-base':
            config = AutoConfig.from_pretrained(model_config.model_name)
            config._attn_implementation = "eager"
            self.encoder = AutoModel.from_pretrained(model_config.model_name, config=config)
        hidden_size = self.encoder.config.hidden_size

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 2)
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, token_type_ids: torch.Tensor = None, label: torch.Tensor = None) -> dict:
        if token_type_ids is not None:
            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
        else:
            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        pooled_output = torch.mean(outputs.last_hidden_state, dim=1)
        logits = self.classifier(pooled_output)
        loss = self.loss_fn(logits, label)
        return {"logits": logits, "loss": loss}