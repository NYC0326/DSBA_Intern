from transformers import AutoTokenizer
from datasets import load_dataset, concatenate_datasets

import torch
from torch.utils.data import Dataset, DataLoader

import omegaconf
from typing import List, Tuple, Literal

class IMDBDataset(torch.utils.data.Dataset):
    def __init__(self, data_config: omegaconf.DictConfig, split: Literal['train', 'valid', 'test']):
        self.split = split
        self.model_name = data_config.model_name.lower()
        dataset = load_dataset('stanfordnlp/imdb')
        tokenizer = AutoTokenizer.from_pretrained(data_config.model_name)

        all_data = concatenate_datasets([dataset['train'], dataset['test']])
        all_data = all_data.shuffle(seed=42).select(range(50000))
        train_size = int(0.8 * len(all_data))
        val_size = int(0.1 * len(all_data))
        test_size = len(all_data) - train_size - val_size

        splits = {
            'train': all_data.select(range(train_size)),
            'valid': all_data.select(range(train_size, train_size + val_size)),
            'test': all_data.select(range(train_size + val_size, train_size + val_size + test_size))
        }
        self.data = splits[split]

        self.tokenized_data = tokenizer(
            self.data['text'],
            padding='max_length',
            truncation=True,
            max_length=data_config.max_len,
            return_tensors='pt',
        )

        self.labels = torch.tensor(self.data['label'])

        print(f">> SPLIT: {self.split} | Total Data Length: {len(self.data['text'])}")
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx) -> Tuple[dict, int]:
        inputs = {
            'input_ids': self.tokenized_data['input_ids'][idx],
            'attention_mask': self.tokenized_data['attention_mask'][idx]
        }
        # ModernBERT token_type_ids exception
        if "modernbert" not in self.model_name:
            if 'token_type_ids' in self.tokenized_data:
                inputs['token_type_ids'] = self.tokenized_data['token_type_ids'][idx]
        return inputs, self.labels[idx]

    @staticmethod
    def collate_fn(batch: List[Tuple[dict, int]]) -> dict:
        input_ids = torch.stack([x[0]['input_ids'] for x in batch])
        attention_mask = torch.stack([x[0]['attention_mask'] for x in batch])
        data_dict = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': torch.tensor([x[1] for x in batch])
        }
        if 'token_type_ids' in batch[0][0]:
            data_dict['token_type_ids'] = torch.stack([x[0]['token_type_ids'] for x in batch])
        return data_dict

def get_dataloader(data_config: omegaconf.DictConfig, split: Literal['train', 'valid', 'test']) -> DataLoader:
    dataset = IMDBDataset(data_config, split)
    batch_size = data_config.batch_size[split]

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        collate_fn=IMDBDataset.collate_fn
    )
    return dataloader


