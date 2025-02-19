from transformers import AutoTokenizer
from datasets import load_dataset, concatenate_datasets

import torch
import torch.nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import omegaconf
from typing import Union, List, Tuple, Literal

class IMDBDataset(torch.utils.data.Dataset):
    def __init__(self, data_config : omegaconf.DictConfig, split : Literal['train', 'valid', 'test']):
        """
        Inputs :
            data_config : omegaconf.DictConfig{
                model_name : str
                max_len : int
                valid_size : float
            }
            split : Literal['train', 'valid', 'test']
        Outputs : None
        """
        self.split = split
        dataset = load_dataset('stanfordnlp/imdb')
        tokenizer = AutoTokenizer.from_pretrained(data_config.model_name)

        # Train / Val / Test Split
        all_data = concatenate_datasets([dataset['train'], dataset['test']])
        all_data = all_data.shuffle(seed=42).select(range(50000))
        train_size = int(0.8 * len(all_data))
        val_size = int(0.1 * len(all_data))
        test_size = len(all_data) - train_size - val_size

        splits = {
            'train' : all_data.select(range(train_size)),
            'valid' : all_data.select(range(train_size, train_size + val_size)),
            'test' : all_data.select(range(train_size + val_size, train_size + val_size + test_size))
        }
        self.data = splits[split]

        # Tokenize
        self.tokenized_data = tokenizer(
            self.data['text'],
            padding='max_length',
            truncation=True,
            max_length=data_config.max_len,
            return_tensors='pt',
        )

        self.labels = torch.tensor(self.data['label'])

        print(f">> SPLIT : {self.split} | Total Data Length : ", len(self.data['text']))
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx) -> Tuple[dict, int]:
        """
        Inputs :
            idx : int
        Outputs :
            inputs : dict{
                input_ids : torch.Tensor
                token_type_ids : torch.Tensor
                attention_mask : torch.Tensor
            }
            label : int
        """
        inputs = {
            'input_ids' : self.tokenized_data['input_ids'][idx],
            'token_type_ids' : self.tokenized_data['token_type_ids'][idx],
            'attention_mask' : self.tokenized_data['attention_mask'][idx]
        }

        return inputs, self.labels[idx]

    @staticmethod
    def collate_fn(batch : List[Tuple[dict, int]]) -> dict:
        """
        Inputs :
            batch : List[Tuple[dict, int]]
        Outputs :
            data_dict : dict{
                input_ids : torch.Tensor
                token_type_ids : torch.Tensor
                attention_mask : torch.Tensor
                label : torch.Tensor
            }
        """
        input_ids = torch.stack([x[0]['input_ids'] for x in batch])
        attention_mask = torch.stack([x[0]['attention_mask'] for x in batch])
        token_type_ids = torch.stack([x[0]['token_type_ids'] for x in batch])
        labels = torch.tensor([x[1] for x in batch])

        return {
            'input_ids' : input_ids,
            'attention_mask' : attention_mask,
            'token_type_ids' : token_type_ids,
            'label' : labels
        }
    
def get_dataloader(data_config : omegaconf.DictConfig, split : Literal['train', 'valid', 'test']) -> torch.utils.data.DataLoader:
    """
    Output : torch.utils.data.DataLoader
    """
    dataset = IMDBDataset(data_config, split)
    dataloader = DataLoader(dataset, batch_size=data_config.batch_size, shuffle=(split=='train'), collate_fn=IMDBDataset.collate_fn)
    return dataloader