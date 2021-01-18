import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
import json


class ColumnDataSet(Dataset):
    """Generic Dataset, that holds a dict with columns"""
 
    def __init__(self, data: dict):
        self.data = data
        self.fields = list(data.keys())
 
    def __len__(self) -> int:
        return len(self.data[self.fields[0]])
 
    def __getitem__(self, idx: int) -> dict:
        return {field: self.data[field][idx] for field in self.data}
 
 
class SERDatamodule(pl.LightningDataModule):
    def __init__(self, path, batch_size=64, split=0.7, train=True):
        super().__init__()
        self.path = path
        self.batch_size = batch_size
        self.split = split
        self.train_mode = train
    
        self.setup()
  
 
    def setup(self):
        train_json = json.load(open(self.path, "r"))
        if self.train_mode:
            train_columnar = {
                "features": [sample["features"] for sample in train_json.values()],
                "activation": [sample["activation"] for sample in train_json.values()],
                "valence": [sample["valence"] for sample in train_json.values()]
            }
        else:
            train_columnar = {
                "features": [sample["features"] for sample in train_json.values()]
            }
        train = ColumnDataSet(train_columnar)
        if not self.split:
            self.train = train
        else:
            len_train_split = int(self.split * len(train))  
            len_val_split = len(train) - len_train_split
            self.train, self.val = random_split(train, (len_train_split, len_val_split))
    
 
    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True, num_workers=4, collate_fn=self.pad)
 
 
    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, shuffle=False, num_workers=4, collate_fn=self.pad)
 
 
    def pad(self, batch_list):
        # Used as collate_fn in Dataloaders
        feature_lens = torch.LongTensor([len(sample["features"]) for sample in batch_list])
        features = pad_sequence(
            [torch.Tensor(sample["features"]) for sample in batch_list],
            batch_first=True, padding_value=0
        )
        if self.train_mode:
            activations = torch.LongTensor([sample["activation"] for sample in batch_list])
            valences = torch.LongTensor([sample["valence"] for sample in batch_list])
            batch = {
                "features": features,
                "activation": activations,
                "valence": valences
            }
        else:
            batch = {"features": features}
        return batch, feature_lens