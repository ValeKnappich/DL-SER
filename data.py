import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
from torch.nn.utils.rnn import pad_sequence
import json
import pandas as pd


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
            train_json = self.balance_classes(train_json)
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


    def balance_classes(self, train_json, n_remove=100):
        def remove(condition):
            removed = 0
            for index in list(train_json.keys()):
                sample = train_json[index]
                if condition(sample):
                    del train_json[index]
                    removed += 1
                if removed >= n_remove:
                    break

        
        print("Initial distribution:")
        valence_counts = pd.Series([sample["valence"] for sample in train_json.values()]).value_counts()
        activation_counts = pd.Series([sample["activation"] for sample in train_json.values()]).value_counts()
        print((valence_counts[0], valence_counts[1]), (activation_counts[0], activation_counts[1]))
        while True:
            # import pdb; pdb.set_trace()
            valence_counts = pd.Series([sample["valence"] for sample in train_json.values()]).value_counts()
            activation_counts = pd.Series([sample["activation"] for sample in train_json.values()]).value_counts()
            val_fac = max(valence_counts) / min(valence_counts)
            act_fac = max(activation_counts) / min(activation_counts)

            if val_fac > 1.3 and act_fac > 1.3:
                remove(lambda sample: sample["valence"] == valence_counts.idxmax() and sample["activation"] == activation_counts.idxmax())
            elif val_fac > 1.3:
                remove(lambda sample: sample["valence"] == valence_counts.idxmax())
            elif act_fac > 1.3:
                remove(lambda sample: sample["activation"] == activation_counts.idxmax())
            else:
                break
        print("Finished balancing:")
        valence_counts = pd.Series([sample["valence"] for sample in train_json.values()]).value_counts()
        activation_counts = pd.Series([sample["activation"] for sample in train_json.values()]).value_counts()
        print((valence_counts[0], valence_counts[1]), (activation_counts[0], activation_counts[1]))
        return train_json


if __name__ == "__main__":
    dm = SERDatamodule("train.json")