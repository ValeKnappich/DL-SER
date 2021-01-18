import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F
from torch.optim import Adam
 
from functools import partial
from tqdm import tqdm


class SERClassifier(pl.LightningModule):
    def __init__(self, n_lstm=5, hidden_size=512, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=26, num_layers=n_lstm, hidden_size=hidden_size, batch_first=True, dropout=dropout)
        self.linear_act = nn.Linear(hidden_size, 2)
        self.linear_val = nn.Linear(hidden_size, 2)
    
        self.log_console = partial(self.log, prog_bar=True, logger=False)
        self.train_acc_act = pl.metrics.Accuracy()
        self.train_acc_val = pl.metrics.Accuracy()
        self.val_acc_act = pl.metrics.Accuracy()
        self.val_acc_val = pl.metrics.Accuracy()
    
    
    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-5)
 
 
    def forward(self, features, feature_lens):
        # pack padded features, so that lstm can handle variable length
        features_packed = pack_padded_sequence(features, feature_lens.cpu(), batch_first=True, enforce_sorted=False)
        # pass through lstm
        lstm_out, (lstm_hidden, lstm_cell) = self.lstm(features_packed)
        # pass last hidden state through linear layers
        print(lstm_cell[-1].shape)
        import pdb; pdb.set_trace()
        logits_act = self.linear_act(lstm_cell[-1])
        logits_val = self.linear_val(lstm_cell[-1])
        # import pdb; pdb.set_trace()
        return logits_act, logits_val
 
 
    def training_step(self, batch, batch_idx):
        batch, feature_lens = batch
        features = batch["features"]
        valence = batch["activation"]
        activation = batch["valence"]
        # run forward
        logits_act, logits_val = self(features, feature_lens)
        # get loss
        loss_act = F.cross_entropy(logits_act, valence)
        loss_val = F.cross_entropy(logits_val, activation)
        # get accuracy
        self.train_acc_act(F.log_softmax(logits_act, dim=1), valence)
        self.train_acc_val(F.log_softmax(logits_val, dim=1), activation)
        self.log_console("train_acc_act", self.train_acc_act, on_step=True, on_epoch=True)
        self.log_console("train_acc_val", self.train_acc_val, on_step=True, on_epoch=True)
        # return combined loss
        return (loss_act + loss_val) / 2
 
 
    def validation_step(self, batch, batch_idx):
        batch, feature_lens = batch
        features = batch["features"]
        valence = batch["activation"]
        activation = batch["valence"]
        # run forward
        logits_act, logits_val = self(features, feature_lens)
        # get loss
        loss_act = F.cross_entropy(logits_act, valence)
        loss_val = F.cross_entropy(logits_val, activation)
        # get accuracy
        self.val_acc_act(torch.argmax(F.log_softmax(logits_act, dim=1), dim=1), valence)
        self.val_acc_val(torch.argmax(F.log_softmax(logits_val, dim=1), dim=1), activation)
        self.log_console("val_acc_act", self.val_acc_act, on_step=False, on_epoch=True)
        self.log_console("val_acc_val", self.val_acc_val, on_step=False, on_epoch=True)
        self.log_console("val_loss", (loss_act + loss_val) / 2)
 

    def transform(self, dataloader):
        self.eval()
        result = {}
        i = 0
        for batch, feature_lens in tqdm(dataloader):  
            logits_act, logits_val = self(batch["features"].cuda(), feature_lens)
            activations = torch.argmax(F.log_softmax(logits_act, dim=1), dim=1)
            valences = torch.argmax(F.log_softmax(logits_val, dim=1), dim=1)
            # import pdb; pdb.set_trace()
            for j, (activation, valence) in enumerate(zip(activations, valences)):
                result[str(i+j)] = {"activation": activation.item(), "valence": valence.item()}
            i += j
        return result