import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
import torchmetrics

class LSTM_ATT(pl.LightningModule):
    def __init__(self, input_size, hidden_size, classes):
        super().__init__()
        self.save_hyperparameters()
        #self.hidden_size = hidden_size

        self.train_acc = torchmetrics.Accuracy(task='multiclass', num_classes=self.hparams.classes)
        self.valid_acc = torchmetrics.Accuracy(task='multiclass', num_classes=self.hparams.classes)
        self.test_acc = torchmetrics.Accuracy(task='multiclass', num_classes=self.hparams.classes)

        self.lstm = nn.LSTM(self.hparams.input_size, self.hparams.hidden_size, batch_first=True)

        # Attention layer
        self.attention = nn.Linear(self.hparams.hidden_size, 1)

        self.fc = nn.Linear(self.hparams.hidden_size, self.hparams.classes)

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hparams.hidden_size).to(self.device)
        c0 = torch.zeros(1, x.size(0), self.hparams.hidden_size).to(self.device)
        out, _ = self.lstm(x, (h0, c0))

        # Attention mechanism
        attention_scores = self.attention(out).squeeze(-1)
        attention_weights = F.softmax(attention_scores, dim=1)
        attention_output = torch.bmm(out.transpose(1, 2), attention_weights.unsqueeze(2)).squeeze(2)

        out = self.fc(attention_output)

        return out

    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hparams.hidden_size)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)
        loss = self.criterion(output, y)
        self.train_acc.update(output, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)
        loss = self.criterion(output, y)
        self.valid_acc.update(output, y)
        self.log('val_loss', loss)
        self.log("valid_acc", self.valid_acc.compute(), prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)
        loss = self.criterion(output, y)
        self.test_acc.update(output, y)
        self.log('test_loss', loss)
        self.log("test_acc", self.test_acc.compute(), prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        self.log("train_acc", self.train_acc.compute())