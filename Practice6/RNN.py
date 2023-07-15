import torch
import torch.nn as nn
import lightning.pytorch as pl

class SimpleRNN(pl.LightningModule):
    def __init__(self, input_size, hidden_size, classes):
        super().__init__()
        self.save_hyperparameters()
        #self.hidden_size = hidden_size

        self.rnn = nn.RNN(self.hparams.input_size, self.hparams.hidden_size, batch_first=True)
        self.fc = nn.Linear(self.hparams.hidden_size, self.hparams.classes)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        batch_size = x.size(0)
        hidden = self.init_hidden(batch_size)

        x = x.to(self.device)
        hidden = hidden.to(self.device)

        output, _ = self.rnn(x, hidden)
        output = output[:, -1, :]  # Берем только последний выход RNN
        output = self.fc(output)

        return output

    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hparams.hidden_size)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)
        loss = self.criterion(output, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)
        loss = self.criterion(output, y)
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)
        loss = self.criterion(output, y)
        self.log('test_loss', loss)