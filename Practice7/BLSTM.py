import torch
import torch.nn as nn
import lightning.pytorch as pl
import torchmetrics
import seaborn as sns

class BLSTM(pl.LightningModule):
    def __init__(self, input_size, hidden_size, classes):
        super().__init__()
        self.save_hyperparameters()
        #self.hidden_size = hidden_size

        self.validation_step_outputs = []

        self.train_acc = torchmetrics.Accuracy(task='multiclass', num_classes=self.hparams.classes)
        self.valid_acc = torchmetrics.Accuracy(task='multiclass', num_classes=self.hparams.classes)
        self.test_acc = torchmetrics.Accuracy(task='multiclass', num_classes=self.hparams.classes)

        self.lstm = nn.LSTM(self.hparams.input_size, self.hparams.hidden_size, batch_first=True, bidirectional=True)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.fc1 = nn.Linear(self.hparams.hidden_size * 2, 300)  # Умножаем на 2, так как LSTM bidirectional
        self.fc2 = nn.Linear(300, self.hparams.classes)

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        h0 = torch.zeros(2, x.size(0), self.hparams.hidden_size).to(self.device)
        c0 = torch.zeros(2, x.size(0), self.hparams.hidden_size).to(self.device)
        out, _ = self.lstm(x, (h0, c0))

        out = self.relu(out[:, -1, :])
        out = self.dropout(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = torch.softmax(out, dim=1)

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

        self.validation_step_outputs.append({'loss': loss, 'pred': output, 'target': y})

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

    def on_validation_epoch_end(self):
        # Получение предсказаний и меток классов из списка outputs
        predictions = torch.cat([x['pred'] for x in self.validation_step_outputs])
        targets = torch.cat([x['target'] for x in self.validation_step_outputs])

        # Вычисление матрицы ошибок
        cm = torchmetrics.functional.confusion_matrix(predictions, targets, task='multiclass', num_classes=self.hparams.classes)
        cm = cm.cpu()

        heatmap = sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square=True, cmap='Blues_r').get_figure()

        # Сохранение изображения в TensorBoard
        self.logger.experiment.add_figure("Confusion Matrix", heatmap, self.current_epoch)