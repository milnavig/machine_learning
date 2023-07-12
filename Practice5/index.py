import lightning.pytorch as pl
import torch
import torch.nn.functional as F
import torchmetrics
import pandas as pd
from lightning.pytorch.cli import LightningCLI
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import ToTensor


# https://www.kaggle.com/competitions/digit-recognizer/data
class DigitsDataset(Dataset):
    def __init__(self, data):
        self.data = data
        # self.transform = ToTensor()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        features = row[1:]  # Предполагаем, что первый столбец - это метки, а остальные - признаки
        features = torch.from_numpy(features.to_numpy()).to(torch.float32)
        label = row[0]
        # features = self.transform(features)
        return features, label


class DigitsDataModule(pl.LightningDataModule):
    def __init__(self, csv_file, batch_size=64):
        super().__init__()
        self.csv_file = csv_file
        self.batch_size = batch_size

    def prepare_data(self):
        # Здесь вы можете выполнить любую предварительную обработку данных, если это необходимо
        pass

    def setup(self, stage=None):
        # Чтение данных из csv файла
        data = pd.read_csv(self.csv_file)
        # Разделение данных на тренировочный и валидационный наборы (и, при необходимости, тестовый набор)
        # Здесь можно выполнить дополнительную предварительную обработку данных, разделение и т. д.

        # Получение количества строк в файле
        num_rows = data.shape[0]

        # Вычисление размеров каждой части
        train_size = int(0.8 * num_rows)
        val_size = int(0.1 * num_rows)
        test_size = num_rows - train_size - val_size

        # Разделение данных на тренировочный, валидационный и тестовый наборы
        self.train_data = data.iloc[:train_size]
        self.val_data = data.iloc[train_size:train_size + val_size]
        self.test_data = data.iloc[train_size + val_size:]

        # Создание экземпляров наборов данных для тренировки и валидации
        self.train_dataset = DigitsDataset(self.train_data)
        self.val_dataset = DigitsDataset(self.val_data)
        self.test_dataset = DigitsDataset(self.test_data)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

class MLP(pl.LightningModule):
    def __init__(self, classes=10):
        super().__init__()
        self.save_hyperparameters()
        self.classes = classes

        # new PL attributes:
        self.train_acc = torchmetrics.Accuracy(task='multiclass', num_classes=self.hparams.classes)
        self.valid_acc = torchmetrics.Accuracy(task='multiclass', num_classes=self.hparams.classes)
        self.test_acc = torchmetrics.Accuracy(task='multiclass', num_classes=self.hparams.classes)

        self.validation_step_outputs = []
        self.l1 = torch.nn.Linear(28 * 28, self.hparams.classes)

    def forward(self, x):
        return torch.relu(self.l1(x))

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.train_acc.update(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.valid_acc.update(y_hat, y)
        self.log("valid_loss", loss, prog_bar=True)
        self.log("valid_acc", self.valid_acc.compute(), prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.test_acc.update(y_hat, y)
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", self.test_acc.compute(), prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        self.log("train_acc", self.train_acc.compute())

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.001, momentum=0.9, nesterov=False)

def cli_main():
    cli = LightningCLI(MLP, DigitsDataModule, save_config_kwargs={"config_filename": "config.yaml"})

if __name__ == "__main__":
    cli_main()
