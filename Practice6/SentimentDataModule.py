import lightning.pytorch as pl
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from SentimentDataset import SentimentDataset

class SentimentDataModule(pl.LightningDataModule):
    def __init__(self, csv_file, batch_size=64):
        super().__init__()
        self.save_hyperparameters()
        self.csv_file = csv_file
        self.batch_size = batch_size

    def prepare_data(self):
        # Здесь вы можете выполнить любую предварительную обработку данных, если это необходимо
        pass

    def setup(self, stage=None):
        # Чтение данных из csv файла
        data = pd.read_csv(self.hparams.csv_file)
        # Разделение данных на тренировочный и валидационный наборы (и, при необходимости, тестовый набор)
        # Здесь можно выполнить дополнительную предварительную обработку данных, разделение и т. д.

        # Преобразование столбца в список
        sentences = data['vector'].tolist()

        # Преобразование предложений в тензоры и заполнение до максимальной длины
        padded_sentences = pad_sequence([torch.tensor(eval(sentence)) for sentence in sentences], batch_first=True)

        # Преобразование тензоров обратно в список
        padded_sentences_list = padded_sentences.tolist()

        # Изменение значения столбца в DataFrame
        data['vector'] = padded_sentences_list

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
        self.train_dataset = SentimentDataset(self.train_data)
        self.val_dataset = SentimentDataset(self.val_data)
        self.test_dataset = SentimentDataset(self.test_data)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)