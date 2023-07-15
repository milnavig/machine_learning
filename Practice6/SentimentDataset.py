import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class SentimentDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        vector = row[1]  # Предполагаем, что первый столбец - это метки, а остальные - признаки
        sentiment = row[2]

        # Преобразование вектора в тензор
        vector = torch.tensor(vector, dtype=torch.float32)

        # Заполнение предложений до максимальной длины
        # vector = pad_sequence(vector, batch_first=True)

        return vector, sentiment