from lightning.pytorch.cli import LightningCLI
from SentimentDataModule import SentimentDataModule
from LSTM import LSTM
from GRU import GRU
from BLSTM import BLSTM
from LSTM_ATT import LSTM_ATT
from BGRU_ATT import BGRU_ATT

def cli_main():
    #cli = LightningCLI(model_class=LSTM, datamodule_class=SentimentDataModule)
    cli = LightningCLI(datamodule_class=SentimentDataModule)

if __name__ == "__main__":
    cli_main()
