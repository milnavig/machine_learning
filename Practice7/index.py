from lightning.pytorch.cli import LightningCLI
from SentimentDataModule import SentimentDataModule
from LSTM import LSTM

def cli_main():
    cli = LightningCLI(model_class=LSTM, datamodule_class=SentimentDataModule)

if __name__ == "__main__":
    cli_main()
