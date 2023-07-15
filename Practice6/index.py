from lightning.pytorch.cli import LightningCLI
from SentimentDataModule import SentimentDataModule
from RNN import SimpleRNN

def cli_main():
    cli = LightningCLI(model_class=SimpleRNN, datamodule_class=SentimentDataModule)

if __name__ == "__main__":
    cli_main()
