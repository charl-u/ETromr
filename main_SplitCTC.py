import warnings
warnings.filterwarnings("ignore")
from lightning.pytorch.cli import LightningCLI
import lightning as L

from model.SplitCTC.SplitCTCModule import SplitCTCModule
from data.SplitCTC.SplitCTCDataModule import SplitCTCDataModule

def cli_main():
    L.seed_everything(0)
    cli = LightningCLI(SplitCTCModule, SplitCTCDataModule)
    
if __name__ == '__main__':
    cli_main()

