import warnings
warnings.filterwarnings("ignore")
from lightning.pytorch.cli import LightningCLI
import lightning as L

from model.CommonCTC.CommonCTCModule import CommonCTCModule
from data.CommonCTC.CommonCTCDataModule import CommonCTCDataModule

def cli_main():
    L.seed_everything(0)
    cli = LightningCLI(CommonCTCModule, CommonCTCDataModule)
    
if __name__ == '__main__':
    cli_main()

