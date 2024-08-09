import inspect
import importlib
import pickle as pkl
import lightning as L
from torch.utils.data import DataLoader
from data.CommonCTC.CommonCTCDataSet import CommonCTCDataSet

class CommonCTCDataModule(L.LightningDataModule):
    def __init__(self,
                 root_dir,
                 img_suffix,
                 label_suffix,
                 max_len,
                 batch_size,
                 num_workers,
                 pin_memory=True,
                 shuffle=True,
                 ):
        super(CommonCTCDataModule, self).__init__()
        self.root_dir = root_dir
        self.img_suffix = img_suffix
        self.label_suffix = label_suffix
        self.max_len = max_len

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.shuffle = shuffle

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.trainset = CommonCTCDataSet(data_root=self.root_dir,
                                             img_suffix=self.img_suffix,
                                             label_suffix=self.label_suffix,
                                             max_len=self.max_len,
                                             train=True)
            self.valset = CommonCTCDataSet(data_root=self.root_dir,
                                             img_suffix=self.img_suffix,
                                             label_suffix=self.label_suffix,
                                             max_len=self.max_len,
                                             train=False)
        if stage == 'test' or stage is None:
            self.testset = CommonCTCDataSet(data_root=self.root_dir,
                                             img_suffix=self.img_suffix,
                                             label_suffix=self.label_suffix,
                                             max_len=self.max_len,
                                             train=False)
        if stage == 'validate' or stage is None:
            self.valset = CommonCTCDataSet(data_root=self.root_dir,
                                    img_suffix=self.img_suffix,
                                    label_suffix=self.label_suffix,
                                    max_len=self.max_len,
                                    train=False)
    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=self.shuffle, drop_last=False, pin_memory=self.pin_memory)

    def val_dataloader(self):
        return DataLoader(self.valset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, drop_last=False, pin_memory=self.pin_memory)

    def test_dataloader(self) :
        return DataLoader(self.testset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, drop_last=False, pin_memory=self.pin_memory)
