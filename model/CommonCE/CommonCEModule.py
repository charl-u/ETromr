import torch
import lightning as L
from torch import nn as nn
from torch.functional import F
from editdistance import distance

class CommonCEModule(L.LightningModule):
    def __init__(
        self,
        encoder: torch.nn.Module,
        decoder: torch.nn.Module,
        ce_utils,
        lr
    ):
        super(CommonCEModule, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.ce_utils = ce_utils
        self.lr = lr
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        note_logits = self.note_emb(x)
        note_logits = F.log_softmax(note_logits, dim=2)
        value = {
            'note_logits': note_logits
        }
        return value

    def forward_test(self, x):
        pass
    
    def training_step(self, batch_data, batch_idx):
        images = batch_data['image']
        out = self(images)
        loss = self.calcute_loss(out, batch_data)
        self.log('loss', loss,on_step=True, on_epoch=True, prog_bar=False, logger=True)

    def training_epoch_end(self, outputs):
        pass

    def validation_step(self, batch_data, batch_idx):
        images = batch_data['image']
        out = self(images)

    def validation_epoch_end(self, outputs):
        pass

    def test_step(self, batch_data, batch_idx):
        return self.validation_step(batch_data, batch_idx)

    def test_epoch_end(self, outputs):
        return self.validation_epoch_end(outputs)

    def configure_optimizers(self):
        return torch.optim.Adam(params=self.parameters(), lr=self.lr)

    def calcute_loss(self, out, batch_data):
        pass