import torch
import lightning as L
from torch import nn as nn
from torch.functional import F
from editdistance import distance
from utils.CommonCTCUtils import CommonCTCUtil

class SplitCEModule(L.LightningModule):
    def __init__(
        self,
        encoder: torch.nn.Module,
        decoder: torch.nn.Module,
        ctc_utils: CommonCTCUtil,
        lr
    ):
        super(SplitCEModule, self).__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.ctc_utils = ctc_utils
        self.lr = lr
        
        self.to_pitch = nn.Linear(in_features=decoder.output_dim, out_features=ctc_utils.num_pitch)
        self.to_lift = nn.Linear(in_features=decoder.output_dim, out_features=ctc_utils.num_lift)
        self.to_rhythm = nn.Linear(in_features=decoder.output_dim, out_features=ctc_utils.num_rhythm)
        self.to_note = nn.Linear(in_features=decoder.output_dim, out_features=ctc_utils.num_note)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        note_logits = self.to_note(x)
        note_logits = F.log_softmax(note_logits, dim=2)

        pitch_logits = self.to_pitch(x)
        pitch_logits = F.log_softmax(pitch_logits, dim=2)

        rhythm_logits = self.to_rhythm(x)
        rhythm_logits = F.log_softmax(rhythm_logits, dim=2)

        lift_logits = self.to_lift(x)
        lift_logits = F.log_softmax(lift_logits, dim=2)

        value = {
            'note_logits': note_logits,
            'pitch_logits': pitch_logits,
            'rhythm_logits': rhythm_logits,
            'lift_logits': lift_logits
        }
        return value

    def training_step(self, batch_data, batch_idx):
        images = batch_data['image']
        out = self(images)
        loss_dict = self.calcute_loss(out, batch_data)
        self.log('loss', loss_dict['loss'], on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('loss_pitch', loss_dict['loss_pitch'], on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('loss_rhythm', loss_dict['loss_rhythm'], on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('loss_lift', loss_dict['loss_lift'], on_step=True, on_epoch=True, prog_bar=False, logger=True)
                
        note_logits = out['note_logits']
        note_pred = self.ctc_utils.greedy_decode(note_logits)
        symbol_pred = [self.ctc_utils.detokenize(note_pred[i]) for i in range(len(batch_data['name']))]
        
        note_labels = batch_data['gt_tokens']
        symbol_gt = [self.ctc_utils.detokenize(note_labels[i][:batch_data['label_length'][i]].tolist()) for i in range(len(batch_data['name']))]
        
        eds = [min(distance(symbol_pred[i], symbol_gt[i]), len(symbol_gt[i])) for i in range(len(note_pred))]

        value = {
            'loss':loss_dict['loss'],
            'total_ed': sum(eds),
            'total_symbol': sum(batch_data['label_length'].tolist()),
            'total_sample': len(batch_data['name']),
            'ed_more_than_0': sum([1 if eds[i] > 0 else 0 for i in range(len(eds))]),
            'ed_more_than_1': sum([1 if eds[i] > 1 else 0 for i in range(len(eds))]),
            'ed_more_than_2': sum([1 if eds[i] > 2 else 0 for i in range(len(eds))]),
        }
        return value
        

    def training_epoch_end(self, outputs):
        total_symbol = sum([outputs[i]['total_symbol'] for i in range(len(outputs))])
        total_sample = sum([outputs[i]['total_sample'] for i in range(len(outputs))])
        total_ed = sum([outputs[i]['total_ed'] for i in range(len(outputs))])
        total_ed_more_than_0 = sum([outputs[i]['ed_more_than_0'] for i in range(len(outputs))])
        total_ed_more_than_1 = sum([outputs[i]['ed_more_than_1'] for i in range(len(outputs))])
        total_ed_more_than_2 = sum([outputs[i]['ed_more_than_2'] for i in range(len(outputs))])
        avg_ed = total_ed / total_symbol
        avg_ed_more_than_0 = total_ed_more_than_0 / total_sample
        avg_ed_more_than_1 = total_ed_more_than_1 / total_sample
        avg_ed_more_than_2 = total_ed_more_than_2 / total_sample

        dict = {
            'T_SymbER': avg_ed,
            'T_ER>0': avg_ed_more_than_0,
            'T_ER>1': avg_ed_more_than_1,
            'T_ER>2': avg_ed_more_than_2,
            'total_ed': total_ed,
            'total_symbol': total_symbol,
            'total_sample': total_sample,
        }
        self.log_dict(dict, on_epoch=True, prog_bar=True, logger=True)

    def validation_step(self, batch_data, batch_idx):
        images = batch_data['image']
        out = self(images)

        note_logits = out['note_logits']
        note_pred = self.ctc_utils.greedy_decode(note_logits)
        symbol_pred = [self.ctc_utils.detokenize(note_pred[i]) for i in range(len(batch_data['name']))]
        
        note_labels = batch_data['gt_tokens']
        symbol_gt = [self.ctc_utils.detokenize(note_labels[i][:batch_data['label_length'][i]].tolist()) for i in range(len(batch_data['name']))]
        
        eds = [min(distance(symbol_pred[i], symbol_gt[i]), len(symbol_gt[i])) for i in range(len(note_pred))]

        value = {
            'total_ed': sum(eds),
            'total_symbol': sum(batch_data['label_length'].tolist()),
            'total_sample': len(batch_data['name']),
            'ed_more_than_0': sum([1 if eds[i] > 0 else 0 for i in range(len(eds))]),
            'ed_more_than_1': sum([1 if eds[i] > 1 else 0 for i in range(len(eds))]),
            'ed_more_than_2': sum([1 if eds[i] > 2 else 0 for i in range(len(eds))]),
        }
        return value
    
    def validation_epoch_end(self, outputs):
        total_symbol = sum([outputs[i]['total_symbol'] for i in range(len(outputs))])
        total_sample = sum([outputs[i]['total_sample'] for i in range(len(outputs))])
        total_ed = sum([outputs[i]['total_ed'] for i in range(len(outputs))])
        total_ed_more_than_0 = sum([outputs[i]['ed_more_than_0'] for i in range(len(outputs))])
        total_ed_more_than_1 = sum([outputs[i]['ed_more_than_1'] for i in range(len(outputs))])
        total_ed_more_than_2 = sum([outputs[i]['ed_more_than_2'] for i in range(len(outputs))])
        avg_ed = total_ed / total_symbol
        avg_ed_more_than_0 = total_ed_more_than_0 / total_sample
        avg_ed_more_than_1 = total_ed_more_than_1 / total_sample
        avg_ed_more_than_2 = total_ed_more_than_2 / total_sample
        self.log('total_symbol', total_symbol, on_epoch=True, prog_bar=True, logger=True)
        self.log('total_sample', total_sample, on_epoch=True, prog_bar=True, logger=True)
        self.log('V_SymbER', avg_ed, on_epoch=True, prog_bar=True, logger=True)
        self.log('V_ER>0', avg_ed_more_than_0, on_epoch=True, prog_bar=True, logger=True)
        self.log('V_ER>1', avg_ed_more_than_1, on_epoch=True, prog_bar=True, logger=True)
        self.log('V_ER>2', avg_ed_more_than_2, on_epoch=True, prog_bar=False, logger=True)

    def test_step(self, batch_data, batch_idx):
        return self.validation_step(batch_data, batch_idx)

    def test_epoch_end(self, outputs):
        return self.validation_epoch_end(outputs)
    
    def configure_optimizers(self):
        return torch.optim.Adam(params=self.parameters(), lr=self.lr)
    
    def calcute_loss(self, out, batch_data):
        labels = batch_data['gt_tokens']
        label_length = batch_data['label_length']
        pred_note_logits = out['note_logits']
        input_length = torch.full((len(label_length),), pred_note_logits.size(0), dtype=torch.long).to(pred_note_logits.device)
        loss = F.ctc_loss(pred_note_logits, labels, input_length, label_length, blank=0, reduction='mean', zero_infinity=True)
        return loss