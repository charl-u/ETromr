import os
import torch
import lightning as L
from torch import nn as nn
from torch.functional import F
from editdistance import distance
from utils.CommonCTCUtils import CommonCTCUtil
from utils.Alignment import save_alignment_res

class CommonCTCModule(L.LightningModule):
    def __init__(
        self,
        encoder: torch.nn.Module,
        decoder: torch.nn.Module,
        decoder_out_dim: int,
        ctc_utils: CommonCTCUtil,
        lr,
        save_alignment=False
    ):
        super(CommonCTCModule, self).__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.decoder_out_dim = decoder_out_dim
        self.ctc_utils = ctc_utils
        self.lr = lr
        self.save_alignment = save_alignment
        
        self.note_emb = nn.Linear(decoder_out_dim, ctc_utils.num_notes)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        note_logits = self.note_emb(x)
        # note_logits = note_logits.permute(1, 0, 2)
        note_logits = F.log_softmax(note_logits, dim=2)
        value = {
            'note_logits': note_logits
        }
        return value

    def training_step(self, batch_data, batch_idx):
        images = batch_data['image']
        out = self(images)
        loss = self.calcute_loss(out, batch_data)
        self.log('loss', loss,on_step=True, on_epoch=True, prog_bar=False, logger=True)
        
        note_logits = out['note_logits']
        note_pred = self.ctc_utils.greedy_decode(note_logits)
        symbol_pred = [self.ctc_utils.detokenize(note_pred[i]) for i in range(len(batch_data['name']))]
        
        note_labels = batch_data['gt_tokens']
        symbol_gt = [self.ctc_utils.detokenize(note_labels[i][:batch_data['label_length'][i]].tolist()) for i in range(len(batch_data['name']))]
        
        eds = [min(distance(symbol_pred[i], symbol_gt[i]), len(symbol_gt[i])) for i in range(len(note_pred))]

        value = {
            'loss':loss,
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
        names = batch_data['name']
        images = batch_data['image']
        batch_size = images.size(0)
        out = self(images)

        note_logits = out['note_logits']
        note_pred = self.ctc_utils.greedy_decode(note_logits)
        symbol_pred = [self.ctc_utils.detokenize(note_pred[i]) for i in range(len(batch_data['name']))]
        
        note_labels = batch_data['gt_tokens']
        symbol_gt = [self.ctc_utils.detokenize(note_labels[i][:batch_data['label_length'][i]].tolist()) for i in range(len(batch_data['name']))]
        
        eds = [min(distance(symbol_pred[i], symbol_gt[i]), len(symbol_gt[i])) for i in range(len(note_pred))]

        ret_value = {
            'total_ed': sum(eds),
            'total_symbol': sum(batch_data['label_length'].tolist()),
            'total_sample': len(batch_data['name']),
            'ed_more_than_0': sum([1 if eds[i] > 0 else 0 for i in range(len(eds))]),
            'ed_more_than_1': sum([1 if eds[i] > 1 else 0 for i in range(len(eds))]),
            'ed_more_than_2': sum([1 if eds[i] > 2 else 0 for i in range(len(eds))]),
        }
        
        if self.save_alignment and self.logger.log_dir is not None:
            error_values = {
                "多检": 0,
                "漏检": 0,
                "类型": 0,
                "时值": 0,
                "谱号": 0,
                "调号": 0,
                "拍号": 0,
                "普通音高": 0,
                "语义音高": 0,
                "未知音高": 0,
                "其他": 0,
            }
            logdir = self.logger.log_dir
            for i in range(batch_size):
                alignment_fold = os.path.join(logdir, 'alignment')
                os.makedirs(alignment_fold, exist_ok=True)
                if eds[i] > 0:
                    target_path = os.path.join(alignment_fold, 'ed={}-{}.md'.format(eds[i], names[i]))
                    error_value = save_alignment_res(symbol_gt[i], symbol_pred[i], target_path)
                    for key in error_value.keys():
                        error_values[key] += error_value[key]
            
            ret_value.update(error_values)
        
        return ret_value
    def test_epoch_end(self, outputs):
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
        
        if self.save_alignment:
            for key in outputs[0].keys():
                if key not in ['total_ed', 'total_symbol', 'total_sample', 'ed_more_than_0', 'ed_more_than_1', 'ed_more_than_2']:
                    value = sum([outputs[i][key] for i in range(len(outputs)) if key in outputs[i]])
                    self.log(key, value, on_epoch=True, prog_bar=True, logger=True)

    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(params=self.parameters(), lr=self.lr)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=200)
        return [optimizer], [lr_scheduler]
        
    def calcute_loss(self, out, batch_data):
        labels = batch_data['gt_tokens']
        label_length = batch_data['label_length']
        pred_note_logits = out['note_logits']
        input_length = torch.full((len(label_length),), pred_note_logits.size(0), dtype=torch.long).to(pred_note_logits.device)
        loss = F.ctc_loss(pred_note_logits, labels, input_length, label_length, blank=0, reduction='mean', zero_infinity=True)
        return loss