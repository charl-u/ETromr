import torch
import numpy as np


class CommonCTCUtil:
    def __init__(self, vocab_path):
        self.vocab_path = vocab_path
        self.load_vocab()

    def load_vocab(self):
        with open(self.vocab_path, 'r') as f:
            voc = f.read().splitlines()
        if voc is None:
            raise Exception('No vocabulary file found')
        if len(voc) == 0:
            raise Exception('Vocabulary is empty')
        voc.insert(0, '[BLANK]')
        self.num_notes = len(voc)
        self.symbol2index = {s: i for i, s in enumerate(voc)}
        self.index2symbol = {i: s for i, s in enumerate(voc)}

    def _tokenize(self, tokens, tokens2idx_dict, max_len=64):
        label = np.full(shape=(max_len,), fill_value=tokens2idx_dict['[BLANK]'], dtype=np.int32)
        label_list = []
        for token in tokens:
            if token in tokens2idx_dict:
                label_list.append(tokens2idx_dict[token])
            else:
                print('[Dataset] Symbol not found in vocabulary: {}'.format(token))
        label[:len(label_list)] = np.array(label_list)
        return label
    
    def tokenize(self, labels, max_len=64):
        return self._tokenize(labels, self.symbol2index, max_len=max_len)
    
    
    def detokenize(self, tokens):
        symbols = [self.index2symbol[idx] for idx in tokens]
        return symbols

    def greedy_decode(self, batch_pred_res, blank_val = 0):
        """
        贪心解码
        pred_res = (seq len, batch size, vocab size)
        lengths = (batch size list where length of corresponding seq)
        return = (batch size list of lists where greedily decoded)
        """
        predictions = []
        batch_pred_res = batch_pred_res.permute(1, 0, 2)
        for batch_res in batch_pred_res:
            # (W, vocab size)
            seq = torch.max(batch_res, 1)[1]

            batch_decode_res = []
            prev = -1
            for s in seq:
                if s == blank_val:
                    prev = -1
                    continue
                elif s == prev:
                    continue
                batch_decode_res.append(s.item())
                prev = s
            predictions.append(batch_decode_res)

        return predictions