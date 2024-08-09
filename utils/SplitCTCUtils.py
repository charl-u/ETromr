import torch
import numpy as np
import json

class SplitCTCUtil:
    def __init__(self, lift_path, note_path, pitch_path, rhythm_path, max_len=64):
        self.lift_path = lift_path
        self.note_path = note_path
        self.pitch_path = pitch_path
        self.rhythm_path = rhythm_path
        self.load_vocab()

    def load_vocab(self):
        with open(self.lift_path, 'r') as f:
            lift_tokenizer_dict = json.load(f)
            self.lift_tokens = list(lift_tokenizer_dict["model"]["vocab"].keys())

        with open(self.pitch_path, 'r') as f:
            pitch_tokenizer_dict = json.load(f)
            self.pitch_tokens = list(pitch_tokenizer_dict["model"]["vocab"].keys())
        with open(self.rhythm_path, 'r') as f:
            rhythm_tokenizer_dict = json.load(f)
            self.rhythm_tokens = list(rhythm_tokenizer_dict["model"]["vocab"].keys())
        with open(self.note_path, 'r') as f:
            note_tokenizer_dict = json.load(f)
            self.note_tokens = list(note_tokenizer_dict["model"]["vocab"].keys())

        self.lift_tokens.insert(0, '[BLANK]')
        self.pitch_tokens.insert(0, '[BLANK]')
        self.rhythm_tokens.insert(0, '[BLANK]')
        self.note_tokens.insert(0, '[BLANK]')

        self.lift2index = {token: idx for idx, token in enumerate(self.lift_tokens)}
        self.index2lift = {idx: token for idx, token in enumerate(self.lift_tokens)}
        self.pitch2index = {token: idx for idx, token in enumerate(self.pitch_tokens)}
        self.index2pitch = {idx: token for idx, token in enumerate(self.pitch_tokens)}
        self.rhythm2index = {token: idx for idx, token in enumerate(self.rhythm_tokens)}
        self.index2rhythm = {idx: token for idx, token in enumerate(self.rhythm_tokens)}
        self.note2index = {token: idx for idx, token in enumerate(self.note_tokens)}
        self.index2note = {idx: token for idx, token in enumerate(self.note_tokens)}
        self.num_lift = len(self.lift_tokens)
        self.num_pitch = len(self.pitch_tokens)
        self.num_rhythm = len(self.rhythm_tokens)
        self.num_note = len(self.note_tokens)


    def _split_labels(self, labels):
        lift_tokens = []
        pitch_tokens = []
        rhythm_tokens = []
        note_tokens = []

        for label in labels:
            if label == '[UNKNOWN]' or label == 'tercet':
                lift_tokens.append('nonote')
                note_tokens.append('nonote')
                rhythm_tokens.append('[UNKNOWN]')
                pitch_tokens.append('nonote')
            elif not label.startswith('note'):
                lift_tokens.append('nonote')
                note_tokens.append('nonote')
                rhythm_tokens.append(label)
                pitch_tokens.append('nonote')    
            else:
                pitch = label.split('_')[0]
                rhythm = "_".join(label.split('_')[1:])
                clear_pitch = pitch.translate(str.maketrans({'#' : None, 'b' : None,'N' : None}))

                note_tokens.append('note')
                pitch_tokens.append(clear_pitch)
                rhythm_tokens.append('note-' + rhythm)

                if '##' in pitch:
                    lift_tokens.append('lift_##')
                elif 'bb' in pitch:
                    lift_tokens.append('lift_bb')
                elif '#' in pitch:
                    lift_tokens.append('lift_#')
                elif 'b' in pitch:
                    lift_tokens.append('lift_b')
                elif 'N' in pitch:
                    lift_tokens.append('lift_N')
                else: 
                    lift_tokens.append('lift_null')

        values = {
            'lift': lift_tokens,
            'pitch': pitch_tokens,
            'rhythm': rhythm_tokens,
            'note': note_tokens
        }
        return values

    def _tokenize(self, tokens, tokens2idx_dict, max_len=64):
        label = np.full(shape=(max_len,), fill_value=0, dtype=np.int32)
        label_list = []
        for token in tokens:
            if token in tokens2idx_dict:
                label_list.append(tokens2idx_dict[token])
            else:
                print('[Dataset] Symbol not found in vocabulary: {}'.format(token))
        label[:len(label_list)] = np.array(label_list)
        return label
    
    def tokenize(self, semantic_labels, max_len=64):
        split_res = self._split_labels(semantic_labels)
        lift_tokens = split_res['lift']
        pitch_tokens = split_res['pitch']
        rhythm_tokens = split_res['rhythm']
        note_tokens = split_res['note']

        lift_tokens = self._tokenize(lift_tokens, self.lift2index, max_len=max_len)
        pitch_tokens = self._tokenize(pitch_tokens, self.pitch2index, max_len=max_len)
        rhythm_tokens = self._tokenize(rhythm_tokens, self.rhythm2index, max_len=max_len)
        note_tokens = self._tokenize(note_tokens, self.note2index, max_len=max_len)

        value = {
            'lift': lift_tokens,
            'pitch': pitch_tokens,
            'rhythm': rhythm_tokens,
            'note': note_tokens
        }
        return value
    
    def detokenize(self, pred_lift, pred_rhythm, pred_pitch, max_len=None):
        pitch_symbols = [self.index2pitch[idx] for idx in pred_pitch]
        rhythm_symbols = [self.index2rhythm[idx] for idx in pred_rhythm]
        lift_symbols = [self.index2lift[idx] for idx in pred_lift]

        merge = []
        max_len = min(max_len, len(lift_symbols)) if max_len else len(lift_symbols)
        if len(rhythm_symbols) > 0:
            merge = [rhythm_symbols[0]]
            for j in range(1, max_len):
                if rhythm_symbols[j].startswith('note'):
                    lift = ''
                    if lift_symbols[j] in ("lift_##", "lift_#", "lift_bb", "lift_b", "lift_N"):
                        lift = lift_symbols[j].split('_')[-1]
                    token = pitch_symbols[j][:-1] + lift + pitch_symbols[j][-1] + "_"+rhythm_symbols[j].split('note-')[-1]
                    merge.append(token)
                elif rhythm_symbols[j] != 'nonote':
                    merge.append(rhythm_symbols[j])
        return merge

    def greedy_decode_split(self, batch_pitch_res, batch_note_res, batch_rhythm_res, batch_lift_res, blank_val=0):
        predictions = []
        batch_pitch_res = batch_pitch_res.permute(1, 0, 2)
        batch_note_res = batch_note_res.permute(1, 0, 2)
        batch_rhythm_res = batch_rhythm_res.permute(1, 0, 2)
        batch_lift_res = batch_lift_res.permute(1, 0, 2)


        for pitch, note, rhythm, lift in zip(batch_pitch_res, batch_note_res, batch_rhythm_res, batch_lift_res):
            # (W, vocab size)
            pitch_seq = torch.max(pitch, 1)[1]
            note_seq = torch.max(note, 1)[1]
            rhythm_seq = torch.max(rhythm, 1)[1]
            lift_seq = torch.max(lift, 1)[1]

            batch_pitch_res = []
            batch_note_res = []
            batch_rhythm_res = []
            batch_lift_res = []

            prev_r = -1            
            
            for p, n, r, l in zip(pitch_seq, note_seq, rhythm_seq, lift_seq):
                if r == blank_val:
                    prev_r = -1
                    continue
                elif r == prev_r:
                    continue
                
                batch_pitch_res.append(p.item())
                batch_note_res.append(n.item())
                batch_rhythm_res.append(r.item())
                batch_lift_res.append(l.item())

                prev_r = r

            
            predictions.append({
                'pitch': batch_pitch_res,
                'note': batch_note_res,
                'rhythm': batch_rhythm_res,
                'lift': batch_lift_res
            })

        return predictions