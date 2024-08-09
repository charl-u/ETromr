import torch
import re
import os
import cv2
import torch.utils.data as data
from torchvision import transforms
from utils.CommonCTCUtils import CommonCTCUtil
# from utils.SemanticTokenizer import ctc_label_tokenizer

class CommonCTCDataSet(data.Dataset):
    def __init__(self, 
                 data_root,
                 img_suffix,
                 label_suffix,
                 max_len=64,
                 train=True,
            ):
        super(CommonCTCDataSet, self).__init__()
        self.root = data_root
        self.train = train
        self.img_suffix = img_suffix
        self.label_suffix = label_suffix
        self.max_len = max_len

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128, 1024)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.7931], std=[0.1738]),         
        ])
        self.init_paths()
        self.load_records()
        self.ctc_utils = CommonCTCUtil(vocab_path=self.vocab_path)

    def init_paths(self):
        self.type = 'train' if self.train else 'test'
        self.indexes_path = os.path.join(self.root, self.type + '.txt')
        self.data_path = os.path.join(self.root, 'Corpus')
        self.vocab_path = os.path.join(self.root, 'vocab.txt')

    def load_records(self):
        with open(self.indexes_path, 'r') as f:
            self.dirs = f.readlines()
        self.dirs = [dir.replace('\n', '') for dir in self.dirs]

    def __len__(self):
        return len(self.dirs)
    
    def __getitem__(self, index):
        assert index < len(self), 'index range error'

        record = self.dirs[index]
        record_path = os.path.join(self.data_path, record)
        
        image_path = os.path.join(record_path, record + self.img_suffix)
        label_path = os.path.join(record_path, record + self.label_suffix)

        try:
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        except:
            print('[Dataset] Image not found: {}'.format(image_path))
            return self[index + 1]
        
        if self.transform is not None:
            image = self.transform(image)

        with open(label_path, 'r') as f:
            symbols_label = f.readline()
            symbols_label = re.split(r'\s+', symbols_label)
            symbols_label = [symbol for symbol in symbols_label if symbol != '\n']
        symbols_label = list(filter(None, symbols_label))
        
        gt_tokens = self.ctc_utils.tokenize(symbols_label, self.max_len)
        gt_labels = self.ctc_utils.detokenize(gt_tokens)
        info_dict = {
            "name": record,
            "image_path": image_path,
            "label_path": label_path,
            "label_length": len(symbols_label),
            'gt_tokens': gt_tokens,
            'gt_labels': gt_labels,
            "image": image
        }

        return info_dict
