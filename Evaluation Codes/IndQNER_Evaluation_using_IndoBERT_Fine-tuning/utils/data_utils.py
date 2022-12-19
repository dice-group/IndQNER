import numpy as np
import pandas as pd
import string
import torch
import re
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

#####
# Term Extraction Airy
#####
class AspectExtractionDataset(Dataset):
    # Static constant variable
    LABEL2INDEX = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-LOC': 3, 'I-LOC': 4, 'B-ORG': 5, 'I-ORG': 6}
    INDEX2LABEL = {0: 'O', 1: 'B-PER', 2: 'I-PER', 3: 'B-LOC', 4: 'I-LOC', 5: 'B-ORG', 6: 'I-ORG'}
    NUM_LABELS = 7
    
    def load_dataset(self, path):
        # Read file
        data = open(path,'r').readlines()

        # Prepare buffer
        dataset = []
        sentence = []
        seq_label = []
        for line in data:
            if '\t' in line:
                token, label = line[:-1].split('\t')
                sentence.append(token)
                seq_label.append(self.LABEL2INDEX[label])
            else:
                dataset.append({
                    'sentence': sentence,
                    'seq_label': seq_label
                })
                sentence = []
                seq_label = []
        return dataset
    
    def __init__(self, dataset_path, tokenizer, *args, **kwargs):
        self.data = self.load_dataset(dataset_path)
        self.tokenizer = tokenizer
        
    def __getitem__(self, index):
        data = self.data[index]
        sentence, seq_label = data['sentence'], data['seq_label']
        
        # Add CLS token
        subwords = [self.tokenizer.cls_token_id]
        subword_to_word_indices = [-1] # For CLS
        
        # Add subwords
        for word_idx, word in enumerate(sentence):
            subword_list = self.tokenizer.encode(word, add_special_tokens=False)
            subword_to_word_indices += [word_idx for i in range(len(subword_list))]
            subwords += subword_list
            
        # Add last SEP token
        subwords += [self.tokenizer.sep_token_id]
        subword_to_word_indices += [-1]
        
        return np.array(subwords), np.array(subword_to_word_indices), np.array(seq_label), data['sentence']
    
    def __len__(self):
        return len(self.data)
       
class AspectExtractionDataLoader(DataLoader):
    def __init__(self, max_seq_len=512, *args, **kwargs):
        super(AspectExtractionDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = self._collate_fn
        self.max_seq_len = max_seq_len
        
    def _collate_fn(self, batch):
        batch_size = len(batch)
        max_seq_len = max(map(lambda x: len(x[0]), batch))
        max_seq_len = min(self.max_seq_len, max_seq_len)
        max_tgt_len = max(map(lambda x: len(x[2]), batch))
        
        subword_batch = np.zeros((batch_size, max_seq_len), dtype=np.int64)
        mask_batch = np.zeros((batch_size, max_seq_len), dtype=np.float32)
        subword_to_word_indices_batch = np.full((batch_size, max_seq_len), -1, dtype=np.int64)
        seq_label_batch = np.full((batch_size, max_tgt_len), -100, dtype=np.int64)
        
        seq_list = []
        for i, (subwords, subword_to_word_indices, seq_label, raw_seq) in enumerate(batch):
            subwords = subwords[:max_seq_len]
            subword_to_word_indices = subword_to_word_indices[:max_seq_len]
            
            subword_batch[i,:len(subwords)] = subwords
            mask_batch[i,:len(subwords)] = 1
            subword_to_word_indices_batch[i,:len(subwords)] = subword_to_word_indices
            seq_label_batch[i,:len(seq_label)] = seq_label

            seq_list.append(raw_seq)
            
        return subword_batch, mask_batch, subword_to_word_indices_batch, seq_label_batch, seq_list
    
####
# Indonesian Quran NER Dataset
####
class IndQurNerDataset(Dataset):
    # Static constant variable
    LABEL2INDEX = {'O': 0, 'B-Allah': 1, 'I-Allah': 2, 'B-Throne': 3, 'I-Throne': 4, 'B-Artifact': 5, 'I-Artifact': 6, 'B-AstronomicalBody': 7, 'I-AstronomicalBody': 8, 'B-Event': 9, 'I-Event': 10, 'B-HolyBook': 11, 'I-HolyBook': 12, 'B-Language': 13, 'I-Language': 14, 'B-Angel': 15, 'I-Angel': 16, 'B-Person': 17, 'I-Person': 18, 'B-Messenger': 19, 'I-Messenger': 20, 'B-Prophet': 21, 'I-Prophet': 22, 'B-AfterlifeLocation': 23, 'I-AfterlifeLocation': 24, 'B-GeographicalLocation': 25, 'I-GeographicalLocation': 26, 'B-Color': 27, 'I-Color': 28, 'B-Religion': 29, 'I-Religion': 30, 'B-Food': 31, 'I-Food': 32, 'B-Fruit': 33, 'I-Fruit': 34, 'B-TheBookOfAllah': 35, 'I-TheBookOfAllah': 36, 'B-Plant': 37, 'I-Plant': 38}
    INDEX2LABEL = {0: 'O', 1: 'B-Allah', 2: 'I-Allah', 3: 'B-Throne', 4: 'I-Throne', 5: 'B-Artifact', 6: 'I-Artifact', 7: 'B-AstronomicalBody', 8: 'I-AstronomicalBody', 9: 'B-Event', 10: 'I-Event', 11: 'B-HolyBook', 12: 'I-HolyBook', 13: 'B-Language', 14: 'I-Language', 15: 'B-Angel', 16: 'I-Angel', 17: 'B-Person', 18: 'I-Person', 19: 'B-Messenger', 20: 'I-Messenger', 21: 'B-Prophet', 22: 'I-Prophet', 23: 'B-AfterlifeLocation', 24: 'I-AfterlifeLocation', 25: 'B-GeographicalLocation', 26: 'I-GeographicalLocation', 27: 'B-Color', 28: 'I-Color', 29: 'B-Religion', 30: 'I-Religion', 31: 'B-Food', 32: 'I-Food', 33: 'B-Fruit', 34: 'I-Fruit', 35: 'B-TheBookOfAllah', 36: 'I-TheBookOfAllah', 37: 'B-Plant', 38: 'I-Plant'}
    NUM_LABELS = 39
    
    def load_dataset(self, path):
        # Read file
        data = open(path,'r').readlines()

        # Prepare buffer
        dataset = []
        sentence = []
        seq_label = []
        for line in data:
            if len(line.strip()) > 0:
                token, label = line[:-1].split('\t')
                sentence.append(token)
                seq_label.append(self.LABEL2INDEX[label])
            else:
                dataset.append({
                    'sentence': sentence,
                    'seq_label': seq_label
                })
                sentence = []
                seq_label = []
        return dataset
    
    def __init__(self, dataset_path, tokenizer, *args, **kwargs):
        self.data = self.load_dataset(dataset_path)
        self.tokenizer = tokenizer
        
    def __getitem__(self, index):
        data = self.data[index]
        sentence, seq_label = data['sentence'], data['seq_label']
        
        # Add CLS token
        subwords = [self.tokenizer.cls_token_id]
        subword_to_word_indices = [-1] # For CLS
        
        # Add subwords
        for word_idx, word in enumerate(sentence):
            subword_list = self.tokenizer.encode(word, add_special_tokens=False)
            subword_to_word_indices += [word_idx for i in range(len(subword_list))]
            subwords += subword_list
            
        # Add last SEP token
        subwords += [self.tokenizer.sep_token_id]
        subword_to_word_indices += [-1]
        
        return np.array(subwords), np.array(subword_to_word_indices), np.array(seq_label), data['sentence']
    
    def __len__(self):
        return len(self.data)

class NerDataLoader(DataLoader):
    def __init__(self, max_seq_len=512, *args, **kwargs):
        super(NerDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = self._collate_fn
        self.max_seq_len = max_seq_len
        
    def _collate_fn(self, batch):
        batch_size = len(batch)
        max_seq_len = max(map(lambda x: len(x[0]), batch))
        max_seq_len = min(self.max_seq_len, max_seq_len)
        max_tgt_len = max(map(lambda x: len(x[2]), batch))
        
        subword_batch = np.zeros((batch_size, max_seq_len), dtype=np.int64)
        mask_batch = np.zeros((batch_size, max_seq_len), dtype=np.float32)
        subword_to_word_indices_batch = np.full((batch_size, max_seq_len), -1, dtype=np.int64)
        seq_label_batch = np.full((batch_size, max_tgt_len), -100, dtype=np.int64)
        
        seq_list = []
        for i, (subwords, subword_to_word_indices, seq_label, raw_seq) in enumerate(batch):
            subwords = subwords[:max_seq_len]
            subword_to_word_indices = subword_to_word_indices[:max_seq_len]

            subword_batch[i,:len(subwords)] = subwords
            mask_batch[i,:len(subwords)] = 1
            subword_to_word_indices_batch[i,:len(subwords)] = subword_to_word_indices
            seq_label_batch[i,:len(seq_label)] = seq_label

            seq_list.append(raw_seq)
            
        return subword_batch, mask_batch, subword_to_word_indices_batch, seq_label_batch, seq_list

