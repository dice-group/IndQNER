
import logging
import os
import sys
import torch
import pickle
import numpy as np

from torch.utils.data import TensorDataset
from tqdm import tqdm

logger = logging.getLogger(__name__)

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text, label=None):
        self.guid = guid
        self.text = text 
        self.label = label

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, ori_tokens, ori_labels):

        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.ori_tokens = ori_tokens
        self.ori_labels = ori_labels


class NerProcessor(object):
    def read_data(self, input_file):
        """Reads a BIO data."""
        with open(input_file, "r", encoding="utf-8") as f:
            lines = []
            words = []
            labels = []
            
            for line in f.readlines():   
                content = line.strip()
                tokens = line.strip().split("\t")

                if len(tokens) == 2:
                    words.append(tokens[0])
                    labels.append(tokens[1])
                else:
                    if len(content) == 0 and len(words) > 0:
                        label = []
                        word = []
                        for l, w in zip(labels, words):
                            if len(l) > 0 and len(w) > 0:
                                label.append(l)
                                word.append(w)
                        lines.append([' '.join(label), ' '.join(word)])
                        words = []
                        labels = []
            
            return lines
    
    def get_labels(self, args):
        labels = set()
        if os.path.exists(os.path.join(args.output_dir, "label_list.pkl")):
            logger.info(f"loading labels info from {args.output_dir}")
            with open(os.path.join(args.output_dir, "label_list.pkl"), "rb") as f:
                labels = pickle.load(f)
        else:
            # get labels from train data
            logger.info(f"loading labels info from train file and dump in {args.output_dir}")
            with open(args.train_file) as f:
                for line in f.readlines():
                    tokens = line.strip().split("\t")

                    if len(tokens) == 2:
                        labels.add(tokens[1])

            if len(labels) > 0:
                with open(os.path.join(args.output_dir, "label_list.pkl"), "wb") as f:
                    pickle.dump(labels, f)
            else:
                logger.info("loading error and return the default labels B,I,O")
                labels = {"O", "B", "I"}
        
        return labels 

    def get_examples(self, input_file):
        examples = []
        
        lines = self.read_data(input_file)

        for i, line in enumerate(lines):
            guid = str(i)
            text = line[1]
            label = line[0]

            examples.append(InputExample(guid=guid, text=text, label=label))
        
        return examples


def convert_examples_to_features(args, examples, label_list, max_seq_length, tokenizer):

    label_map = {label : i for i, label in enumerate(label_list)}
    
    features = []

    for (ex_index, example) in tqdm(enumerate(examples), desc="convert examples"):
        # if ex_index % 10000 == 0:
        #     logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        
        textlist = example.text.split(" ")
        labellist = example.label.split(" ")
        assert len(textlist) == len(labellist)
        tokens = []
        labels = []
        ori_tokens = []
        input_ids = []
        for i, word in enumerate(textlist):
            # Prevent the wordPiece situation from appearing, but it doesn’t seem to be
            token = tokenizer.tokenize(word)
            ids = tokenizer.convert_tokens_to_ids(token)
            ids = np.average(ids)
            input_ids.append(ids)
            tokens.append(token)
            label_1 = labellist[i]
            ori_tokens.append(word)
            labels.append(label_1)
            
        if len(tokens) >= max_seq_length - 1:
            tokens = tokens[0:(max_seq_length - 2)]  # The reason for -2 is because the sequence needs to add a sentence beginning and ending mark
            labels = labels[0:(max_seq_length - 2)]
            ori_tokens = ori_tokens[0:(max_seq_length - 2)]
            input_ids = input_ids[0:(max_seq_length - 2)]

        ori_tokens = ["[CLS]"] + ori_tokens + ["[SEP]"]
        labels = ["O"] + labels + ["O"]
        ntokens = []
        segment_ids = []
        label_ids = []
        ntokens.append("[CLS]")
        input_ids.append(tokenizer.convert_tokens_to_ids("[CLS]"))
        segment_ids.append(0)
        label_ids.append(label_map["O"])
        for i, token in enumerate(tokens):
            ntokens.append(token)
            segment_ids.append(0)
            label_ids.append(label_map[labels[i+1]])
            
        ntokens.append("[SEP]")
        input_ids.append(tokenizer.convert_tokens_to_ids("[SEP]"))
        segment_ids.append(0)
        label_ids.append(label_map["O"])
        input_mask = [1] * len(input_ids)
        assert len(ori_tokens) == len(ntokens), f"{len(ori_tokens)}, {len(ntokens)}, {ori_tokens}"
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            # we don't concerned about it!
            label_ids.append(0)
            ntokens.append("**NULL**")
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in ntokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))

        # if not os.path.exists(os.path.join(output_dir, 'label2id.pkl')):
        #     with open(os.path.join(output_dir, 'label2id.pkl'), 'wb') as w:
        #         pickle.dump(label_map, w)

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_ids,
                              ori_tokens=ori_tokens,
                              ori_labels=labels 
                              ))

    return features


def get_Dataset(args, processor, tokenizer, mode="train"):
    if mode == "train":
        filepath = args.train_file
    elif mode == "eval":
        filepath = args.eval_file
    elif mode == "test":
        filepath = args.test_file
    else:
        raise ValueError("mode must be one of train, eval, or test")

    examples = processor.get_examples(filepath)
    label_list = args.label_list

    features = convert_examples_to_features(
        args, examples, label_list, args.max_seq_length, tokenizer
    )

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)

    data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

    return examples, features, data

