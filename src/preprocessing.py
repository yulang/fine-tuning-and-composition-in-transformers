from config import config
from collections import Counter
import os
from torch.utils import data
import logging
from sklearn.model_selection import train_test_split

import sys
sys.path.append("/path/to/phrasal/composition/src")
from sentiment_exp import construct_sentiment_dic

logger = logging.getLogger(__name__)

# label dic is organized as {src_sent: {trg_sent1: pos, trg_sent2: neg}}
# sentence dic is organized as {src_sent: [(trg_sent1, score1), (trg_sent2, score2)]}
def preprocess_ellipsis(file_path):
    handler = open(file_path, "r")
    sentence_dic = dict()
    # loading
    for line_no, line in enumerate(handler):
        line = line.strip()
        if line_no == 0:
            # skip header
            _ = line
            continue
        _, sent1, sent2, score = line.split("\t")
        score = int(score)

        if sent1 in sentence_dic:
            sentence_dic[sent1].append((sent2, score))
        elif sent2 in sentence_dic:
            # unexpected case??!!
            assert False
        else:
            # create new entry
            sentence_dic[sent1] = [(sent2, score)]

    # refining
    label_dic = dict()
    discard_count = 0
    for src_sent, trg_sent_list in sentence_dic.items():
        candidate_scores = Counter()
        for trg_sent_tuple in trg_sent_list:
            trg_sent, score = trg_sent_tuple
            candidate_scores[trg_sent] += score
        
        if len(candidate_scores) != 2:
            # each source phrase should have two target phrases
            discard_count += 1
            continue

        label_dic[src_sent] = {}
        candidate1, candidate2 = candidate_scores.items()
        # comparing scores
        if candidate1[1] > candidate2[1]:
            label_dic[src_sent][candidate1[0]] = "pos"
            label_dic[src_sent][candidate2[0]] = "neg"
        else: 
            label_dic[src_sent][candidate1[0]] = "neg"
            label_dic[src_sent][candidate2[0]] = "pos"

    print("total count: {}, discarded: {}".format(len(sentence_dic), discard_count))
    return sentence_dic, label_dic


class EvalDataset(data.Dataset):
    def __init__(self, sequence_list):
        self.text_list = sequence_list
        self.size = len(sequence_list)

    def __getitem__(self, index):
        return self.text_list[index]

    def __len__(self):
        return self.size


class SentencePairDataset(data.Dataset):
    def __init__(self):
        self.size = 0
        self.sentence_tuples = []
        self.labels = []
    def read_line(self, line):
        segments = line.strip().split('\t')
        self.labels += int(segments[-1]),
        sent1, sent2 = segments[1][2:-1], segments[2][2:-1]
        self.sentence_tuples.append([sent1, sent2])
        self.size += 1
    def __getitem__(self, index):
        return self.sentence_tuples[index], self.labels[index]
    
    def __len__(self):
        return self.size


def load_paws(data_dir, balanced=False, batch_size=16):
    # output two loaders
    # itertating over either one will give sentence_batch, label_batch
    # sentence_batch: (2, batch_size)
    # label_batch: (batch_size)
    if balanced:
        train_path = os.path.join(data_dir, "balanced_train.tsv")
        dev_test_path = os.path.join(data_dir, "balanced_dev_test.tsv")
    else:
        train_path = os.path.join(data_dir, "train.tsv")
        dev_test_path = os.path.join(data_dir, "dev_and_test.tsv")
    
    train_set = SentencePairDataset()
    dev_test_set = SentencePairDataset()
    
    with open(train_path) as train_f:
        for line_no, line in enumerate(train_f):
            if line_no == 0:
                continue
            train_set.read_line(line)
            
    with open(dev_test_path) as dev_f:
        for line_no, line in enumerate(dev_f):
            if line_no == 0:
                continue
            dev_test_set.read_line(line)

    # turn off dev-test splitting for now
    # directly optimized model on full dev-test set

    # total_len = len(dev_test_set)
    # dev_len = int(total_len * 0.4)
    # dev_set, test_set = data.random_split(dev_test_set, [dev_len, total_len - dev_len])

    train_loader = data.DataLoader(dataset=train_set, shuffle=True, batch_size=batch_size)
    # dev_loader = data.DataLoader(dataset=dev_set, shuffle=False, batch_size=batch_size)
    test_loader = data.DataLoader(dataset=dev_test_set, shuffle=True, batch_size=batch_size)
    
    return train_loader, test_loader


class SentenceDataset(data.Dataset):
    def __init__(self, phrase_len, phrases, labels):
        self.size = 0
        self.sentences = phrases
        self.labels = labels
        self.target_len = phrase_len
        self.size = len(phrases)
    # def parse_dict(self, phrase2label):
    #     # filter prhase based on target len specified by the config
    #     for phrase, label in phrase2label.items():
    #         self.sentences.append(phrase)
    #         self.labels.append(label)

    #     self.size = len(self.sentences)
    def filter_phrase(self):
        if self.target_len is None:
            return
        else:
            raw_sentences = self.sentences
            raw_labels = self.labels
            self.sentences = []
            self.labels = []
            for phrase, label in zip(raw_sentences, raw_labels):
                word_count = len(phrase.split())
                if word_count == self.target_len:
                    self.sentences.append(phrase)
                    self.labels.append(label)
            self.size = len(self.sentences)

    def __getitem__(self, index):
        return self.sentences[index], self.labels[index]
    def __len__(self):
        return self.size


def load_stanford(treebank_folder, phrase_len, reserve_test, batch_size=16):
    # mapping string label to int label
    label_mapping = {
        "very neg": 0,
        "neg": 1,
        "neutral": 2,
        "pos": 3,
        "very pos": 4,
    }
    id2phrase, id2sent_score, id2sent_label, phrase2id = construct_sentiment_dic(treebank_folder)

    phrase2label = {}
    for phrase_id, phrase in id2phrase.items():
        if phrase in phrase2label:
            logger.debug("duplicate phrase {}".format(phrase))
        phrase2label[phrase] = label_mapping[id2sent_label[phrase_id]]

    # full_dataset = SentenceDataset(phrase_len)
    # full_dataset.parse_dict(phrase2label)
    # total_len = len(full_dataset)
    full_phrases = list(phrase2label.keys())
    full_labels = list(phrase2label.values())
    total_len = len(full_phrases)

    # updated logic: splitting dataset before filtering based on phrase length
    if reserve_test:
        # reserve part of dataset for evaluation with mixed phrase lengths.
        # train_len = int(total_len * 0.7)
        # dev_len = int(total_len * 0.15)
        # test_len = total_len - train_len - dev_len
        # train_set, dev_set, test_set = data.random_split(full_dataset, [train_len, dev_len, test_len])
        phrase_train, phrase_val_test, label_train, label_val_test = train_test_split(full_phrases, full_labels, test_size=0.3)
        phrase_dev, phrase_test, label_dev, label_test = train_test_split(phrase_val_test, label_val_test, test_size=0.5)

        train_set = SentenceDataset(phrase_len, phrase_train, label_train)
        dev_set = SentenceDataset(phrase_len, phrase_dev, label_dev)
        test_set = SentenceDataset(phrase_len, phrase_test, label_test)

        train_set.filter_phrase()
        dev_set.filter_phrase()

        train_loader = data.DataLoader(dataset=train_set, shuffle=True, batch_size=batch_size)
        dev_loader = data.DataLoader(dataset=dev_set, shuffle=True, batch_size=batch_size)
        test_loader = data.DataLoader(dataset=test_set, shuffle=True, batch_size=batch_size)
    else:
        # train_len = int(total_len * 0.8)
        # train_set, dev_set = data.random_split(full_dataset, [train_len, total_len - train_len])
        phrase_train, phrase_dev, label_train, label_dev = train_test_split(full_phrases, full_labels, test_size=0.3)
        
        train_set = SentenceDataset(phrase_len, phrase_train, label_train)
        dev_set = SentenceDataset(phrase_len, phrase_dev, label_dev)

        train_set.filter_phrase()
        dev_set.filter_phrase()

        train_loader = data.DataLoader(dataset=train_set, shuffle=True, batch_size=batch_size)
        dev_loader = data.DataLoader(dataset=dev_set, shuffle=True, batch_size=batch_size)
        test_loader = None
    
    return train_loader, dev_loader, test_loader

