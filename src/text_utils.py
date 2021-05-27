import argparse
from collections import defaultdict, Counter
import pickle
from typing import List

import nltk
import torch

padding_token = '<pad>'
unknown_token = '<unk>'
start_token = '<start>'
end_token = '<end>'

special_tokens = {
    padding_token: 0, # padding_token MUST be 0
    unknown_token: 1,
    start_token: 2,
    end_token: 3
}

def tokenize(sentence: str) -> List[str]:
    return nltk.word_tokenize(sentence)

class Vocabulary(object):
    def __init__(self, 
            unknown_token_idx: int = special_tokens[unknown_token]):
        self.unknown_token_idx = unknown_token_idx
        self.token2idx = {}
        self.idx2token = {}
        self.idx = 0
        pass

    def add_token(self, token: int) -> None:
        self.token2idx[token] = self.idx
        self.idx2token[self.idx] = token
        self.idx += 1

    def doc2idx(self, doc: List[str]) -> List[int]:
        return [self[token] 
            if token in self.token2idx else self.unknown_token_idx 
            for token in doc]

    def __call__(self, doc: List[str]) -> List[int]:
        return self.doc2idx(doc)

    def __getitem__(self, token: str) -> int:
        return self.token2idx[token]

    def __len__(self) -> int:
        return self.idx

class NltkTokenization(object):
    def __call__(self, sentence: str) -> List[str]:
        return nltk.word_tokenize(sentence)

class Sentence2Tensor(torch.nn.Module):
    def __init__(self, vocabulary):
        super().__init__()
        self.vocabulary = vocabulary

    def forward(self, sentence: str) -> torch.Tensor:
        tokens = tokenize(sentence)
        indices = self.vocabulary(tokens)
        return torch.tensor(indices, dtype=torch.long)

def fit_vocabulary(corpus, min_count=0, max_frequency=1):
    counter = Counter()
    for doc in corpus:
        counter.update(doc)
    total_count = sum(counter.values())

    vocabulary = Vocabulary()

    sorted_special_tokens = sorted(special_tokens.items(),
            key=lambda x: x[1])
    for token, idx in sorted_special_tokens:
        vocabulary.add_token(token)

    for token, count in counter.items():
        if count < min_count:
            continue
        if (count / total_count) > max_frequency:
            continue
        vocabulary.add_token(token)

    return vocabulary

