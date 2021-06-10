import math
from collections import defaultdict
from typing import Union, List

import torch
import pytorch_lightning as pl
import torchmetrics


"""
The way metrics are computed in the general case is inspired (copied) from:
https://torchmetrics.readthedocs.io/en/latest/references/modules.html#retrieval
Some code here is more or less copied from their implementation.

When the scores are computed between N queries and N documents (square matrix), 
with positives on the diagonal, we use a vectorized implementation.
"""

def apply_retrieval_metric(preds, targets, indices, metric_fn):
    preds, targets = preds.view(-1), targets.view(-1)
    losses = []
    groups_indices = get_groups_indices(indices)
    for group_indices in groups_indices:
        loss = metric_fn(preds[group_indices], targets[group_indices])
        losses.append(loss)
    losses = torch.stack(losses)
    return losses

def get_groups_indices(indices):
    groups = defaultdict(list)
    indices = indices.tolist()
    for i, index in enumerate(indices):
        groups[index].append(i)
    groups = [
            torch.tensor(group, dtype=torch.long) 
            for group in groups.values()]
    return groups

def positive_sparse2dense(positive_pairs, num_queries: List[int]):
        targets = torch.zeros(*num_queries)
        targets[positive_pairs[:,0], positive_pairs[:,1]] = 1
        targets = targets

        indices_1 = torch.arange(0, num_queries[0]).view(-1, 1)\
                .repeat(1, num_queries[1]).view(-1)
        indices_2 = torch.arange(0, num_queries[1])\
                .repeat(num_queries[0], 1).view(-1)

        return targets, indices_1, indices_2

class CrossmodalRecallAtK(torchmetrics.Metric):
    def __init__(self, k1: int, k2: int, matrix_preds: bool = False):
        super().__init__()
        self.k1 = k1
        self.k2 = k2
        self.matrix_preds = matrix_preds

        self.add_state("recalled_1", 
                default=torch.tensor(0, dtype=torch.float),
                dist_reduce_fx="cat")
        self.add_state("recalled_2", 
                default=torch.tensor(0, dtype=torch.float),
                dist_reduce_fx="cat")

        self.add_state("total_1", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total_2", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds, targets, indices_1, indices_2):
        recalls_1 = recall_at_k(preds, targets, self.k1)
        recalls_2 = recall_at_k(preds.t(), targets.t(), self.k2)

        self.total_1 += recalls_1.size(0)
        self.total_2 += recalls_2.size(0)

        self.recalled_1 += recalls_1.sum()
        self.recalled_2 += recalls_2.sum()

    def compute(self):
        value_1 = self.recalled_1 / self.total_1
        value_2 = self.recalled_2 / self.total_2
        # Harmonic mean
        value = (2 * value_1 * value_2) / (value_1 + value_2 + 1e-6)
        return value

class RecallAtK(torchmetrics.Metric):
    def __init__(self, k: int, matrix_preds: bool = False):
        super().__init__()
        self.k = k
        self.matrix_preds = matrix_preds

        self.add_state("recalled", 
                default=torch.tensor(0, dtype=torch.float),
                dist_reduce_fx="cat")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds, targets = None, indices = None):
        recalls = recall_at_k(preds, targets, self.k)

        self.total += recalls.size(0)
        self.recalled += recalls.sum()

    def compute(self):
        value = self.recalled / self.total
        return value


def naive_recall_at_k(scores, targets, k):
    def row_recall(scores, targets, k):

        ranks = sorted(enumerate(scores), key=lambda t: t[1], reverse=True)

        num_retrieved = 0
        for i, (idx, score) in enumerate(ranks[:k]):
            if targets[idx] == True:
                num_retrieved += 1

        num_targets = min(k, sum(targets))

        return num_retrieved / num_targets

    recalls = []
    for i in range(scores.size(0)):
        recalls.append(row_recall(scores[i], targets[i], k))
    return recalls

# single anchor
def anchor_recall_at_k(preds, targets, k):
    """ Recall@K, normalized by best possible score.  """
    retrieved_indices = preds.argsort(dim=-1, descending=True)

    # .clamp(max=k) makes it possible to get a perfect score 
    # when (k < num_relevants).
    num_relevants = targets.sum().clamp(max=k)
    num_retrieved_relevants = targets[retrieved_indices][:k].sum().float()

    recall_value = num_retrieved_relevants / num_relevants
    return recall_value

def recall_at_k(preds, targets, k):
    topk_indices = preds.topk(k, dim=1, largest=True)[1]
    num_retrieved = targets.gather(1, topk_indices).sum(dim=1).float()

    num_targets = targets.sum(dim=1).clamp(max=k)

    recalls = (num_retrieved / num_targets)

    return recalls

if __name__ == '__main__':
    preds = torch.tensor([
        [0.1, 0.6, 0.8],
        [0.2, 1.0, 0.8],
        [0.1, 0.7, 0.2]])
    targets = torch.tensor([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]], dtype=torch.long).view(-1)
    indices_1 = torch.tensor([
        [0, 0, 0],
        [1, 1, 1],
        [2, 2, 2]], dtype=torch.long).view(-1)
    indices_2 = torch.tensor([
        [0, 1, 2],
        [0, 1, 2],
        [0, 1, 2]], dtype=torch.long).view(-1)

    ratk = CrossmodalRecallAtK(1, 1, True)
    ratk.update(preds, targets, indices_1, indices_2)
    r = ratk.compute()
    print(r)

    r = square_retrieval_at(preds, 2)
    print(r)
