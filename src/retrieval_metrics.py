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
        targets = targets.view(-1)

        indices_1 = torch.arange(0, num_queries[0]).view(-1, 1)\
                .repeat(1, num_queries[1]).view(-1)
        indices_2 = torch.arange(0, num_queries[1])\
                .repeat(num_queries[0], 1).view(-1)

        return targets, indices_1, indices_2

class HardestTripletMarginLoss(torch.nn.Module):
    def __init__(self, margin: float, hardest_fraction: float = 0.0,
            single_positive: bool = False): 
        """
            Computes a triplet margin ranking loss on similarity scores.

            hardest_fraction: the fraction of the triplets 
            sorted by decreasing difficulty to consider. 

            single_positive: whether there is only one positive for each
            query (uses a different implementation if so)
        """
        super().__init__()
        
        self.margin = margin
        self.hardest_fraction = hardest_fraction
        self.single_positive = single_positive

    def forward(self, preds, targets, indices_1, indices_2):
        if self.single_positive:
            return self._square_forward(preds)
        else:
            return self._sparse_forward(
                    preds.view(-1), targets, indices_1, indices_2)

    def _sparse_forward(self, preds, targets, indices_1, indices_2):
        """
            preds, targets, indices: (N,) tensor
        """
        loss_1 = apply_retrieval_metric(preds, targets, indices_1,
                self._query_loss).mean()
        loss_2 = apply_retrieval_metric(preds, targets, indices_2, 
                self._query_loss).mean()

        loss = (loss_1 + loss_2) / (2 * self.margin)
        return loss

    def _query_loss(self, preds, targets):
        return hardest_triplet_margin_loss(preds, targets, 
                self.margin, self.hardest_fraction)

    def _square_forward(self, preds):
        """
            preds: (N,N) tensor
        """
        if preds.size(0) != preds.size(1):
            raise ValueError(
                    "_square_forward only accepts square score matrices."
                    f" Received ({preds.size(0)}, {preds.size(1)})")

        positive_scores = preds.diagonal().view(preds.size(0), 1)
        positive_scores1 = positive_scores.expand_as(preds)
        positive_scores2 = positive_scores.t().expand_as(preds)

        loss1 = (self.margin + preds - positive_scores1).clamp(min=0)
        loss2 = (self.margin + preds - positive_scores2).clamp(min=0)

        loss1 = loss1.fill_diagonal_(0)
        loss2 = loss2.fill_diagonal_(0)

        k1 = max(1, int(self.hardest_fraction * loss1.size(1)))
        k2 = max(1, int(self.hardest_fraction * loss1.size(0)))

        loss1 = torch.mean(torch.topk(loss1, k1, 1)[0], 1)
        loss2 = torch.mean(torch.topk(loss2, k2, 0)[0], 0)

        loss = (torch.mean(loss1) + torch.mean(loss2)) 
        loss /= (2 * self.margin)
        return loss

def hardest_triplet_margin_loss(preds, targets, margin, hardest_fraction):
    """
        preds is a vector of N scores.
        targets is a vector of N booleans, with True/False meaning the
        corresponding element in preds is a positive/negative.
    """
    targets = targets >= 0.5
    positives = preds[targets]
    negatives = preds[~targets]

    hardest_positives = top_fraction(positives, hardest_fraction, False)\
            .unsqueeze(1)
    hardest_negatives = top_fraction(negatives, hardest_fraction, True)\
            .unsqueeze(0).expand(hardest_positives.numel(), -1) 

    losses = (margin + hardest_negatives - hardest_positives).clamp(min=0)
    loss = losses.mean()
    return loss

def top_fraction(x, fraction, largest=True):
    k = max(1, math.ceil(fraction * x.numel()))
    out = x.topk(k, largest=largest)[0]
    return out

def square_hardest_triplet_margin_loss(preds, margin, hardest_fraction):
    positives = preds.diagonal().unsqueeze(1)
    
    #diag_mask = torch.eye(preds.size(0)) > .5
    #preds = preds.masked_fill(diag_mask, float('-inf'))

    #k = max(1, int(hardest_fraction * (preds.size(1) - 1)))
    #hardest_negatives = preds.topk(k, 1)[0]

    #losses = (margin + hardest_negatives - positives).clamp(min=0)
    #loss = losses.mean()
    #return loss

    losses = (margin + preds - positives).clamp(min=0)
    losses = losses.fill_diagonal_(0)
    k = max(1, int(hardest_fraction * (losses.size(1) - 1)))
    loss = losses.topk(k, 1)[0].mean()
    return loss

def vectorized_html(preds, margin, hardest_fraction):
    num_queries

    num_positives
    num_negatives

    max_positives = num_positives.max()

    positives = torch.zeros(num_queries, max_positives)
    negatives = torch.zeros(num_queries, max_negatives)

    positives_k = max(1, int(hardest_fraction * max_positives))
    hardest_positives = positives.topk(positives_k, 1)[0]

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
        if self.matrix_preds:
            recalls_1 = matrix_retrieval_at(preds, self.k1)
            recalls_2 = matrix_retrieval_at(preds.t(), self.k2)
        else:
            recalls_1 = apply_retrieval_metric(preds.view(-1), targets, 
                    indices_1, self._recall_at_k1)
            recalls_2 = apply_retrieval_metric(preds.view(-1), targets, 
                    indices_2, self._recall_at_k2)

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

    def _recall_at_k1(self, preds, targets):
        return recall_at_k(preds, targets, self.k1)
    def _recall_at_k2(self, preds, targets):
        return recall_at_k(preds, targets, self.k2)

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
        if self.matrix_preds:
            recalls = matrix_retrieval_at(preds, self.k, targets)
        else:
            recalls = apply_retrieval_metric(preds.view(-1), targets, 
                    indices, self._recall_at_k)

        self.total += recalls.size(0)
        self.recalled += recalls.sum()

    def compute(self):
        value = self.recalled / self.total
        return value

    def _recall_at_k(self, preds, targets):
        return recall_at_k(preds, targets, self.k)

def recall_at_k(preds, targets, k):
    """ Recall@K, normalized by best possible score.  """
    retrieved_indices = preds.argsort(dim=-1, descending=True)

    # .clamp(max=k) makes it possible to get a perfect score 
    # when (k < num_relevants).
    num_relevants = targets.sum().clamp(max=k)
    num_retrieved_relevants = targets[retrieved_indices][:k].sum().float()

    recall_value = num_retrieved_relevants / num_relevants
    return recall_value

def matrix_retrieval_at(scores: torch.Tensor, k: torch.Tensor, 
        positive_indices: torch.Tensor = None)\
        -> torch.Tensor:
    """
        Retrieval@k measures the average ability of returning the 
        correct document in the top-k results.

        scores: (N, *) of the scores of N documents with all the others 

        positive_indices: (N) indices of the correct documents for the N queries.
        If None, assumes the matrix is square with pairs of corresponding
        elements having the same index on a different axis.
    """
    sorted_indices = torch.argsort(scores, dim=1, descending=True)
    if positive_indices is None:
        positive_indices = torch.arange(scores.size(0),
                device=sorted_indices.device)
    _, positive_ranks = torch.nonzero(
            sorted_indices == positive_indices.view(-1, 1),
            as_tuple=True)

    retrieved_at_k = (positive_ranks < k)
    return retrieved_at_k.squeeze().float()

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
