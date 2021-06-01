import sys
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

def TripletHingeSquare_forward(scores, margin, hardest_fraction):
    positive_scores = scores.diagonal().view(scores.size(0), 1)
    positive_scores1 = positive_scores.expand_as(scores)
    positive_scores2 = positive_scores.t().expand_as(scores)

    loss1 = (margin + scores - positive_scores1).clamp(min=0)
    loss2 = (margin + scores - positive_scores2).clamp(min=0)

    loss1 = loss1.fill_diagonal_(0)
    loss2 = loss2.fill_diagonal_(0)

    k1 = max(1, int(hardest_fraction * loss1.size(1)))
    k2 = max(1, int(hardest_fraction * loss1.size(0)))

    loss1 = torch.sum(torch.topk(loss1, k1, 1)[0], 1) / min(k1, loss1.size(1)-1)
    loss2 = torch.sum(torch.topk(loss2, k2, 0)[0], 0) / min(k2, loss2.size(0)-1)

    loss = (torch.mean(loss1) + torch.mean(loss2)) 
    loss /= (2 * margin)
    return loss

from collections import defaultdict
def get_groups_indices(indices):
    """
        From torchmetrics.
    """
    groups = defaultdict(list)
    for i, index in enumerate(indices):
        groups[index.item()].append(i)
    groups = [torch.tensor(group, dtype=torch.long) 
            for group in groups.values()]
    return groups

def retrieval_loss(preds, targets, indices, query_loss):
    losses = []
    groups_indices = get_groups_indices(indices)
    for group_indices in groups_indices:
        losses.append(query_loss(preds[group_indices], targets[group_indices]))
    return torch.tensor(losses)

def top_hinge_loss(preds, targets, margin, hardest_fraction):
    def hardest(x, f, largest):
        k = max(1, math.ceil(f * x.numel()))
        out = x.topk(k, largest=largest)[0]
        return out

    positives = preds[targets >= 0.5]
    negatives = preds[targets < 0.5]

    hardest_positives = hardest(positives, hardest_fraction, False)\
            .unsqueeze(1)
    hardest_negatives = hardest(negatives, hardest_fraction, True)\
            .unsqueeze(0).expand(hardest_positives.numel(), -1) 

    losses = (margin + hardest_negatives - hardest_positives).clamp(min=0)
    loss = losses.mean()

    return loss


def TripletHinge_forward(scores, positive_indices, margin, hardest_fraction):
    def query_loss(preds, targets):
        return top_hinge_loss(preds, targets, margin, hardest_fraction)

    preds = scores.view(-1)

    targets = torch.zeros(scores.size())
    targets[positive_indices[:,0], positive_indices[:,1]] = 1
    targets = targets.view(-1)

    indices_1 = torch.arange(0, scores.size(0)).view(-1, 1)\
            .repeat(1, scores.size(1))
    indices_2 = torch.arange(0, scores.size(1))\
            .repeat(scores.size(0), 1)

    loss_1 = retrieval_loss(preds, targets, 
            indices_1.view(-1), 
            query_loss)
    loss_2 = retrieval_loss(preds, targets, 
            indices_2.view(-1), 
            query_loss)

    loss = (torch.mean(loss_1) + torch.mean(loss_2)) / (2 * margin)
    return loss


def test_TripletHinge_forward():
    #torch.manual_seed(7)
    numel_1 = 5
    numel_2 = 5
    scores_shape = (numel_1, numel_2)


    positive_indices = torch.tensor([
            [0, 0],
            [1, 1],
            [2, 2],
            [3, 3],
            [4, 4],
            #[0, 1],
            #[1, 2],
            #[2, 1],
        ], dtype=torch.long)
    scores = torch.distributions.uniform.Uniform(-1, 1)\
            .sample(scores_shape)
    #scores = torch.tensor([
    #    [-0.1,  0.2, -0.6, -0.5, 0.7],
    #    [ 0.7, -0.5, -0.5,  0.5, ],
    #    [-0.2, -0.3,  0.3, -0.0],
    #    [-0.9,  0.4, -0.1,  0.4]])

    print(scores)
    margin = 0.2
    hardest_fraction = 0.5

    indices_loss = TripletHinge_forward(scores, positive_indices,
            margin, hardest_fraction)

    square_loss = TripletHingeSquare_forward(scores, margin, hardest_fraction)

    print(square_loss - indices_loss)

def test_rnn_output():
    torch.manual_seed(0)

    batch_size = 5

    x = torch.Tensor([
        [
            [-1, -1],
            [0, 0],
            [100, 100],
        ]
    ]).repeat(batch_size, 1, 1)

    lengths = torch.LongTensor([3]*batch_size)


    num_layers = 2
    hidden_size = 4
    bidirectional = True

    gru = nn.GRU(2, hidden_size, num_layers, bias=False, bidirectional=bidirectional)

    x = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), 
            batch_first=True, enforce_sorted=True)

    x, h = gru(x)

    print("x:", x, x.data.size())
    print("h:", h, h.size())

    h = h.view(num_layers, 1+bidirectional, batch_size, hidden_size)[-1]\
        .transpose(0, 1)\
        .reshape(batch_size, -1)

    print(h, h.size())

    # Extract the output at the last timestep of each sequence
    x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True) 
    indices = (lengths - 1).view(-1, 1, 1)
    indices = indices.expand(x.size(0), 1, x.size(-1))
    x = torch.gather(x, 1, indices).squeeze(1)

def test_batch_retrieval():
    import torchmetrics
    import evaluation
    import retrieval_metrics

    metrics = torchmetrics.MetricCollection({
        'R@1': retrieval_metrics.RecallAtK(1, matrix_preds=False)
    })

    num_sources = 1024
    embedding_size = 128
    batch_size = 64

    sources = torch.rand((num_sources, embedding_size))
    targets = sources.clone()

    num_non_matching = num_sources // 6
    targets[:num_non_matching] = torch.rand((num_non_matching, embedding_size))

    sources = F.normalize(sources, p=2, dim=1)
    targets = F.normalize(targets, p=2, dim=1)

    score_fn = lambda a,b: torch.matmul(a, b.t())

    batches = []
    current_offset = 0
    while current_offset < num_sources:
        batch_end_index = min(num_sources, current_offset+batch_size)
        current_batch_size = batch_end_index - current_offset

        batch_positive_pairs = torch.arange(current_batch_size).view(-1, 1).repeat(1, 2)
        #batch_positive_pairs = torch.arange(current_batch_size).view(1, -1).repeat(2, 1)
        batch = {
            'sources': sources[current_offset:batch_end_index],
            'targets': targets[current_offset:batch_end_index],
            'positive_pairs': batch_positive_pairs
        }
        print(batch['sources'].size(), batch['targets'].size(), batch['positive_pairs'].size())
        batches.append(batch)

        current_offset += batch_size

    evaluation.batch_retrieval(
        batches, 
        source_key='sources', target_key='targets', 
        swap_positive_pairs=False,
        score_fn=score_fn, metrics=metrics)

    metric_values = {
        metric_name: value.item()
        for metric_name, value in metrics.compute().items()
    }
    print(metric_values)

if __name__ == '__main__':
    test_batch_retrieval()
