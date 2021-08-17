import math
import torch
import pytorch_lightning as pl

class HardestTripletMarginLoss(torch.nn.Module):
    def __init__(self, margin: float, hardest_fraction: float = 0.0):
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

    def forward(self, preds, targets):
        if self.hardest_fraction <= 0.0:
            tml = lambda p,t:\
                hardest_triplet_margin_loss(p, t, self.margin)
        else:
            tml = lambda p,t:\
                chunked_hardest_fraction_triplet_margin_loss(
                    p, t, self.margin, self.hardest_fraction)

        loss1 = tml(preds, targets)
        loss2 = tml(preds.t(), targets.t())

        loss = (loss1 + loss2) / 2
        return loss

class HardestFractionDecay(pl.callbacks.Callback):
    _schedules = ['cosine', 'linear']
    def __init__(self, 
            total_steps: int, 
            min_fraction: float = 0.0,
            schedule: str = 'cosine',
        ):
        super().__init__()
        self.step_count = 0
        self.total_steps = total_steps
        self.min_fraction = min_fraction

        if schedule not in self._schedules:
            raise ValueError(f"schedule must be one of {self._schedules}")
        self.schedule = schedule

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, 
            batch_idx, dataloader_idx):
        if self.step_count > self.total_steps:
            return
        if self.step_count == self.total_steps:
            pl_module.loss.hardest_fraction = self.min_fraction
            pl_module.log('loss.hardest_fraction', self.min_fraction)
            return
        if self.step_count == 0:
            self.initial_hardest_fraction = pl_module.loss.hardest_fraction

        self.step_count += 1

        steps_fraction = self.step_count / self.total_steps

        if self.schedule == 'linear':
            decay = 1 - steps_fraction
        if self.schedule == 'cosine':
            decay = 1 - math.cos(
                    (3 * math.pi / 2)
                    + (math.pi/2) * steps_fraction)

        new_fraction = max(self.min_fraction, 
                self.initial_hardest_fraction * decay)

        pl_module.loss.hardest_fraction = new_fraction
        pl_module.log('loss.hardest_fraction', new_fraction, 
                on_step=True, on_epoch=False)

# naive python implementation
def naive_hardest_fraction_triplet_margin_loss(preds, targets, margin, hardest_fraction):
    assert preds.dim() == 2
    assert preds.size() == targets.size()
    assert 0 <= hardest_fraction <= 1

    def anchor_loss(preds, targets, margin, hardest_fraction):
        def single_triplet_loss(positive_score, negative_score, margin):
            return max(0, negative_score + margin - positive_score)

        def topf(x, f, largest=True):
            k = max(1, math.ceil(f * len(x)))
            out = sorted(x, reverse=largest)[:k]
            return out
        num_cols = preds.size(0)

        positive_preds = [
            preds[j] for j in range(num_cols)
            if targets[j] == True
        ]
        negative_preds = [
            preds[j] for j in range(num_cols)
            if targets[j] == False
        ]

        positive_preds = topf(positive_preds, hardest_fraction,
            largest=False)
        negative_preds = topf(negative_preds, hardest_fraction,
            largest=True)

        triplet_losses = [
            single_triplet_loss(positive_score, negative_score, margin)
            for positive_score in positive_preds 
            for negative_score in negative_preds
        ]

        loss = sum(triplet_losses) / len(triplet_losses)

        return loss

    num_rows = preds.size(0)
    row_losses = [
        anchor_loss(preds[i], targets[i], margin, hardest_fraction)
        for i in range(num_rows)
    ]
    row_loss = sum(row_losses) / len(row_losses)
    row_loss = row_loss / margin

    loss = row_loss
    return loss

# Multiple positives, hardest fraction, single anchor (=non-vectorized)
def anchor_hardest_fraction_triplet_margin_loss(
        preds, targets, margin, hardest_fraction):
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
    loss = losses.mean() / margin
    return loss

def top_fraction(x, fraction, largest=True):
    k = max(1, math.ceil(fraction * x.numel()))
    out = x.topk(k, largest=largest)[0]
    return out

# Single positive, hardest fraction, vectorized O(n^2) memory
def single_hardest_fraction_triplet_margin_loss(
        preds, targets, margin, hardest_fraction):
    if targets is None:
        targets = torch.eye(preds.size(0), device=preds.device)
    targets = targets.view(preds.size())
    target_indices = torch.nonzero(targets > 0.5)

    num_negatives = preds.size(1) - 1

    positive_scores = preds[target_indices[:,0],target_indices[:,1]].view(-1, 1)
    
    positive_scores = positive_scores.expand_as(preds)

    loss = (margin + preds - positive_scores).clamp(min=0)

    loss[target_indices[:,0],target_indices[:,1]] = 0

    k = max(1, math.ceil(hardest_fraction * num_negatives))

    loss = torch.topk(loss, k, 1)[0]
    loss = torch.sum(loss, 1) / k

    loss = loss.mean() / margin
    return loss

# Multiple positives, hardest fraction, chunk vectorized O(n^2) memory
def chunked_hardest_fraction_triplet_margin_loss(
        preds, targets, margin, hardest_fraction):
    targets = (targets >= 0.5)

    num_positives = targets.sum(dim=1)

    unique_num_positives = num_positives.unique()

    loss = torch.tensor(0.0, device=preds.device)
    for i, single_num_positives in enumerate(unique_num_positives):
        group_indices = (num_positives == single_num_positives)\
            .nonzero(as_tuple=True)

        group_loss = constant_hardest_fraction_triplet_margin_loss(
                preds[group_indices], targets[group_indices], 
                margin, hardest_fraction)
        loss += group_loss * group_indices[0].size(0)
    loss /= preds.size(0)

    return loss

# Multiple (same # of) positives, hardest fraction, vectorized O(n^2) memory
def constant_hardest_fraction_triplet_margin_loss(
        preds, targets, margin, hardest_fraction):

    num_docs = preds.size(0)
    targets = (targets >= 0.5)

    positive_indices = targets.nonzero(as_tuple=True)
    negative_indices = (~targets).nonzero(as_tuple=True)

    positives = preds[positive_indices].view(num_docs, -1)
    negatives = preds[negative_indices].view(num_docs, -1)

    pos_k = max(1, math.ceil(hardest_fraction * positives.size(1)))
    neg_k = max(1, math.ceil(hardest_fraction * negatives.size(1)))

    hardest_positives = positives.topk(pos_k, dim=1, largest=False)[0]
    hardest_negatives = negatives.topk(neg_k, dim=1, largest=True)[0]

    hardest_positives = hardest_positives.unsqueeze(1)
    hardest_negatives = hardest_negatives.unsqueeze(2)

    triplet_losses = (hardest_negatives + margin - hardest_positives).clamp(min=0)
    anchor_losses = triplet_losses.mean(dim=1)

    loss = anchor_losses.mean() / margin
    return loss
    
# Multiple positives, hardest triplet only, vectorized with O(n^2) memory
def hardest_triplet_margin_loss(
        preds, targets, margin):
    max_pred = preds.max(dim=1)[0]
    min_pred = preds.min(dim=1)[0]

    pos_mask = (targets >= 0.5).int()
    neg_mask = 1 - pos_mask

    positives = preds * pos_mask + neg_mask * (max_pred + 1).unsqueeze(1)
    hardest_pos = positives.min(dim=1)[0]

    negatives = preds * neg_mask + pos_mask * (min_pred - 1).unsqueeze(1)
    hardest_neg = negatives.max(dim=1)[0]

    anchor_losses = (hardest_neg + margin - hardest_pos).clamp(min=0)
    loss = anchor_losses.mean() / margin
    return loss

# Multiple positives, hardest fraction, vectorized with O(n^3) memory
def vectorized_hardest_fraction_triplet_margin_loss(
        preds, targets, margin, hardest_fraction):
    """
        Triplet margin loss with multiple positives and average over the hardest
        triplets.
        Implementation from user "KFrank" on discuss.pytorch.org
    """
    pos_mask = (targets >= 0.5).int()
    neg_mask = 1 - pos_mask

    num_docs = preds.size(1)

    max_pred = preds.max(dim=1)[0]
    min_pred = preds.min(dim=1)[0]

    # Mask negatives with large value
    sorted_pos = pos_mask * preds + neg_mask * (1 + max_pred).unsqueeze(1)
    # Sort in descending order (positives are last).
    sorted_pos = sorted_pos.sort(dim=1, descending=True)[0]
    # Determine how many positives to consider per anchor.
    pos_k = (hardest_fraction * pos_mask.sum(dim=1)).ceil().clamp(min=1)
    # Create a mask for values to be considered.
    hard_pos_mask = (
        (num_docs - pos_k).unsqueeze(1) <= torch.arange(num_docs)).int()

    # Same as above for the negatives.
    sorted_neg = neg_mask * preds + pos_mask * (min_pred - 1).unsqueeze(1)
    sorted_neg = sorted_neg.sort(dim=1, descending=False)[0]
    neg_k = (hardest_fraction * neg_mask.sum(dim=1)).ceil().clamp(min=1)
    hard_neg_mask = (
        (num_docs - neg_k).unsqueeze(1) <= torch.arange(num_docs)).int()

    # Compute all triplets losses (num_anchors, num_docs, num_docs).
    triplet_losses = (
        sorted_neg.unsqueeze(1) + margin - sorted_pos.unsqueeze(2)).clamp(min=0)
    # Create a mask for the triplets (num_anchors, num_docs, num_docs).
    triplet_mask = hard_neg_mask.unsqueeze(1) * hard_pos_mask.unsqueeze(2)

    anchor_losses = (triplet_mask * triplet_losses)
    anchor_losses = anchor_losses.sum(dim=(1,2)) / triplet_mask.sum(dim=(1,2))

    loss = anchor_losses.mean() / margin
    return loss

