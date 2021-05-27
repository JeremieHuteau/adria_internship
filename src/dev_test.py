import sys
import math

import torch
import torch.nn as nn

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

"""
def test_albumentations():
    import numpy as np
    import transforms
    import albumentations

    images = [
        np.array([[1, 2], [3, 4]]),
        np.array([[5, 6], [7, 8]]),
    ]
    texts = [
        "cat left dog riGht upright",
        "cat RIGHT dog lEfT leftist",
    ]

    class SetTransform(albumentations.BasicTransform):
        @property
        def targets(self):
            return {
                'images': self.apply_to_images,
                'texts': self.apply_to_texts,
            }

        def apply_to_images(self, images, **params):
            transformed_images = []
            for image in images:
                transformed_images.append(self.apply(image, **params))
            return transformed_images

    {'1': [], '2': []}
    for transform in composed_transform:
        if len(transform.targets) > 1:
            pass
            # apply the same to all

        else:
            pass
            # apply different

    def transform(images, texts, **params):
        for transform in self.transforms:
            if len(transform.targets & self.targets) > 1:
                transform_params = ???
                images = map(
                        lambda img: transform(image=img, transform_params),
                        images)
                texts = map(
                        lambda txt: transform(text=txt, transform_params),
                        texts)
            else:
                targets = images if ??? else texts
                targets = map(transform, targets)

    text_normalization = transforms.TextNormalization(p=1)
    flip = transforms.HorizontalFlip(p=1)

    augmentation = albumentations.Compose([text_normalization, flip])

    data = {'image': image, 'text': text}

    #print(augmentation(**data))

    data = text_normalization(**data)
    print(data)
    data = flip(**data)
    print(data)
"""

def pretraining_draft():
    # Step 0
    data = precompute_part_dataset()
    data.save()

    # Step 1
    data = PrecomputedPartXXXDataset()
    model = Model()
    model.part = Sequential()
    fit(model, data)

    # Step 2
    data = XXXDataset()
    model = Model.load_data_dict(checkpoint_data['data_dict'], strict=False)
    fit(model, data)

def part_loading(checkpoint):
    data = torch.load(checkpoint)
    print(data['state_dict'].keys())

    class LolModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.image_encoder = nn.Sequential

    data['state_dict']


if __name__ == '__main__':
    part_loading(sys.argv[1])
