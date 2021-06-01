import math
import torch
import torchmetrics

import retrieval_metrics

def evaluate(model, dataloader, device="cpu"):
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        embedded_batches = []

        for batch_idx, batch in enumerate(dataloader):
            images = batch['images']
            texts = batch['texts']
            lengths = batch['lengths']
            image_positive_pairs = batch['positive_pairs']

            image_embeddings_batch, text_embeddings_batch = model(
                    images.to(device), texts.to(device), lengths.to(device))

            image_embeddings_batch = image_embeddings_batch.cpu()
            text_embeddings_batch = text_embeddings_batch.cpu()

            embedded_batches.append({
                'images': image_embeddings_batch,
                'texts': text_embeddings_batch,
                'positive_pairs': batch['positive_pairs'].cpu(),
            })

        # Put the model on the cpu in case its similarity function uses 
        # learned tensors.
        model = model.to("cpu")

        k_values = [1, 5, 10]

        image2text_metrics = torchmetrics.MetricCollection({
            f'I2T_R@{k}': retrieval_metrics.RecallAtK(k)
            for k in k_values
            })
        batch_retrieval(
                embedded_batches,
                source_key='images', target_key='texts', 
                swap_positive_pairs=False,
                score_fn=model.similarity, 
                metrics=image2text_metrics)

        text2image_metrics = torchmetrics.MetricCollection({
            f'T2I_R@{k}': retrieval_metrics.RecallAtK(k)
            for k in k_values
            })
        batch_retrieval(
                embedded_batches,
                source_key='texts', target_key='images', 
                swap_positive_pairs=True,
                score_fn=model.similarity, 
                metrics=text2image_metrics)

    image2text_retrieval_scores = {
            metric_name: value.item()
            for metric_name, value in image2text_metrics.compute().items()}
    text2image_retrieval_scores = { 
            metric_name: value.item()
            for metric_name, value in text2image_metrics.compute().items()}
    print("K:", k_values)
    print("Image to Text R@K:", image2text_retrieval_scores)
    print("Text to Image R@K:", text2image_retrieval_scores)

    results = {
            'K': k_values, 
            'I2T_R@K': image2text_retrieval_scores,
            'T2I_R@K': text2image_retrieval_scores,
            }
    return results

def batch_retrieval(batches, source_key, target_key, swap_positive_pairs,
        score_fn, metrics):

    batches_num_targets = [batch[target_key].size(0) for batch in batches]
    current_targets_offset = 0

    targets = torch.cat([batch[target_key] for batch in batches], dim=0)

    for i, batch in enumerate(batches):
        batch_sources = batch[source_key]

        batch_scores = score_fn(batch_sources, targets)

        batch_positive_pairs = batch['positive_pairs'].clone()
        if swap_positive_pairs:
            swapped_positive_pairs = torch.zeros(batch_positive_pairs.size(),
                    dtype=torch.long)
            swapped_positive_pairs[:,0] = batch_positive_pairs[:,1]
            swapped_positive_pairs[:,1] = batch_positive_pairs[:,0]
            batch_positive_pairs = swapped_positive_pairs
        batch_positive_pairs[:,1] += current_targets_offset
        current_targets_offset += batches_num_targets[i]

        batch_labels, batch_indices, _ = \
                retrieval_metrics.positive_sparse2dense(
                    batch_positive_pairs, 
                    [batch_scores.size(0), batch_scores.size(1)]
                )

        metrics.update(batch_scores, batch_labels, batch_indices)

    return metrics
    metric_values = [metric.compute().item() for metric in metrics]
    return metric_values

"""
Old evaluate: basically operates on transposed batch list.
def evaluate(model, dataloader, device="cpu"):
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        image_embeddings_batches = []
        text_embeddings_batches = []
        image_positive_pairs_batches = []
        text_positive_pairs_batches = []

        for batch_idx, batch in enumerate(dataloader):
            images = batch['images']
            texts = batch['texts']
            lengths = batch['lengths']
            image_positive_pairs = batch['positive_pairs']

            image_embeddings_batch, text_embeddings_batch = model(
                    images.to(device), texts.to(device), lengths.to(device))

            image_embeddings_batch = image_embeddings_batch.cpu()
            text_embeddings_batch = text_embeddings_batch.cpu()

            image_embeddings_batches.append(image_embeddings_batch)
            text_embeddings_batches.append(text_embeddings_batch)

            text_positive_pairs = torch.zeros(image_positive_pairs.size(),
                    dtype=torch.long)
            text_positive_pairs[:,0] = image_positive_pairs[:,1]
            text_positive_pairs[:,1] = image_positive_pairs[:,0]

            image_positive_pairs_batches.append(image_positive_pairs)
            text_positive_pairs_batches.append(text_positive_pairs)

        # Put the model on the cpu in case its similarity function uses 
        # learned tensors.
        model = model.to("cpu")

        k_values = [1, 5, 10]

        image2text_metrics = [
            retrieval_metrics.RecallAtK(k)
            for k in k_values]
        image2text_retrieval_scores = batch_retrieval(
                image_embeddings_batches, text_embeddings_batches,
                image_positive_pairs_batches,
                model.similarity, 
                image2text_metrics)

        text2image_metrics = [
            retrieval_metrics.RecallAtK(k)
            for k in k_values]
        text2image_retrieval_scores = batch_retrieval(
                text_embeddings_batches, image_embeddings_batches,
                text_positive_pairs_batches,
                model.similarity, 
                text2image_metrics)

    print("K:", k_values)
    print("Image to Text R@K:", image2text_retrieval_scores)
    print("Text to Image R@K:", text2image_retrieval_scores)

    results = {
            'K': k_values, 
            'I2T_R@K': image2text_retrieval_scores,
            'T2I_R@K': text2image_retrieval_scores,
            }
    return results

def batch_retrieval(source_batches, target_batches, pairs_batches, 
        score_fn, metrics):

    num_batches_targets = [target_batch.size(0) for target_batch in target_batches]
    current_targets_offset = 0

    targets = torch.cat(target_batches, dim=0)

    batches = zip(source_batches, pairs_batches, num_batches_targets)
    for source_batch, pairs_batch, num_batch_target in batches:
        batch_scores = score_fn(source_batch, targets)

        pairs_batch[:,1] += current_targets_offset
        current_targets_offset += num_batch_target

        batch_labels, batch_indices, _ = \
                retrieval_metrics.positive_sparse2dense(
                    pairs_batch, [batch_scores.size(0), batch_scores.size(1)])

        for metric in metrics:
            metric.update(batch_scores, batch_labels, batch_indices)

    metric_values = [metric.compute().item() for metric in metrics]
    return metric_values
"""

def matrix_batch_retrieval(sources, targets, score_fn, metric,
        max_num_scores: int = 2**25):

    num_sources = sources.size(0)
    num_targets = targets.size(0)
    # Biggest power of 2 such that the number of scores doesn't exceed
    # max_num_scores
    batch_size = 2**math.floor(math.log(max_num_scores // num_targets, 2))

    batch_start = 0
    batch_end = min(batch_start+batch_size, num_sources)
    while batch_start < num_targets:
        sources_batch = sources[batch_start:batch_end]
        batch_positive_indices = torch.arange(batch_start, batch_end)

        batch_scores = score_fn(sources_batch, targets)
        metric.update(batch_scores, batch_positive_indices) 

        batch_start += batch_size
        batch_end = min(batch_start+batch_size, num_sources)
    return metric.compute()


