import math
import torch
import torchmetrics

import retrieval_metrics

def evaluate(model, dataloader, device="cpu"):
    with torch.no_grad():
        embedded_batches = generate_embeddings(model, dataloader, device)

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

def generate_embeddings(model, dataloader, device='cpu'):
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        embedded_batches = []

        for batch_idx, batch in enumerate(dataloader):
            images = batch['images']
            texts = batch['texts']
            lengths = batch['lengths']

            image_embeddings, text_embeddings = model(
                    images.to(device), texts.to(device), lengths.to(device))

            embedded_batches.append({
                'images': image_embeddings.cpu(),
                'texts': text_embeddings.cpu(),
                'positive_pairs': batch['positive_pairs'].cpu(),
                'images_data': batch['images_data'],
                'annotations_data': batch['annotations_data'],
            })

    return embedded_batches

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
                dtype=torch.long, device=batch_positive_pairs.device)
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

