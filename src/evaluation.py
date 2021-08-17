import math
import torch
import torchmetrics
import hydra

import factories
import retrieval_metrics as rm

def evaluate(model, dataloader, device="cpu"):
    with torch.no_grad():
        embedded_batches = generate_embeddings(model, dataloader, device)

        # Put the model on the cpu in case its similarity function uses 
        # learned tensors.
        model = model.to("cpu")

        ks = [1, 5, 10]
        query_target_strs = ['image2text', 'text2image']

        binary_metrics = {
            qt: torchmetrics.MetricCollection({
                f"{qt}_Retrieval@{k}": rm.RetrievalAtK(k)
                for k in ks
            })
            for qt in query_target_strs}

        ratio_metrics = {
            qt: torchmetrics.MetricCollection({
                **{
                    f"{qt}_Recall@{k}": rm.RecallAtK(k)
                    for k in ks
                },
                f"{qt}_MeanRank": rm.MeanRank(),
            })
            for qt in query_target_strs}

        batch_retrieval(
                embedded_batches,
                source_key='images', target_key='texts', 
                swap_positive_pairs=False,
                score_fn=model.similarity, 
                metrics={
                    'binary': binary_metrics['image2text'],
                    'ratio': ratio_metrics['image2text'],
                })

        batch_retrieval(
                embedded_batches,
                source_key='texts', target_key='images', 
                swap_positive_pairs=True,
                score_fn=model.similarity, 
                metrics={
                    'binary': binary_metrics['text2image'],
                    'ratio': ratio_metrics['text2image'],
                })

    binary_metric_values = {
        qt: {
            metric_name: metric_value.item()
            for metric_name, metric_value 
            in binary_metrics[qt].compute().items()
        }
        for qt in query_target_strs
    }
    ratio_metric_values = {
        qt: {
            metric_name: metric_value.item()
            for metric_name, metric_value 
            in ratio_metrics[qt].compute().items()
        }
        for qt in query_target_strs
    }

    results = {}
    for qt in query_target_strs:
        results.update(binary_metric_values[qt])
        results.update(ratio_metric_values[qt])

    for k,v in results.items():
        print(k, v)

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

        batch_labels, batch_indices, _ = rm.positive_sparse2dense(
            batch_positive_pairs, 
            [batch_scores.size(0), batch_scores.size(1)]
        )

        batch_ranks = rm.target_ranks(batch_scores, batch_labels)
        batch_counts = rm.target_counts(batch_labels)

        if 'binary' in metrics:
            metrics['binary'].update(batch_ranks)
        if 'ratio' in metrics:
            metrics['ratio'].update(batch_ranks, batch_counts)

    return metrics


@hydra.main()
def main(training_cfg):

    data_cfg = training_cfg['data']
    transforms_cfg = training_cfg['transforms']

    model_checkpoint_path = training_cfg['best_model_path']

    model = factories.VSEModelFactory.PRODUCTS[training_cfg['model']['name']]\
            .load_from_checkpoint(model_checkpoint_path)

    datamodule = factories.DataModuleFactory.create(
            data_cfg['name'],
            **data_cfg['kwargs'],
            transforms_cfg=transforms_cfg,
            num_workers=training_cfg['num_cpus'],
            pin_memory=(training_cfg['num_gpus'] > 0),
    )
    datamodule.prepare_data()
    datamodule.setup('fit')
    dataloader = datamodule.val_dataloader()

    device = "cpu" if training_cfg['num_gpus'] == 0 else "cuda:0"

    if training_cfg['evaluation_mode'] == 'dump_embeddings':
        dump_path = training_cfg['embeddings_dump_path']

        embedding_batches = evaluation.generate_embeddings(
                model, dataloader, device=device)

        pickle.dump(embedding_batches, open(dump_path, 'wb'))
    else:
        evaluate(model, dataloader, device=device)

if __name__ == '__main__':
    main()
