import argparse
import yaml
import omegaconf
import hydra

import factories
import evaluation
import pickle

# to select the config, call like this:
# python script.py -cd config_directory -cn cfg.yaml

@hydra.main()
def main(training_cfg):
    print(training_cfg)
    dump_path = training_cfg['embeddings_dump_path']

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

    device = "cpu" if cfg['num_gpus'] == 0 else "cuda:0"

    embedding_batches = evaluation.generate_embeddings(
            model, dataloader, device=device)

    pickle.dump(embedding_batches, open(dump_path, 'wb'))

if __name__ == '__main__':
    main()
