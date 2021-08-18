import os
import pickle

import yaml
import omegaconf
import hydra
import pytorch_lightning as pl

import factories
import evaluation

@hydra.main(config_name="train", config_path="../cfg")
def main(cfg):
    import json
    cfg = omegaconf.OmegaConf.to_container(cfg, resolve=True)
    if cfg['check_cfg']:
        print(json.dumps(cfg, indent=2))
        return

    pl.seed_everything(cfg['seed'])

    logger = pl.loggers.TensorBoardLogger(
            cfg['paths']['logs_dir'], name=cfg['experiment_name'])
    experiment_version = logger.version
    experiment_key = f"{cfg['experiment_name']}_v{experiment_version}"

    vocabulary = pickle.load(open(
        cfg['transforms']['text']['vocabulary']['kwargs']['path'], 'rb'))
    cfg['model']['text_encoder']['kwargs']['vocabulary_size'] = \
            len(vocabulary)
    del vocabulary

    datamodule = factories.DataModuleFactory.create(
            cfg['data']['name'],
            **cfg['data']['kwargs'],
            transforms_cfg=cfg['transforms'],
            num_workers=cfg['num_cpus'],
            pin_memory=(cfg['num_gpus'] > 0),
    )

    datamodule.prepare_data()
    datamodule.setup('fit')

    if not cfg['not_interactive']:
        if cfg['visualize_inputs']:
            normalization_parameters = cfg['transforms']['image']['normalization']['kwargs']
            vocabulary = pickle.load(open(
                cfg['transforms']['text']['vocabulary']['kwargs']['path'], 'rb'))
            batch = next(iter(datamodule.train_dataloader()))
            datamodule.visualize_batch(batch,
                normalization_parameters=normalization_parameters,
                vocabulary=vocabulary,
                num_display=4)
            return

    experiment_checkpoints_dir = os.path.join(
            cfg['paths']['checkpoints_dir'], experiment_key)
    monitor_name = 'val/MeanCrossRecall@1'
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
            monitor=monitor_name, mode='max',
            dirpath=experiment_checkpoints_dir, 
            filename=experiment_key + '-{epoch}-val_R@1={'+monitor_name+':.3f}',
            auto_insert_metric_name=False,
            )
    callbacks = [
        checkpoint_callback,
        pl.callbacks.LearningRateMonitor(log_momentum=True),
    ]
    if (cfg['num_gpus'] > 0):
        callbacks.append(pl.callbacks.GPUStatsMonitor(
                memory_utilization=True, gpu_utilization=True, 
                intra_step_time=True, inter_step_time=True,))
    if (not cfg['fast_dev_run']) and (not cfg['tune']):
        callbacks.append(GitTagCreator())
    for callback in cfg['trainer']['callbacks']:
        callbacks.append(factories.CallbackFactory.create(
                callback['name'],
                **callback['kwargs']))
        
    model = factories.VSEModelFactory.create(
            cfg['model']['name'],
            image_encoder_cfg=cfg['model']['image_encoder'],
            text_encoder_cfg=cfg['model']['text_encoder'],
            loss_cfg=cfg['loss'],
            optimizer_cfg=cfg['optimizer'], 
            scheduler_cfg=cfg['scheduler'],
            **cfg['model']['kwargs'],
            )

    example_input_array = next(iter(datamodule.val_dataloader()))
    model.example_input_array = example_input_array

    if cfg['profiler'] is not None:
        output_filename = os.path.join(
               cfg['paths']['logs_dir'],  
               cfg['experiment_name'],
               f'profile_version_{experiment_version}.txt')
        if cfg['fast_dev_run']:
            output_filename = None

        if cfg['profiler'] == 'simple': 
            profiler = pl.profiler.SimpleProfiler(output_filename)
        if cfg['profiler'] == 'advanced': 
            profiler = pl.profiler.AdvancedProfiler(output_filename)
    else:
        profiler = pl.profiler.PassThroughProfiler()
    model.profiler = profiler

    trainer = pl.Trainer(
            **cfg['trainer']['kwargs'],
            logger=logger,
            callbacks=callbacks,
            gpus=cfg['num_gpus'], auto_select_gpus=cfg['num_gpus']>0, 
            accelerator='dp' if cfg['num_gpus'] > 1 else None,
            progress_bar_refresh_rate=0 if cfg['not_interactive'] else 1,
            weights_summary='full',
            fast_dev_run=cfg['fast_dev_run'],
            profiler=profiler,
    )

    if cfg['tune']:
        lr_finder = trainer.tuner.lr_find(model, datamodule,
                min_lr=1e-6, num_training=1000, early_stop_threshold=4.0)
        output_filename = os.path.join(
               cfg['paths']['logs_dir'],  
               cfg['experiment_name'],
               f'lr-tuning_version_{experiment_version}.pkl')
        pickle.dump(lr_finder.results, open(output_filename, 'wb'))
        if not cfg['not_interactive']:
            fig = lr_finder.plot(suggest=True)
            fig.show()
            input()
        return

    trainer.fit(model, datamodule)

    if not cfg['fast_dev_run']: 
        best_model_path = checkpoint_callback.best_model_path
        print(f"Loading best model from {best_model_path}")
        model = factories.VSEModelFactory.PRODUCTS[cfg['model']['name']]\
                .load_from_checkpoint(best_model_path)

        cfg['best_model_path'] = best_model_path
        cfg_dump_path = os.path.join(
                cfg['paths']['logs_dir'], cfg['experiment_name'], 
                f'version_{experiment_version}', 'cfg.yaml')
        yaml.dump(cfg, open(cfg_dump_path, 'w'))
        

    evaluation_device = "cpu" if cfg['num_gpus'] == 0 else "cuda:0"
    results = evaluation.evaluate(model, datamodule.val_dataloader(),
            device=evaluation_device)
    if not cfg['fast_dev_run']:
        results_path = os.path.join(
                cfg['paths']['logs_dir'], cfg['experiment_name'], 
                f'version_{experiment_version}', 'evaluation.pkl')
        print(f"Saving evaluation results to {results_path}")
        pickle.dump(results, open(results_path, 'wb'))

class GitTagCreator(pl.callbacks.Callback):
    """ Creates a git tag when sanity check has successfully passed.
        git tag name: "exp_{logger.name}_v{logger.version}"
    """
    def __init__(self):
        super().__init__()

    def on_sanity_check_end(self, trainer, pl_module):
        if trainer.fast_dev_run:
            return

        name = pl_module.logger.name
        version = pl_module.logger.version

        exp_hash = f'{name}_v{version}'
        tag_msg = f"Experiment: {name}"
        cmd = f'git tag -a exp_{exp_hash} -m "{tag_msg}"'
        os.system(cmd)
        print(f"Created experiment tag: exp_{exp_hash}")

if __name__ == '__main__':
    main()
