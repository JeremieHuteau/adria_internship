Code for running experiments with Visual Semantic Embedding models on MS-COCO Captions, using multiple positive captions instead of a single one.

# Requirements

  - `conda`
  - `make`

# Setup
```
cp setup/Makefile.base setup/Makefile
```
Then edit `setup/Makefile` to reflect your particular setup. By default, `setup/Makefile` will use the project directory to store artifacts and will use a GPU to train.

Make the `setup` target to generate a `environment.yaml` which will reflect `setup/Makefile` and be usable by the python scripts. This will also create a conda environment with the necessary packages.
```
make setup
```

# Usage
Data needs to be downloaded manually to the data folder defined in `setup/Makefile`.
To run the whole pipeline with default options:
```
make
```

Trained models are saved in `$(ARTIFACTS_DIR)/checkpoints`, training logs and evaluations in `$(ARTIFACTS_DIR)/logs`.
Logs can be displayed with `tensorboard --logdir $(ARTIFACTS_DIR)/logs`.

The configuration files can be found in the `cfg` directory, with the main configuration being `cfg/train.yaml`.


To run the VSE++/VSE\*\* experiment:
`make HYDRA_ARGS="+experiment=VSEplus"`
`make HYDRA_ARGS="+experiment=VSEstar"`

To generate the embeddings for a given version of an experiment:
`make embeddings RUN_DIR=$(EXPERIMENT_NAME)/$(VERSION)`
(`RUN_DIR` will be looked up in the logs dir, and the embeddings will be saved there)

The main script can also easily be called manually, here with a larger image encoder: `python src/train_model.py +experiment=VSEstar model.image_encoder.kwargs.architecture_name=resnet101`.

# Code organisation
Code is organised with the [Pytorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning) framework, and uses [hydra](https://github.com/facebookresearch/hydra) to handle the configuration:

 - classes are instantiated using the factories in `src/factories.py`
 - training is handled by `src/train_model.py` which loads `cfg/train.yaml`
 - data is loaded in `src/coco_captions_dataset.py`, and transformed/batched in `src/coco_captions_datamodule.py`
 - model is defined in `src/vse_models.py`, loss function in `src/triplet_margin_loss.py` and metrics in `src/retrieval_metrics.py`

To check the current configuration of the training script for an experiment, use `python src/train_model.py +experiment=$(EXPERIMENT) check_cfg=true`.

