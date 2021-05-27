# Requirements

  - `conda`
  - `make`

Both requirements can be dispensed with, but will require using alternatives (installing packages with `pip`, running commands manually).

# Setup
```
cp setup/Makefile.base setup/Makefile
```
Then edit `setup/Makefile` to reflect your particular setup.


```
make setup
```

# Usage
To run the whole pipeline with default options:
```
make
```

Trained models can be found in `$(CHECKPOINTS_DIR)`, training/evaluation data in `$(LOGS_DIR)`.
Logs can be displayed with `tensorboard --logdir $(LOGS_DIR)`.

The training code uses [hydra](https://github.com/facebookresearch/hydra) to handle configurations. 
The configuration files can be found in the `cfg` directory, with the main configuration being `cfg/train.yaml`.

You can override some Makefile variables from the command line. Check `Makefile` to see which.
For example, to check the configuration of the `dev` experiment, you can run (assuming training is the only step to do):
```
make HYDRA_ARGS="+experiment=dev check_cfg=true"
```
