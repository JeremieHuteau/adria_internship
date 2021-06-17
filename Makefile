SETUP_DIR = setup
include $(SETUP_DIR)/Makefile

###############################################################################
# OVERRIDABLE VARIABLES
###############################################################################
NUM_CPUS = $(DEFAULT_NUM_CPUS)
NUM_GPUS = $(DEFAULT_NUM_GPUS)

# (COCO) 
DATASET = COCO

# Arguments to be passed to the training scripts. Uses hydra syntax.
#override HYDRA_ARGS += 
HYDRA_ALL_ARGS = num_cpus=$(NUM_CPUS) num_gpus=$(NUM_GPUS) data=$(DATASET) \
		 $(HYDRA_ARGS)

###############################################################################
# CONSTANT/DERIVED VARIABLES
###############################################################################

include $(SETUP_DIR)/Makefile$(DATASET)
DATASET_DIR = $($(DATASET)_ROOT_DIR)

OUTPUT_DIRS = $(DATA_DIR) $(ARTIFACTS_DIR) \
	      $(LOGS_DIR) $(CHECKPOINTS_DIR) $(TRANSFORMS_DIR)

###############################################################################
# RULES
###############################################################################

### PRODUCTS RULES

.PHONY: all
all: model

.PHONY: embeddings
embeddings:
	$(PYTHON) $(SRC_DIR)/save_embeddings.py \
		-cd $(LOGS_DIR)/$(RUN_DIR) -cn cfg.yaml \
		+embeddings_dump_path=$(LOGS_DIR)/$(RUN_DIR)/embeddings.pkl \
		$(HYDRA_ALL_ARGS)

.PHONY: model
# Train model
model: $(DATASET_DIR)/preprocessed $(TRANSFORMS_DIR)/$(DATASET)_train_vocabulary.pkl
	$(PYTHON) $(SRC_DIR)/train_model.py \
		$(HYDRA_ALL_ARGS)

# Create vocabulary transform
.PHONY: vocabulary
vocabulary: $(TRANSFORMS_DIR)/$(DATASET)_train_vocabulary.pkl
$(TRANSFORMS_DIR)/$(DATASET)_train_vocabulary.pkl: \
		$(DATASET_DIR)/preprocessed/annotations
	$(PYTHON) $(SRC_DIR)/create_vocabulary.py \
		$(DATASET_DIR)/preprocessed/annotations/$($(DATASET)_TRAIN_ANNOTATIONS) \
		$(TRANSFORMS_DIR)/$(DATASET)_train_vocabulary.pkl

# Prepare data
.PHONY: preprocessed
preprocessed: $(DATASET_DIR)/preprocessed
$(DATASET_DIR)/preprocessed: $(DATASET_DIR)/preprocessed/images $(DATASET_DIR)/preprocessed/annotations

.PHONY: preprocessed_images
preprocessed_images: $(DATASET_DIR)/preprocessed/images
$(DATASET_DIR)/preprocessed/images: $(DATASET_DIR)/raw/images
	mkdir -p $(DATASET_DIR)/preprocessed/images
	$(PYTHON) $(SRC_DIR)/image_preprocessing.py \
		$(DATASET_DIR)/raw/images \
		$(DATASET_DIR)/preprocessed/images \
		--width 256 --height 256 \
		--num-cpus $(NUM_CPUS)

.PHONY: preprocessed_captions
preprocessed_captions: $(DATASET_DIR)/preprocessed/annotations
$(DATASET_DIR)/preprocessed/annotations: $(DATASET_DIR)/raw/annotations
	cp -r $(DATASET_DIR)/raw/annotations $(DATASET_DIR)/preprocessed

### SETUP RULES

.PHONY: setup
setup: $(OUTPUT_DIRS) configuration
	conda env create -n $(CONDA_ENV_NAME) -f $(CONDA_ENV_YAML)
	$(PIP) install hydra-core --upgrade --pre
	$(PIP) install albumentations
	$(PYTHON) $(SETUP_DIR)/setup.py

.PHONY: configuration
configuration: $(CFG_DIR)/environment.yaml
$(CFG_DIR)/environment.yaml:
	@echo -n "" > $@
	@echo -e "paths:" >> $@
	@echo -e "  data_dir: $(DATA_DIR)" >> $@
	@echo -e "  logs_dir: $(LOGS_DIR)" >> $@
	@echo -e "  checkpoints_dir: $(CHECKPOINTS_DIR)" >> $@
	@echo -e "  transforms_dir: $(TRANSFORMS_DIR)" >> $@
	@echo -e "num_cpus: $(DEFAULT_NUM_CPUS)" >> $@
	@echo -e "num_gpus: $(DEFAULT_NUM_GPUS)" >> $@
	@echo -e "not_interactive: $(NON_INTERACTIVE_ENV)" >> $@
	
$(OUTPUT_DIRS): %: 
	@mkdir -p $@

