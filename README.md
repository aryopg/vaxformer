<!-- omit in toc -->
# Vaxformer: Immunogenicity-controlled Transformer for Vaccine Design Against SARS-CoV-2

This repository contains pre-trained models, corpora, indices, and code for pre-training, finetuning, retrieving and evaluating for "Vaxformer: Immunogenicity-controlled Transformer for Vaccine Design Against SARS-CoV-2" (Paper in writing)

<!-- omit in toc -->
## Table of Contents
- [🛠️ Setup](#️-setup)
  - [Python packages](#python-packages)
  - [netMHCpan](#netmhcpan)
  - [Alphafold 2](#alphafold-2)
  - [Dataset](#dataset)
- [⌨️ Codebase Structure](#️-codebase-structure)
- [🤖 Training](#-training)
  - [Prepare the dataset](#prepare-the-dataset)
  - [Training the model](#training-the-model)
- [📝 Generating sequences](#-generating-sequences)
- [📈 Evaluation](#-evaluation)
  - [DDGun](#ddgun)
  - [netMHCpan](#netmhcpan-1)
  - [AlphaFold 2](#alphafold-2-1)


## 🛠️ Setup
### Python packages
This codebase requires the following dependencies:
```
- biopython
- matplotlib
- numpy
- pandas
- pydantic
- python-dotenv
- PyYAML
- tqdm
- wandb
```

We opted in to using conda as our package manager. The following will install all necessary dependencies for a GPU training:
```
ENV_NAME=vaxformer
conda create -n ${ENV_NAME} python=3.8 -y
conda activate ${ENV_NAME}
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia -y
pip install -r requirements.txt
```


### netMHCpan
> **Note**
> 
> netMHCpan only runs in Linux or Darwin machine

Follow this step to setup netMHCpan:
1. Download https://services.healthtech.dtu.dk/services/NetMHCpan-4.1/
2. Follow their installation instruction outlined in the `netMHCpan-4.1.readme` file


### Alphafold 2


### Dataset
Vaxformer is trained with a dataset of spike Covid proteins from [GI-SAID](https://gisaid.org/register/). You have to have the appropriate GI-SAID credentials to download the dataset.
To obtain comparable data splits, we inquired to [the author of the previous publication (ImmuneConstrainedVAE)](https://github.com/hcgasser/SpikeVAE).

## ⌨️ Codebase Structure
```
.
├── configs                                       # Config files
│   ├── test/                                     # Config files for sampling runs
│   └── train/                                    # Config files for training runs
├── datasets/                                     # Datasets of sequences and immunogenicity scores
├── scripts                                       # Scripts to start runs
│   ├── netmhcpan/
│   │   ├── netmhcpan_allele_scores.sh            # Script to run netMHCpan scoring for peptide files
│   │   ├── generate_peptides_from_sequences.py   # Script to generate peptides from sequences
│   ├── slurm/                                    # Slurm scripts for training and sampling runs
│   ├── sample.py                                 # Script to run sampling with a model of choice
│   └── train.py                                  # Script to run training with a model configuration of choice
├── src
│   ├── dataset
│   │   ├── __init__.py
│   │   ├── dataset.py                            # Dataset class to prepare and iterate through the dataset
│   │   └── tokenizer.py                          # Tokenizer class to preprocess the input sequence
│   ├── models 
│   │   ├── __init__.py
│   │   ├── baseline.py                           # Naive Bayes baseline model
│   │   ├── lstm.py                               # Conditional LSTM model
│   │   ├── vae.py                                # Conditional VAE model
│   │   └── vaxformer.py                          # The proposed Vaxformer model
│   ├── utils
│   │   ├── __init__.py
│   │   ├── common_utils.py                       # Utility functions to prepare trainings
│   │   └── model_utils.py                        # Utility functions for modelling purposes
│   ├── __init__.py
│   ├── configs.py                                # Pydantic configs validator
│   ├── constants.py                              # Constants for the training
│   └── trainer.py                                # Trainer class to handle training operations
├── requirements.txt                              # Necessary Python Packages
├── README.md                                     # You are here
```

## 🤖 Training

### Prepare the dataset

```

```

### Training the model
Once the sequences and immunogenicity scores datasets are obtained, we can run a training process.
```
python scripts/train.py \
--config_filepath=PATH/TO/TRAIN_CONFIG_FILE
```
Selections of train config files can be found in the [`configs/train/`](https://github.com/aryopg/vaxformer/tree/main/configs/train) folder.

Both LSTM and Vaxformer can be trained with `-small`, `-base`, or `-large` setting. They differ in terms of the number of hidden layers and their sizes.

## 📝 Generating sequences

Before any evaluation steps, we need to first generate the sequences with a pretrained model of choice

```
python scripts/sample.py \
--config_filepath=PATH/TO/TEST_CONFIG_FILE
--num_sequences=2000
```

Examples of test config files can be found in the [`configs/test/`](https://github.com/aryopg/vaxformer/tree/main/configs/test) folder.

## 📈 Evaluation


### DDGun


### netMHCpan
To evaluate the generated the netMHCpan
```

```

### AlphaFold 2

