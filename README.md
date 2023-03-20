<!-- omit in toc -->
# Vaxformer: Immunogenicity-controlled Transformer for Vaccine Design Against SARS-CoV-2

This repository contains pre-trained models, corpora, indices, and code for pre-training, finetuning, retrieving and evaluating for "Vaxformer: Immunogenicity-controlled Transformer for Vaccine Design Against SARS-CoV-2" (in writing)

Authors (equal contribution):
- [Aryo Pradipta Gema](https://aryopg.github.io/)
- [Michał Kobiela](https://www.linkedin.com/in/michal-kobiela137/)
- [Achille Fraisse](https://www.linkedin.com/in/achille-fraisse-4b3b11210/?originalSubdomain=fr)

<!-- omit in toc -->
## Table of Contents
- [🛠️ Setup](#️-setup)
  - [Python packages](#python-packages)
  - [netMHCpan](#netmhcpan)
  - [DDGun](#ddgun)
  - [Alphafold 2](#alphafold-2)
  - [Dataset](#dataset)
- [⌨️ Codebase Structure](#️-codebase-structure)
- [🤖 Training](#-training)
  - [Prepare the dataset](#prepare-the-dataset)
  - [Training the model](#training-the-model)
- [📝 Generating sequences](#-generating-sequences)
- [📈 Evaluation](#-evaluation)
  - [DDGun](#ddgun-1)
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
> netMHCpan only runs on Linux or Darwin machine

Follow this step to setup netMHCpan:
1. Download https://services.healthtech.dtu.dk/services/NetMHCpan-4.1/
2. Follow their installation instruction outlined in the `netMHCpan-4.1.readme` file

### DDGun

We included DDGun as a submodule to this repo. To install, you need to:
```
# Clone the DDGun submodule
git submodule update --init --recursive

# Install DDGun
cd submodules/ddgun
python setup.py
```

### Alphafold 2
TODO: Michal
1. Install [PyMol](https://pymol.org/2/)
2. Install AlphaFold


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
│   │   ├── compute_hits_from_peptides.py         # Script to compute netMHCpan hits and scores from peptides
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
├── submodules
│   └── ddgun/                                    # The DDGun package that we use for one of the evaluations
├── requirements.txt                              # Necessary Python Packages
├── README.md                                     # You are here
```

## 🤖 Training

### Prepare the dataset

Once you obtained the dataset, you need to first run netMHCpan to all sequences to compute the immunogenicity scores. First, you need to generate 9-mer peptides of each sequence of each dataset split:
```
python scripts/netmhcpan/generate_peptides_from_sequences.py \
--sequences_filepath=PATH/TO/TRAIN_DATASET.txt

python scripts/netmhcpan/generate_peptides_from_sequences.py \
--sequences_filepath=PATH/TO/VALID_DATASET.txt

python scripts/netmhcpan/generate_peptides_from_sequences.py \
--sequences_filepath=PATH/TO/TEST_DATASET.txt
```

These runs will create `.pep` files containing 9-mer peptides for each sequences which then can be passed to the netMHCpan:
```
cd scripts/netmhcpan/
bash netmhcpan_allele_scores.sh PATH/TO/TRAIN_DATASET_PEPTIDES_FILE.pep
bash netmhcpan_allele_scores.sh PATH/TO/VALID_DATASET_PEPTIDES_FILE.pep
bash netmhcpan_allele_scores.sh PATH/TO/TEST_DATASET_PEPTIDES_FILE.pep
```

These runs will create `.pep.out` files which contains the immunogenicity score for each peptides. Finally, we need to reconcile the peptides into sequences and calculate the hits scores of each sequence:
```
python scripts/netmhcpan/compute_hits_from_peptides.py \
--sequences_filepath=PATH/TO/TRAIN_DATASET.txt \
--peptides_dir=PATH/TO/TRAIN_PEPTIDES_DIR/ \
--peptide_file_prefix=TRAIN_PEPTIDE_FILE_PREFIX

python scripts/netmhcpan/compute_hits_from_peptides.py \
--sequences_filepath=PATH/TO/VALID_DATASET.txt \
--peptides_dir=PATH/TO/VALID_PEPTIDES_DIR/ \
--peptide_file_prefix=VALID_PEPTIDE_FILE_PREFIX

python scripts/netmhcpan/compute_hits_from_peptides.py \
--sequences_filepath=PATH/TO/TEST_DATASET.txt \
--peptides_dir=PATH/TO/TEST_PEPTIDES_DIR/ \
--peptide_file_prefix=TEST_PEPTIDE_FILE_PREFIX
```

Each of the peptide files that are generated from the previous command would have a prefix. For instance, from the previous commands you obtained 12 files (`vaxformer_large_A0101.pep.out`, ... , `vaxformer_large_B5801.pep.out`), then the peptide file prefix is `vaxformer_large`.

Practically, we can obtain quantiles from the distribution of hits of the training data split. For the sake of simplicity and reproducibility (and sanity check), you can check the Q1 and Q3 that we used to calculate the immunogenicity scores in the [`src/constants.py`](https://github.com/aryopg/vaxformer/tree/main/src/constants.py) which are denoted as `IMMUNOGENICITY_Q1` and `IMMUNOGENICITY_Q3` respectively.

These quantiles are used to compute the immunogenicity scores of each sequence. Sequences whose scores are lower than the Q1 are considered to have low immunogenicity scores (0), in between Q1 and Q3 are intermediate immunogenicity scores (1), and above Q3 are high immunogenicity scores (3).

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
TODO: Michal
```
python3 ddgun_seq.py wuhan_ddgan.fasta alpha_ddgan_mutations > output.txt
```


### netMHCpan
To evaluate the immunogenicity level of the generated sequences, we need to first generate 9-mer peptides for each generated sequences
```
python scripts/netmhcpan/generate_peptides_from_sequences.py \
--sequences_filepath=PATH/TO/GENERATED_SEQUENCES.txt
```

This run will create a `.pep` file. Next, we need to run the netMHCpan scoring script:
```
cd scripts/netmhcpan/
bash netmhcpan_allele_scores.sh PATH/TO/GENERATED_PEPTIDES_FILE.pep
```

This run will create a `.pep.out` file which contains the immunogenicity score for each peptides. Finally, we need to reconcile the peptides into sequences and calculate the score quantiles:
```
python scripts/netmhcpan/compute_hits_from_peptides.py \
--sequences_filepath=PATH/TO/GENERATED_SEQUENCES.fasta \
--peptides_dir=PATH/TO/PEPTIDES_DIR/ \
--peptide_file_prefix=PEPTIDE_FILE_PREFIX
```

### AlphaFold 2
TODO: Michal
1. Fold the proteins
2. Open PyMol and open the proteins and run:
```
align wuhan_alphafold, alpha_alphafold, cycles=0, transform=0
```
where `wuhan_alphafold` denotes the reference protein and `alpha_alphafold` denotes the generated protein.