<!-- omit in toc -->
# Vaxformer: Immunogenicity-controlled Transformer for Vaccine Design Against SARS-CoV-2

This repository contains pre-trained models, corpora, indices, and code for pre-training, finetuning, retrieving and evaluating for the paper Vaxformer: Immunogenicity-controlled Transformer for Vaccine Design Against SARS-CoV-2

<!-- omit in toc -->
## Table of Contents
- [Installation](#installation)
  - [Python packages](#python-packages)
  - [netMHCpan](#netmhcpan)
  - [Alphafold 2](#alphafold-2)
- [Codebase](#codebase)
- [Generation](#generation)
  - [Training](#training)
  - [Sampling](#sampling)
- [Evaluation](#evaluation)
  - [DDGun](#ddgun)
  - [netMHCpan](#netmhcpan-1)
  - [AlphaFold 2](#alphafold-2-1)


## Installation
### Python packages
This codebase requires the following dependencies:
```

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
> :warning: netMHCpan only runs in Linux or Darwin machine

Follow this step to setup netMHCpan:
1. Download https://services.healthtech.dtu.dk/services/NetMHCpan-4.1/
2. Follow their installation instruction in "netMHCpan-4.1.readme"


### Alphafold 2


## Codebase


## Generation
### Training

### Sampling


## Evaluation
### DDGun

### netMHCpan

### AlphaFold 2

