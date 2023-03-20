from typing import List, Optional

from pydantic import BaseModel


class DataSplitConfigs(BaseModel):
    sequences_path: str
    immunogenicity_scores_path: Optional[str]


class DatasetConfigs(BaseModel):
    train: DataSplitConfigs
    val: DataSplitConfigs
    test: DataSplitConfigs


class HyperparametersConfigs(BaseModel):
    # General hyperparameters
    max_seq_len: int
    dropout: float = 0.0
    batch_size: int = 64
    optimizer: str = "adam"
    learning_rate: float = 0.1
    grad_accumulation_step: int = 1
    # VAE specific
    hidden_dims: Optional[List[int]] = [2048, 1024, 512, 256, 128]
    latent_dim: Optional[int] = 2
    kl_target: Optional[int] = 0.1
    # Language Model specific
    embedding_dim: Optional[int] = 32
    hidden_dim: Optional[int] = 512
    nhead: Optional[int] = 8
    num_layers: Optional[int] = 6


class ModelConfigs(BaseModel):
    model_type: str
    model_state_dict_path: Optional[str]
    hyperparameters: Optional[HyperparametersConfigs]


class SetupConfigs(BaseModel):
    epochs: int = 20
    eval_steps: int = 1
    checkpoint_steps: int = 1
    device: Optional[int] = 0
    random_seed: int = 1234
    outputs_dir: str = "outputs"


class TrainingConfigs(BaseModel):
    dataset_configs: DatasetConfigs
    model_configs: ModelConfigs
    training_configs: SetupConfigs


class TestingConfigs(BaseModel):
    random_seed: int = 1234
    outputs_dir: str = "test_outputs"
    device: Optional[int] = 0
    pretrained_model_state_dict_path: str
    pretrained_model_configs_path: str
