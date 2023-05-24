import argparse
import os
import sys

import torch

sys.path.append(os.getcwd())

from dotenv import load_dotenv

load_dotenv("env/.env")

import wandb

from src.configs import TrainingConfigs
from src.dataset.dataset import SequenceDataset
from src.trainer import Trainer
from src.utils import common_utils


def argument_parser():
    parser = argparse.ArgumentParser(
        description="Vaxformer project"
    )
    parser.add_argument("--config_filepath", type=str, required=True)
    parser.add_argument("--log_to_wandb", action="store_true")
    args = parser.parse_args()
    return args


def main():
    args = argument_parser()
    configs = TrainingConfigs(**common_utils.load_yaml(args.config_filepath))

    common_utils.setup_random_seed(configs.training_configs.random_seed)
    outputs_dir = common_utils.setup_experiment_folder(
        os.path.join(os.getcwd(), configs.training_configs.outputs_dir)
    )
    common_utils.save_training_configs(configs, outputs_dir)
    device = common_utils.setup_device(configs.training_configs.device)
    print(f"Running on {device}")

    wandb.init(
        project="spike-protein-design",
        entity="protein-kge",
        mode="online" if args.log_to_wandb else "disabled",
    )
    wandb.config.update(configs.dict())
    wandb.config.update(
        {"outputs_dir": outputs_dir, "device_count": torch.cuda.device_count()}
    )

    if configs.model_configs.model_type == "vae":
        sequence_one_hot = True
        label_one_hot = True
        prepend_start_token = False
    if configs.model_configs.model_type in [
        "vaxformer",
        "lstm",
    ]:
        sequence_one_hot = False
        label_one_hot = False
        prepend_start_token = True

    train_dataset = SequenceDataset(
        configs.dataset_configs,
        "train",
        configs.model_configs.hyperparameters.max_seq_len,
        sequence_one_hot,
        label_one_hot,
        prepend_start_token=prepend_start_token,
    )
    val_dataset = SequenceDataset(
        configs.dataset_configs,
        "val",
        configs.model_configs.hyperparameters.max_seq_len,
        sequence_one_hot,
        label_one_hot,
        prepend_start_token=prepend_start_token,
    )
    test_dataset = SequenceDataset(
        configs.dataset_configs,
        "test",
        configs.model_configs.hyperparameters.max_seq_len,
        sequence_one_hot,
        label_one_hot,
        prepend_start_token=prepend_start_token,
    )

    data_stats = {
        "train_num_sequences": len(train_dataset),
        "val_num_sequences": len(val_dataset),
        "test_num_sequences": len(test_dataset),
    }
    wandb.config.update(data_stats)

    trainer = Trainer(
        configs,
        train_dataset,
        val_dataset,
        test_dataset,
        outputs_dir,
        device=device,
    )

    trainer.train()


if __name__ == "__main__":
    main()
