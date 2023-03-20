import os

import torch
import tqdm
import wandb
from torch import nn
from torch.utils.data import DataLoader

from .configs import ModelConfigs, TrainingConfigs
from .constants import START_TOKEN
from .dataset.dataset import SequenceDataset
from .models.lstm import VaxLSTM
from .models.vae import SpikeFCVAE
from .models.vaxformer import Vaxformer

MODELS_MAP = {"spike_vae": SpikeFCVAE, "vaxformer": Vaxformer, "lstm": VaxLSTM}


class Trainer:
    def __init__(
        self,
        configs: TrainingConfigs,
        train_dataset: SequenceDataset,
        val_dataset: SequenceDataset,
        test_dataset: SequenceDataset,
        outputs_dir: str,
        device: torch.device = None,
        verbose: bool = False,
    ):
        """
        A Trainer class that contains necessary components for training and operational

        Args:
            configs (TrainingConfigs): Config file for training
            train_dataset (SequenceDataset): Training Dataset
            val_dataset (SequenceDataset): Validation Dataset
            test_dataset (SequenceDataset): Testing Dataset
            outputs_dir (str): Path to the output directory
            device (torch.device, optional): Device used for the runs. Defaults to None.
            verbose (bool, optional): Regulate TQDM verbosity. Defaults to False.
        """
        # General setup
        self.configs = configs

        # Dataset setup
        self.padding_idx = train_dataset.tokenizer.enc_dict["-"]
        if START_TOKEN in train_dataset.tokenizer.enc_dict:
            self.start_idx = train_dataset.tokenizer.enc_dict[START_TOKEN]
        else:
            self.start_idx = None

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

        # Operational setup
        self.outputs_dir = outputs_dir
        self.checkpoint_path = os.path.join(outputs_dir, "checkpoint")
        self.eval_steps = self.configs.training_configs.eval_steps
        self.checkpoint_steps = self.configs.training_configs.checkpoint_steps
        self.verbose = verbose

        # Modelling setup
        self.model_type = configs.model_configs.model_type
        self.device = device

        self.model = self.setup_model(configs.model_configs)
        self.optimizer = self.setup_optimizer(
            configs.model_configs.hyperparameters.optimizer
        )
        self.grad_accumulation_step = (
            configs.model_configs.hyperparameters.grad_accumulation_step
        )

        if configs.model_configs.model_state_dict_path:
            self.load_checkpoint(configs.model_configs.model_state_dict_path)

    def setup_model(self, model_configs: ModelConfigs) -> nn.Module:
        """
        Setup model based on the model type, number of entities and relations
        mentioned in the config file

        Args:
            model_configs (dict): configurations

        Returns:
            nn.Module: The model to be trained
        """

        kwargs = {}
        if self.model_type in ["vaxformer", "lstm"]:
            kwargs.update(
                {"padding_idx": self.padding_idx, "start_idx": self.start_idx}
            )

        if model_configs.model_type not in MODELS_MAP:
            raise NotImplementedError(
                f"Model {model_configs.model_type} not implemented"
            )

        return MODELS_MAP[model_configs.model_type](
            model_configs, self.device, **kwargs
        ).to(self.device)

    def setup_optimizer(self, optimizer: str) -> torch.optim.Optimizer:
        """
        Setup optimizer based on the optimizer name

        Args:
            optimizer (str): optimizer name

        Returns:
            torch.optim.Optimizer: Optimizer class
        """
        if optimizer == "adam":
            return torch.optim.Adam(
                list(self.model.parameters()),
                lr=self.configs.model_configs.hyperparameters.learning_rate,
            )
        else:
            raise NotImplementedError(f"Optimizer {optimizer} is not implemented")

    def epoch(self, epoch, data_split) -> float:
        """
        Method that represents one epoch (multiple training steps).

        Args:
            dataset (tbd): Dataset object

        Returns:
            float: the averaged loss of the epoch
        """

        # Set model mode to train
        if data_split == "train":
            self.model.train()
            dataset = self.train_dataset
        elif data_split == "val":
            self.model.eval()
            dataset = self.val_dataset

        ## Create DataLoader for sampling
        data_loader = DataLoader(
            dataset,
            self.configs.model_configs.hyperparameters.batch_size,
            shuffle=True,
        )

        if self.model_type == "vae":
            total_loss = {
                "combined_loss": 0,
                "reconstruction_loss": 0,
                "kl_divergence": 0,
            }
        elif self.model_type in ["vaxformer", "lstm"]:
            total_loss = {"combined_loss": 0, "perplexity": 0}

        total_examples = 0
        for iteration, batch in tqdm.tqdm(
            enumerate(data_loader),
            desc=f"EPOCH {epoch}, {data_split}, batch ",
            unit="",
            total=len(data_loader),
            disable=self.verbose,
        ):
            # Load data to GPU
            batch_sequences, batch_immunogenicity_scores = batch
            batch_sequences = batch_sequences.to(self.device)
            batch_immunogenicity_scores = batch_immunogenicity_scores.to(self.device)

            # Run one step of training
            outputs = self.model.step(batch_sequences, batch_immunogenicity_scores)
            if self.model_type == "vae":
                combined_loss = outputs["combined_loss"]
                reconstruction_loss = outputs["reconstruction_loss"]
                kl_divergence = outputs["kl_divergence"]
            elif self.model_type in ["vaxformer", "lstm"]:
                combined_loss = outputs["loss"]
                perplexity = outputs["perplexity"]

            # Stop training if loss becomes inf or nan
            if torch.isinf(combined_loss):
                raise Exception("Loss is infinity. Stopping training...")
            elif torch.isnan(combined_loss):
                raise Exception("Loss is NaN. Stopping training...")

            # Average by gradient accumulation step if any
            combined_loss = combined_loss / self.grad_accumulation_step

            if data_split == "train":
                # Compute combined_loss
                combined_loss.backward()

                # Run backprop if iteration falls on the gradient accumulation step
                if ((iteration + 1) % self.grad_accumulation_step == 0) or (
                    (iteration + 1) == len(data_loader)
                ):
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            # Accumulate epoch loss
            num_examples = batch_sequences.size(0)
            if self.model_type == "vae":
                total_loss["combined_loss"] += combined_loss.item() * num_examples
                total_loss["reconstruction_loss"] += (
                    reconstruction_loss.item() * num_examples
                )
                total_loss["kl_divergence"] += kl_divergence.item() * num_examples
            elif self.model_type in ["vaxformer", "lstm"]:
                total_loss["combined_loss"] += combined_loss.item() * num_examples
                total_loss["perplexity"] += perplexity.item() * num_examples
            total_examples += num_examples

        return {
            metrics_name: metrics_value / total_examples
            for metrics_name, metrics_value in total_loss.items()
        }

    def train(self):
        for epoch in range(1, 1 + self.configs.training_configs.epochs):
            # Run one epoch of training
            train_loss = self.epoch(epoch, "train")

            # Run evaluation if it is the eval step
            if epoch % self.eval_steps == 0:
                # Run one epoch of validation without any gradients computation
                with torch.no_grad():
                    val_loss = self.epoch(epoch, "val")

                # Log to WandB
                wandb_logs = {
                    "epoch": epoch,
                }
                wandb_logs.update(
                    {
                        f"train_{metric_name}": value
                        for metric_name, value in train_loss.items()
                    }
                )
                wandb_logs.update(
                    {
                        f"val_{metric_name}": value
                        for metric_name, value in val_loss.items()
                    }
                )
                wandb.log(wandb_logs)

                # Print the log for verbosity
                for metric_name, value in wandb_logs.items():
                    print(f"- {metric_name}: {value:.4f}")

            # Save checkpoint if it is the checkpoint step
            if epoch % self.checkpoint_steps == 0:
                self.save_checkpoint(epoch, wandb_logs)

    def save_checkpoint(self, epoch: int, metrics: dict) -> None:
        """
        Save checkpoints of all training components

        Args:
            epoch (int): Current epoch
            metrics (dict): Current metrics achieved by the model
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        checkpoint.update(metrics)
        torch.save(checkpoint, f"{self.checkpoint_path}_{epoch}.pt")

    def load_checkpoint(self, model_state_dict_path: str) -> None:
        """
        Load a training checkpoint

        Args:
            model_state_dict_path (str): Path to the pretrained model file
        """
        self.model.load_state_dict(
            torch.load(model_state_dict_path, self.device)["model_state_dict"]
        )
        self.optimizer.load_state_dict(
            torch.load(model_state_dict_path, self.device)["optimizer_state_dict"]
        )
