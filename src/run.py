import os
import numpy as np
import uuid

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelSummary, ModelCheckpoint

from src.utils.utils import use_best_hyperparams, get_available_accelerator
from src.datasets.data_loading import get_dataset, get_dataset_split
from src.datasets.dataset import FullBatchGraphDataset
from src.model import get_model, LightingFullBatchModelWrapper
from src.utils.arguments import args


def run(args):
    torch.manual_seed(0)

    # Get dataset and dataloader
    dataset, evaluator = get_dataset(
        name=args.dataset,
        root_dir=args.dataset_directory,
        undirected=args.undirected,
        self_loops=args.self_loops,
        transpose=args.transpose,
    )
    data = dataset._data
    data_loader = DataLoader(FullBatchGraphDataset(data), batch_size=1, collate_fn=lambda batch: batch[0])

    val_accs, test_accs = [], []
    for num_run in range(args.num_runs):
        # Get train/val/test splits for the current run
        train_mask, val_mask, test_mask = get_dataset_split(args.dataset, data, args.dataset_directory, num_run)

        # Get model
        args.num_features, args.num_classes = data.num_features, dataset.num_classes
        model = get_model(args)
        lit_model = LightingFullBatchModelWrapper(
            model=model,
            lr=args.lr,
            weight_decay=args.weight_decay,
            evaluator=evaluator,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
        )

        # Setup Pytorch Lighting Callbacks
        early_stopping_callback = EarlyStopping(monitor="val_acc", mode="max", patience=args.patience)
        model_summary_callback = ModelSummary(max_depth=-1)
        if not os.path.exists(f"{args.checkpoint_directory}/"):
            os.mkdir(f"{args.checkpoint_directory}/")
        model_checkpoint_callback = ModelCheckpoint(
            monitor="val_acc",
            mode="max",
            dirpath=f"{args.checkpoint_directory}/{str(uuid.uuid4())}/",
        )

        # Setup Pytorch Lighting Trainer
        trainer = pl.Trainer(
            log_every_n_steps=1,
            max_epochs=args.num_epochs,
            callbacks=[
                early_stopping_callback,
                model_summary_callback,
                model_checkpoint_callback,
            ],
            profiler="simple" if args.profiler else None,
            accelerator=get_available_accelerator(),
            devices=[args.gpu_idx],
        )

        # Fit the model
        trainer.fit(model=lit_model, train_dataloaders=data_loader)

        # Compute validation and test accuracy
        val_acc = model_checkpoint_callback.best_model_score.item()
        test_acc = trainer.test(ckpt_path="best", dataloaders=data_loader)[0]["test_acc"]
        test_accs.append(test_acc)
        val_accs.append(val_acc)

    print(f"Test Acc: {np.mean(test_accs)} +- {np.std(test_accs)}")


if __name__ == "__main__":
    args = use_best_hyperparams(args, args.dataset) if args.use_best_hyperparams else args
    run(args)
