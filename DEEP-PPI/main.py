import argparse

import pandas as pd
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from data import CustomStarDataset
from model import BaseLineModel


def read_cmd_line_args():
    parser = argparse.ArgumentParser(
        description="protein-protein interaction prediction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-d", "--data", help="path to dataframe", default="protein_data.csv"
    )
    parser.add_argument(
        "-m", "--model", help="model name", default="facebook/esm2_t6_8M_UR50D"
    )
    parser.add_argument("-e", "--epochs", help="number of epochs", default=5)
    parser.add_argument("-b", "--batch_size", help="batch size", default=8)
    parser.add_argument(
        "-acc",
        "--accelerator",
        help="used accelerator, can be gpu, cou or mps",
        default="gpu",
    )
    args = parser.parse_args()
    config = vars(args)
    return config


def create_dataloaders(config):
    dataframe = pd.read_csv(config["data"])
    train, valid = train_test_split(dataframe, test_size=0.2, random_state=42)

    tokenizer = AutoTokenizer.from_pretrained(config["model"])

    train_ds = CustomStarDataset(train, tokenizer)
    valid_ds = CustomStarDataset(valid, tokenizer)
    train_dl = DataLoader(
        train_ds,
        batch_size=config["batch_size"],
        num_workers=4,
        shuffle=True,
        drop_last=True,
    )
    valid_dl = DataLoader(
        valid_ds,
        batch_size=1,
        num_workers=4,
        shuffle=False,
        drop_last=True,
    )

    return train_dl, valid_dl


if __name__ == "__main__":
    config = read_cmd_line_args()
    print("Used config:")
    for key, value in config.items():
        print(key, ":", value)

    print("Loading data...")
    train_dl, valid_dl = create_dataloaders(config)
    print("Data loaded!")
    print("Creating model...")
    model = BaseLineModel(base_model=config["model"])
    print("Model created!")
    print("Training model...")
    trainer = pl.Trainer(accelerator=config["accelerator"], max_epochs=config["epochs"])
    trainer.fit(model=model, train_dataloaders=train_dl, val_dataloaders=valid_dl)
    print("Model trained!")
