"""This module aims to generate data and clean data."""

import string
from pathlib import Path
from typing import Tuple

import hydra
from datasets import Dataset, load_dataset, load_dataset_builder
from omegaconf import DictConfig
from tqdm import tqdm


def remove_punctuation(text: str) -> str:
    """Defining the function to remove punctuation.

    Args:
        text: string to process.

    Returns:
        punctuation_free: processed text.
    """
    punctuation_free = "".join([i for i in text if i not in string.punctuation])
    return punctuation_free


def store_data_as_txt(split_name: str, dataset: Dataset, store_path: Path):
    """Store dataset as txt files in directories.

    Args:
        split_name: data set split name ["train", "test", "val"].
        dataset: huggingface dataset object.
        store_path: directory to store the text files.
    """
    # create split directory
    cur_path = store_path.joinpath(split_name)
    if cur_path.exists():
        print(f"{split_name} dataset has been generated!")
        return None
    cur_path.mkdir()
    print(f"Start generating {split_name} dataset!")
    for i, data in enumerate(tqdm(dataset)):
        # create a directory for each pair texts.
        text_dir = cur_path.joinpath(f"text_{i}")
        if text_dir.exists():
            continue
        text_dir.mkdir()
        # write document
        document_path = text_dir.joinpath(f"doc_{i}.txt")
        with open(document_path, "w") as document:
            document.write(remove_punctuation(data["document"]))
        # write summary
        summary_path = text_dir.joinpath(f"sum_{i}.txt")
        with open(summary_path, "w") as document:
            document.write(remove_punctuation(data["summary"]))

        return None


def data_gen() -> Tuple[Dataset, Dataset, Dataset]:
    """Generate data."""
    (train_dataset, val_dataset, test_dataset) = (
        load_dataset("xsum", split="train"),
        load_dataset("xsum", split="test"),
        load_dataset("xsum", split="validation"),
    )
    return train_dataset, val_dataset, test_dataset


@hydra.main(config_path="../conf", config_name="config")
def main(args: DictConfig):
    """Download datasets and clean datasets."""
    ds_builder = load_dataset_builder("xsum")
    print(f"Description of xsum dataset: \n {ds_builder.info.description}")

    # generate datasets
    print("Start generating datasets!")
    train_dataset, val_dataset, test_dataset = data_gen()

    # store datasets to text files
    if not hasattr(args.data, "store_path"):
        raise ValueError("Store_path has to be defined in ../conf/data/data.yaml !")
    store_path = args.data.store_path
    split_names = ["train", "val", "test"]
    datasets_list = [train_dataset, val_dataset, test_dataset]
    _ = [
        store_data_as_txt(split_names[i], datasets_list[i], store_path)
        for i in range(len(split_names))
    ]
