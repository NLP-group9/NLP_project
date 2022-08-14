"""This module aims to generate data and clean data."""

import string
from pathlib import Path
from shutil import rmtree
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
    punctuation_free = " ".join(punctuation_free.split("\n"))
    return punctuation_free


def store_data_as_txt(args: DictConfig, split_name: str, dataset: Dataset, store_path: Path):
    """Store dataset as txt files in directories.

    Args:
        args: arguments of data_gen
        split_name: data set split name ["train", "test", "val"].
        dataset: huggingface dataset object.
        store_path: directory to store the text files.
    """
    # create split directory
    cur_path = store_path.joinpath(split_name)
    if cur_path.exists():
        print(f"{split_name} dataset has been generated!")
        if args.overwrite is False:
            return None
        else:
            print("Delete generated dataset")
            rmtree(cur_path)
    cur_path.mkdir()
    print(f"Start generating {split_name} dataset!")
    num_all = 0
    num_select = 0
    for i, data in enumerate(tqdm(dataset)):
        num_all += 1
        # remove doc > max_len and doc < min_len.
        doc = remove_punctuation(data["document"])
        summ = remove_punctuation(data["summary"])
        length_doc = len(" ".join(doc.split("\n")).split(" "))
        if length_doc > args.max_length or length_doc < args.min_length:
            continue
        # create a directory for each pair texts.
        text_dir = cur_path.joinpath(f"text_{i}")
        if text_dir.exists():
            continue
        text_dir.mkdir()
        # write document
        document_path = text_dir.joinpath(f"doc_{i}.txt")
        with open(document_path, "w") as document:
            document.write(doc)
        # write summary
        summary_path = text_dir.joinpath(f"sum_{i}.txt")
        with open(summary_path, "w") as summary:
            summary.write(summ)
        num_select += 1

    print(
        f"Finished generating {split_name} dataset, with {num_select} out of {num_all}. "
        f"Extract rate:{num_select/num_all}"
    )

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
    store_path = Path(args.data.store_path)
    if not store_path.exists():
        store_path.mkdir()
    split_names = ["train", "val", "test"]
    datasets_list = [train_dataset, val_dataset, test_dataset]
    _ = [
        store_data_as_txt(args.data, split_names[i], datasets_list[i], store_path)
        for i in range(len(split_names))
    ]


if __name__ == "__main__":
    main()
