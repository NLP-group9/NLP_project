"""This module aims to analysis the length of documents in datasets."""
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
from omegaconf import DictConfig
from tqdm import tqdm


def analysis(args: DictConfig, data_path: Path, split_name: str):
    """Analyse the data set, with max length, min length.

        clean data length > 1000 and < 50

    Args:
        args: data arguments.
        data_path: path of dataset
        split_name: name of data set that is processing.
    """
    length = []
    for file in tqdm(list(data_path.rglob("doc*.txt"))):
        with open(file) as doc:
            doc_str = ""
            for line in doc.readlines():
                doc_str += line
            length.append(len("".join(doc_str.split("\n")).split(" ")))

    plt.hist(length, bins=100)
    plt.xlabel("Num of words in one text")
    plt.ylabel("Num of passages")
    plt.savefig(f"{Path(args.analysis_path).joinpath(split_name)}.png", dpi=500)
    plt.show()


@hydra.main(config_path="../conf", config_name="config")
def main(args: DictConfig):
    """Data length analysis."""
    # configuration settings
    args_data = args.data
    if not Path(args_data.analysis_path).exists():
        Path(args_data.analysis_path).mkdir()

    # analysis and clean data
    split_names = ["train", "val", "test"]
    _ = [
        analysis(args_data, Path(args_data.store_path).joinpath(split_name), split_name)
        for split_name in split_names
    ]
