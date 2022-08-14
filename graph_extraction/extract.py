"""This module aims to extract graph from text files."""
import json
import shutil
from pathlib import Path

import hydra
from omegaconf import DictConfig
from utils import build_graphs, get_node_mapping


@hydra.main(config_path="../conf", config_name="config")
def extract(args: DictConfig):
    """Extract graph as pt files."""
    # configuration settings
    args_data = args.data
    args_extract = args.extract

    # generate mappings for node
    print("Start generating node mappings.")
    node_mapping = get_node_mapping(args)

    if not Path(args_extract.PyG_path).exists():
        Path(args_extract.PyG_path).mkdir()
    # save node_mapping
    with open(Path(args_extract.PyG_path) / "atom_mapping.json", "w") as file:
        json.dump(node_mapping, file)

    # generate pt graph
    split_names = ["train", "test", "val"]
    data_root = Path(args_data.store_path)

    if Path(args_extract.PyG_path).exists():
        print("PyG has been generated!")
        if not args_extract.overwrite:
            return None
        else:
            print("Delete generated PyG.")
            shutil.rmtree(args_extract.PyG_path)
            print("Regenerate PyG.")
            Path(args_extract.PyG_path).mkdir()
    else:
        print("Start generating PyG.")
    for split_name in split_names:
        build_graphs(args_extract, split_name, data_root, node_mapping)


if __name__ == "__main__":
    extract()
