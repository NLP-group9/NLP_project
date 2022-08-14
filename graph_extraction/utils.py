"""This module is for utils of graph extraction."""
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Tuple

import numpy as np
import torch
from omegaconf import DictConfig
from torch_geometric.data import Data
from tqdm import tqdm


def collect_token_for_dataset(dataset_path: Path, node_mapping: DefaultDict) -> DefaultDict:
    """Collect tokens in one document.

    Args:
        dataset_path: dataset to be collected.
        node_mapping: current vocabulary.

    Returns:
        node_mapping: updated vocabulary.
    """
    for file in tqdm(list(dataset_path.rglob("doc*.txt"))):
        with open(file) as doc:
            doc_str = ""
            for line in doc.readlines():
                doc_str += line.lower()
            doc_str = " ".join(doc_str.split("\n"))
            doc_str_list = doc_str.split("\n")
            for word in doc_str_list:
                node_mapping[word] += 1
    return node_mapping


def clean_node_mapping(args: DictConfig, node_mapping: DefaultDict) -> DefaultDict:
    """Clean generated node mapping, remove word with frequency < threshold.

    Args:
        args: arguments in extract.yaml
        node_mapping: current vocabulary.

    Returns:
        updated_node_mapping: updated vocabulary, a list containing tokens.
    """
    # filter vocab with freq > threshold
    sorted_node_mapping = np.array(sorted(node_mapping.items(), key=lambda x: x[1], reverse=True))
    preserve_idx = np.where(sorted_node_mapping[:, 1].astype(np.int32) > args.lowest_freq)
    token_list = sorted_node_mapping[:, 0][preserve_idx]
    updated_node_mapping: DefaultDict = defaultdict(int)
    # update node mappings
    # start with 1.
    for i, token in enumerate(token_list):
        updated_node_mapping[token] = i + 1
    return updated_node_mapping


def get_node_mapping(args: DictConfig) -> DefaultDict:
    """This function aims to generate node mappings for dataset.

    Args:
        args: arguments in extract.yaml

    Returns:
        node_mapping: {str: int}
    """
    data_root = Path(args.data.store_path)
    split_names = ["train", "test", "val"]
    node_mapping: DefaultDict = defaultdict(int)
    for split_name in split_names:
        dataset_path = data_root.joinpath(split_name)
        node_mapping = collect_token_for_dataset(dataset_path, node_mapping)
    updated_node_mapping = clean_node_mapping(args.extract, node_mapping)

    # add UNK to the vocab, UNK will be automatically assign with 0.
    return updated_node_mapping


def word2idx(directory: Path, node_mapping: DefaultDict) -> Tuple[np.ndarray, np.ndarray]:
    """Extract document and summary as index.

    Args:
        directory: directory path of one doc-sum pair.
        node_mapping: vocab mapping str to idx.

    Returns:
        doc: converted document idxes.
        summ: converted summ idxes.
    """
    doc = []
    summ = []
    for file in directory.iterdir():
        if "doc" in str(file):
            with open(file) as document:
                doc_str = ""
                for line in document.readlines():
                    doc_str += line
                doc = np.array([int(node_mapping[word]) for word in doc_str.split(" ")])
        if "sum" in str(file):
            with open(file) as summary:
                sum_str = ""
                for line in summary.readlines():
                    sum_str += line
                summ = np.array([int(node_mapping[word]) for word in sum_str.split(" ")])
    return doc, summ


def get_entity(doc: np.ndarray):
    """Get unique words from the document.

    Args:
        doc: document idxes.
    """
    return np.unique(doc)


def get_relations(doc: np.ndarray) -> np.ndarray:
    """Create bigram edge.

    Args:
        doc: document indexes.

    Returns:
        edge_index: edge index created by bigram.
    """
    # create node index mapping.
    node_idx_mapping: DefaultDict = defaultdict(int)
    for i in range(doc.shape[0]):
        node_idx_mapping[doc[i]] = i
    bigrams = []
    edge_index = []
    for i in range(len(doc) - 1):
        pair = [doc[i], doc[i + 1]]
        # create edge between bigrams
        if pair not in bigrams:
            bigrams.append(pair)
        edge_index.append([node_idx_mapping[doc[i]], node_idx_mapping[doc[i + 1]]])
    return np.array(edge_index).T


def build_graphs(args: DictConfig, split_name: str, data_root: Path, node_mapping: DefaultDict):
    """This function aims to generate pt graph.

    Args:
        args: arguments in extract.yaml
        split_name: data set name ["train", "test", "val"]
        data_root: data set store path.
        node_mapping: vocab mapping str to idx
    """
    dataset_path = data_root.joinpath(split_name)
    out_path_dataset = Path(args.PyG_path).joinpath(split_name)
    if not out_path_dataset.exists():
        out_path_dataset.mkdir()
    for directory in tqdm(list(dataset_path.rglob("text*"))):
        out_path_folder = Path(args.PyG_path).joinpath(split_name, directory.stem)
        if not out_path_folder.exists():
            out_path_folder.mkdir()
        out_path = out_path_folder / f"{directory.stem}.pt"
        doc, summ = word2idx(directory, node_mapping)
        edge_index = get_relations(doc)
        data = Data(
            x=torch.from_numpy(doc),
            edge_index=torch.from_numpy(edge_index),
            y=torch.from_numpy(summ),
            file_name=str(directory.stem),
        )
        torch.save(data, out_path)
