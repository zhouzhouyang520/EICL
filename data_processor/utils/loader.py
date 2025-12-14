import json
import os
import pickle
from functools import lru_cache
from typing import Iterable, Tuple

from data_processor.utils.tools.time_warpper import wrapper_calc_time

DEFAULT_SPLITS: Tuple[str, ...] = ("train", "test") #"dev", 


def _dataset_dir(auxiliary_model: str, data_name: str, data_root: str) -> str:
    rel_path = f"{auxiliary_model}_auxiliary_model_data/{data_name}"
    full_path = os.path.join(data_root, rel_path)
    if not os.path.isdir(full_path):
        raise FileNotFoundError(
            f"Dataset directory not found: {full_path}. "
            f"Expected under data_root={data_root}."
        )
    return full_path


@wrapper_calc_time(print_log=True)
@lru_cache(maxsize=None)
def load_dataset(
    auxiliary_model: str,
    data_name: str,
    data_root: str = "data",
    splits: Iterable[str] = DEFAULT_SPLITS,
):
    """
    Load datasets such as train/dev/test for an auxiliary model.

    Args:
        auxiliary_model: EI, GE, etc.
        data_name: dataset folder, e.g. ED, EDOS.
        data_root: root directory that contains `{aux}_auxiliary_model_data`.
        splits: iterable of split names to load.

    Returns:
        Tuple of datasets aligned with `splits`. Missing files yield None.
    """
    dataset_dir = _dataset_dir(auxiliary_model, data_name, data_root)
    loaded = []
    for split in splits:
        file_path = os.path.join(dataset_dir, f"{split}.p")
        if os.path.exists(file_path):
            print(f"Loading {split} dataset from path: {file_path}")
            loaded.append(load_pickle(file_path))
        else:
            loaded.append(None)
    return tuple(loaded)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_data(json_data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as data_file:
        json.dump(json_data, data_file, indent=4)