import pathlib as pl
from functools import lru_cache
from typing import Union

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from src.data.dataset import MultiTS


@lru_cache(maxsize=1024)
def _find_equal_target(x, target_lists):
    uuid_ = x.split('/')[-1].split('.npz')[0]
    target_path = list(filter(lambda x: x.find(uuid_) != -1, target_lists))
    assert len(target_path) == 1
    return x, target_path[0]


@lru_cache(maxsize=10)
def _get_tuples(ds_dir):
    df = pd.read_csv(ds_dir)
    return list(zip(
            df['data'].tolist(), df['target'].tolist()
        ))


class MultiTSDataset(Dataset):
    """MultiTS dataset."""

    def __init__(self, ds: Union[str, pd.DataFrame], mode='gru', use_rel=False):
        self.ds_dir = ds
        if isinstance(ds, pd.DataFrame):
            self.combined_paths = list(zip(
                ds['features_path'].tolist(),
                ds['no2_path'].tolist()
            ))
        else:
            self.combined_paths = _get_tuples(ds)
        self.mode = mode
        self.use_rel = use_rel

    def __len__(self):
        return len(self.combined_paths)

    @lru_cache(maxsize=1024)
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        ts_path, target_path = self.combined_paths[idx]
        if self.use_rel:
            ts_path = str(pl.Path(ts_path).absolute()) \
                .replace('../../', '')
            target_path = str(pl.Path(target_path).absolute()) \
                .replace('../../', '')
        data = MultiTS.load_multits(ts_path).series
        if self.mode == 'fc':
            data = data.flatten()
        target = np.load(target_path, allow_pickle=True)['no2']
        return data.astype(np.float32), target.astype(np.float32)


if __name__=="__main__":
    ds = MultiTSDataset('../../data/processed/dataset2')
    for i, sample in enumerate(ds):
        print(sample[0].shape, sample[1].shape)
        break
