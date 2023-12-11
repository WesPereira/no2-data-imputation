import glob
import uuid
from dataclasses import dataclass
from functools import lru_cache

import numpy as np


@dataclass
class MultiTS:

    series: np.ndarray
    dse: np.ndarray = None
    masks: np.ndarray = None

    def save_numpy(self, path: str):
        data = {
            'series': self.series,
            'dse': self.dse,
            'masks': self.masks
        }
        if path.find('.npz') != -1:
            np.savez(
                path,
                series= self.series,
                dse= self.dse,
                masks= self.masks
            )
            return None
        else:
            uuid_ = uuid.uuid4()
            np.savez(
                f'{path}/{uuid_}.npz',
                series= self.series,
                dse= self.dse,
                masks= self.masks
            )
            return uuid_

    @classmethod
    @lru_cache(maxsize=4096)
    def load_multits(cls, npz_path: str):
        data = np.load(npz_path, allow_pickle=True)
        return cls(**data)


if __name__=="__main__":
    # ts = MultiTS(
    #     series=np.array([1, 2, 3, 4], dtype=np.int32),
    #     dse=np.array([1001, 1002, 1003, 1004], dtype=np.int32),
    #     masks=np.array([1, 0, 0, 1], dtype=bool)
    # )

    # ts.save_numpy('.')

    path_ = '/Users/wesleypereira/Documents/tcc/exploration/gee_exploration/data/interim/dataset1/0101000000A4A2486DED5649C080FFEBD61C381FC0/0a5af334-f895-4015-aed6-b0f504aa1e93.npz'
    ts2 = MultiTS.load_multits(path_)
    print(ts2)
