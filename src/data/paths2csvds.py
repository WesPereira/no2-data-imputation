import glob
import logging
from typing import List, Tuple

import numpy as np
import pandas as pd


DS_PATH = ("/Users/wesleypereira/Documents/tcc/exploration/"
           "gee_exploration/data/processed/dataset_max")


logging.basicConfig(datefmt='%d-%b-%y %H:%M:%S', level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def _find_equal_target(x, datas_lists):
    modified_data = x.replace('/target/', '/data/').replace('_no2', '')
    target_path = list(filter(lambda x: modified_data == x, datas_lists))
    assert len(target_path) == 1
    return target_path[0], x


def split_array(
    array: List[Tuple[str, str]],
    train_size: float,
    val_size: float,
    test_size: float
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    assert train_size + val_size + test_size == 1.0

    len_array = len(array)

    np.random.seed(42)

    np.random.shuffle(array)

    train_ds = array[0: int(train_size*len_array)]
    val_beg = int(train_size*len_array)
    val_end = int(train_size*len_array) + int(val_size*len_array)
    val_ds = array[val_beg:val_end]
    test_ds = array[val_end:]

    train_df = pd.DataFrame(train_ds, columns=['data', 'target'])
    val_df = pd.DataFrame(val_ds, columns=['data', 'target'])
    test_df = pd.DataFrame(test_ds, columns=['data', 'target'])

    return train_df, val_df, test_df


def main():
    data_paths = glob.glob(f'{DS_PATH}/data/*/*.npz')
    target_paths = glob.glob(f'{DS_PATH}/target/*/*.npz')

    logging.info(f'Found {len(data_paths)} data paths and '
                 f'{len(target_paths)} target paths')

    combined_paths = list(map(
        lambda x: _find_equal_target(x, tuple(data_paths)), target_paths
    ))

    logging.info('Finished to combine paths and targets. Saving...')

    train_df, val_df, test_df = split_array(
        combined_paths, 0.7, 0.15, 0.15
    )

    logging.info(f'Train df len: {train_df.shape}')
    logging.info(f'Train df len: {val_df.shape}')
    logging.info(f'Train df len: {test_df.shape}')

    train_df.to_csv(f'{DS_PATH}/train_df_mapping.csv', index=False)
    val_df.to_csv(f'{DS_PATH}/val_df_mapping.csv', index=False)
    test_df.to_csv(f'{DS_PATH}/test_df_mapping.csv', index=False)

    logging.info('Fertig.')


if __name__=="__main__":
    main()
