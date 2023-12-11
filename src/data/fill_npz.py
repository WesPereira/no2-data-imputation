import shutil
import pathlib as pl
from joblib import dump, load
import logging
import random

import fire
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from dataset import MultiTS


logging.basicConfig(datefmt='%d-%b-%y %H:%M:%S', level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
np.random.seed(42)


def main(
    mapped_path: str,
    artifacts_path: str,
    method: str = 'mean',
    norm_no2 = False,
    normalizer = None
):
    df = pd.read_csv(mapped_path)

    logging.info(f'Found {df.shape[0]} paths.')

    logging.info(f'Starting normalization model fitting step')

    all_series = []
    for p in df['features_path'].tolist():
        ts = MultiTS.load_multits(p)
        serie = ts.series
        all_series.append(serie)

    complete_serie = np.concatenate(all_series)
    if normalizer:
        scaler = load(normalizer)
        logging.info('Logged scaler!')
    else:
        scaler = StandardScaler()
        scaler.fit(complete_serie)

    logging.info('Finished to fit scaler')

    dataset_name = mapped_path.split('/')[-2]

    for p in df['features_path'].tolist():
        ts = MultiTS.load_multits(p)
        serie = ts.series
        if method == 'mean':
            col_mean = np.nanmean(serie, axis=0)
            inds = np.where(np.isnan(serie))
            serie[inds] = np.take(col_mean, inds[1])
        scaled_serie = scaler.transform(serie)
        ts_filled = MultiTS(
            series=scaled_serie,
            dse=ts.dse,
            masks=ts.masks
        )
        p_tosave = p.replace(f'interim/{dataset_name}',
                             f'processed/{dataset_name}/data')
        ppl = pl.Path(p_tosave)
        ppl.parents[0].mkdir(parents=True, exist_ok=True)
        
        ts_filled.save_numpy(str(p_tosave))

    logging.info('Finished to fill NaNs. Starting to mount targets.')

    if norm_no2:
        all_no2 = []
        for p in df['no2_path'].tolist():
            no2_d = np.load(p, allow_pickle=True)['no2']
            all_no2.append(no2_d)
        scaler_no2 = StandardScaler()
        scaler_no2.fit(np.array(all_no2).reshape(-1, 1))

    for p in df['no2_path'].tolist():
        p_tosave = p.replace(f'interim/{dataset_name}',
                             f'processed/{dataset_name}/target')
        ppl = pl.Path(p_tosave)
        ppl.parents[0].mkdir(parents=True, exist_ok=True)
        if norm_no2:
            no2_d = np.load(p, allow_pickle=True)['no2']
            no2_data = scaler_no2.transform(no2_d.reshape(1, -1))
            np.savez(p_tosave, no2=no2_data.squeeze(-1))
        else:
            shutil.copy2(p, p_tosave)

    df['features_path'] = df['features_path'].apply(lambda x: x.replace(
        f'interim/{dataset_name}',
        f'processed/{dataset_name}/data'
    ))

    df['no2_path'] = df['no2_path'].apply(lambda x: x.replace(
        f'interim/{dataset_name}',
        f'processed/{dataset_name}/target'
    ))

    def _map_date_to_ds_type_yearly(_date: str) -> str:
        year = int(_date.split('-')[0])
        match year:
            case 2022:
                return 'current'
            case 2021:
                return 'val'
            case 2019 | 2020:
                return 'val'
            case _:
                return 'others'

    def _map_date_to_ds_type_randomly(_date: str) -> str:
        value = np.random.random()
        if value <= 0.15:
            return 'test'
        elif value <= 0.3:
            return 'val'
        else:
            return 'train'

    df['ds_type'] = df['date'].apply(_map_date_to_ds_type_yearly)
    #df['ds_type'] = df['date'].apply(_map_date_to_ds_type_randomly)

    df.to_csv(mapped_path.replace('interim', 'processed'), index=False)
    pl.Path(artifacts_path).mkdir(parents=True, exist_ok=True)
    if norm_no2:
        dump(scaler_no2, f'{artifacts_path}/standard_scaler_no2.joblib')
    dump(scaler, f'{artifacts_path}/standard_scaler.joblib')
    logging.info('Finished to move targets to processed dir.')


if __name__=="__main__":
    fire.Fire(main)
