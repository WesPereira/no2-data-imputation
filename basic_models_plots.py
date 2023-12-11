import os
import time
import logging
import glob
from joblib import load, dump
from typing import List, Any, Dict, Union, Tuple
from dataclasses import dataclass
import pathlib as pl

import mlflow
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import Lasso
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.decomposition import PCA
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from torch.utils.data import DataLoader

from src.models.dataset import MultiTSDataset
from src.metrics_and_plots import get_metrics_n_plots

logging.basicConfig(datefmt='%d-%b-%y %H:%M:%S', level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

np.random.seed(seed=42)

ds_path = 'data/processed/gee_ds_20231204_2'
pca_path = 'artifacts/121023/pca_550_components.joblib'


def _load_pca(pca_path: str) -> PCA:
    return load(pca_path)


def load_datasets(
    ds_path: str,
    pca_path: str,
    get_all: bool = False
) -> List[DataLoader]:

    logging.info(f'Getting from {ds_path=} e {pca_path=}.')

    df = pd.read_csv(f'{ds_path}/mapped_paths.csv')

    if not get_all:
        df_train = df[df['ds_type'] == 'train']
        df_val = df[df['ds_type'] == 'val']
        df_test = df[df['ds_type'] == 'test']

    pca_model = _load_pca(pca_path)

    logging.info('PCA loaded.')

    def _extract_feat_n_targets(base_df, type: str):

        logging.info(f'Extracting {type} dataset.')
        ds = MultiTSDataset(base_df, use_rel=True)

        loader = DataLoader(ds, batch_size=1)

        x_data = []
        target_data = []
        for x, y in loader:
            x_data.append(x.flatten().tolist())
            target_data.append(y.flatten().tolist())

        features = np.array(x_data)
        target = np.array(target_data).flatten()
        logging.info(f'Features: {features.shape}, Target {target.shape}')

        reducted_features = pca_model.transform(features)
        logging.info(f'Features reducted: {reducted_features.shape}, '
                     f'Target {target.shape}')

        return reducted_features, target

    if not get_all:
        train_ds = _extract_feat_n_targets(df_train, 'train')
        val_ds = _extract_feat_n_targets(df_val, 'val')
        test_ds = _extract_feat_n_targets(df_test, 'test')

        logging.info('All datasets loaded correctly.')

        return train_ds, val_ds, test_ds
    else:
        all_ds = _extract_feat_n_targets(df, 'all')

        logging.info('All datasets loaded correctly.')

        return all_ds, all_ds, all_ds


if __name__=="__main__":
    train_ds, val_ds, test_ds = load_datasets(ds_path, pca_path, get_all=True)
    model = load('artifacts/models_05112023/rf/rf_r2test_0.4255.joblib')
    yps = model.predict(test_ds[0])
    yts = test_ds[1]

    get_metrics_n_plots('', 'plots/final_val_amazon/rf', yts, yps)
