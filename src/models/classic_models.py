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
from xgboost import XGBRegressor

from src.models.dataset import MultiTSDataset

logging.basicConfig(datefmt='%d-%b-%y %H:%M:%S', level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

np.random.seed(seed=42)


def _load_pca(pca_path: str) -> PCA:
    return load(pca_path)


def load_datasets(ds_path: str, pca_path: str) -> List[DataLoader]:

    logging.info(f'Getting from {ds_path=} e {pca_path=}.')

    df = pd.read_csv(f'{ds_path}/mapped_paths.csv')

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

    train_ds = _extract_feat_n_targets(df_train, 'train')
    val_ds = _extract_feat_n_targets(df_val, 'val')
    test_ds = _extract_feat_n_targets(df_test, 'test')

    logging.info('All datasets loaded correctly.')

    return train_ds, val_ds, test_ds


@dataclass
class BaseTrainer:

    experiment_name: str = None
    estimator: Any | None = None
    grid: Dict[str, Union[float, int]] = None
    train_ds: Tuple[np.ndarray, np.ndarray] = None
    test_ds: Tuple[np.ndarray, np.ndarray] = None
    out_path: str = None

    def train_model(self, params: dict):
        logging.info(f'Starting to fit model with {params=}')

        model = self.estimator(**params)

        start = time.time()
        model.fit(self.train_ds[0], self.train_ds[1])
        elapsed = time.time() - start
        logging.info(f'Model fitted in {elapsed:.4f}s. Going to metrics step')

        preds_train = model.predict(self.train_ds[0])

        preds = model.predict(self.test_ds[0])

        return model, preds, preds_train

    def calculate_metrics(
        self,
        preds: np.ndarray,
        trues: np.ndarray,
        mode = 'train'
    ) -> dict:
        r2 = r2_score(trues, preds)
        mse = mean_squared_error(trues, preds)
        rmse = mean_squared_error(trues, preds, squared=False)
        mae = mean_absolute_error(trues, preds)
        r_per, p_val = pearsonr(trues, preds)

        logging.info(f'Metrics: r2={r2:.4f} mse={mse:.4f} rmse={rmse:.4f}'
                     f' mae={mae:.4f} pearsonr={r_per:.4f}')

        return {
            f'r2_{mode}': r2,
            f'mse_{mode}': mse,
            f'rmse_{mode}': rmse,
            f'mae_{mode}': mae,
            f'r_pearson_{mode}': r_per 
        }

    def grid_search(self):

        logging.info(f'Starting grid search...')

        exp = mlflow.set_experiment(experiment_name=self.experiment_name)

        best_model = {
            'model': None,
            'r2_test': -9999
        }

        for params in ParameterGrid(self.grid):

            with mlflow.start_run(experiment_id=exp.experiment_id):
                for k, v in params.items():
                    mlflow.log_param(k, v)

                trained_model, preds, preds_train = self.train_model(params=params)

                metrics_train = self.calculate_metrics(
                    preds=preds_train,
                    trues=self.train_ds[1],
                    mode='train'
                )

                metrics_test = self.calculate_metrics(
                    preds=preds,
                    trues=self.test_ds[1],
                    mode='test'
                )

                for k, v in metrics_train.items():
                    mlflow.log_metric(k, v)

                for k, v in metrics_test.items():
                    mlflow.log_metric(k, v)

                if metrics_test['r2_test'] > best_model['r2_test']:
                    best_model['model'] = trained_model
                    best_model['r2_test'] = metrics_test['r2_test']

                    pl.Path(self.out_path).mkdir(parents=True, exist_ok=True)
                    files = glob.glob(f'{self.out_path}/*.joblib')
                    for f in files:
                        os.remove(f)
                    dump(
                        best_model['model'],
                        f'{self.out_path}/rf_r2test_{best_model["r2_test"]:.4f}.joblib'
                    )


def fit_classic_models(ds_path: str, pca_path: str, out_path: str):

    logging.info('Starting fit classic models script.')

    train_ds, val_ds, test_ds = load_datasets(ds_path, pca_path)

    # rf_trainer = BaseTrainer(
    #     experiment_name='Random Forest - v1',
    #     estimator=RandomForestRegressor,
    #     train_ds=train_ds,
    #     test_ds=test_ds,
    #     grid={
    #         'bootstrap': [True, False],
    #         'max_features': [None, 'sqrt'],
    #         'n_estimators': [100, 200, 300, 400, 500],
    #         'max_depth': [20, 40, 60, 80, 100, None],
            
    #     },
    #     out_path=f'{out_path}/rf'
    # )

    # rf_trainer.grid_search()

    xgb_trainer = BaseTrainer(
        experiment_name='XGBoost',
        estimator=XGBRegressor,
        train_ds=train_ds,
        test_ds=test_ds,
        grid={
            'eta': [0.5, 0.1, 0.05, 0.05, 0.005, 0.001],
            'subsample': [0.7, 0.9, 1.0],
            'n_estimators': [200, 400, 600, 800, 1000],
            'max_depth': [20, 40, 60, 80, 100, None],
            'colsample_bytree': [0.8, 0.9, 1.0]
            
        },
        out_path=f'{out_path}/xgboost'
    )

    xgb_trainer.grid_search()

    # ada_trainer = BaseTrainer(
    #     experiment_name='AdaBoost',
    #     estimator=AdaBoostRegressor,
    #     train_ds=train_ds,
    #     test_ds=test_ds,
    #     grid={
    #         'loss': ['linear', 'square', 'exponential'],
    #         'n_estimators': [100, 200, 300, 400, 500],
    #         'learning_rate': [1, 5e-1, 1e-1, 5e-2, 1e-2, 5e-3, 1e-3],
    #     },
    #     out_path=f'{out_path}/adaboost'
    # )

    # ada_trainer.grid_search()
 
    # lasso_trainer = BaseTrainer(
    #     experiment_name='Lasso',
    #     estimator=Lasso,
    #     train_ds=train_ds,
    #     test_ds=test_ds,
    #     grid={
    #         'alpha': [0.01, 0.05, 0.1, 0.5, 1.0],
    #         'max_iter': [500, 1000, 1500, 2000],
    #         'warm_start': [False, True]
    #     },
    #     out_path=f'{out_path}/lasso'
    # )

    # lasso_trainer.grid_search()

    # lgbm_trainer = BaseTrainer(
    #     experiment_name='LightGBM',
    #     estimator=LGBMRegressor,
    #     train_ds=train_ds,
    #     test_ds=test_ds,
    #     grid={
    #         'boosting_type': ['rf', 'dart'],
    #         'learning_rate': [1, 5e-1, 1e-1, 5e-2, 1e-2, 5e-3, 1e-3],
    #         'n_estimators': [100, 300, 500, 700, 800, 1000],
    #         'num_leaves': [31, 50, 70, 100],
    #         'bagging_fraction': [0.1, 0.2, 0.3],
    #         'bagging_freq': [5, 10, 15]
    #     },
    #     out_path=f'{out_path}/lgbm'
    # )

    # lgbm_trainer.grid_search()
