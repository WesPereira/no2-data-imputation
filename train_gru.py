import gc

import fire
import torch
import mlflow
import pandas as pd
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from scipy.stats import pearsonr
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from src.models.dataset import MultiTSDataset
from src.models.trainer import ModelsTrainer
from src.infer import infer


DS_PATH = "/Users/wesleypereira/Documents/tcc/exploration/gee_exploration/data/processed/gee_ds_20231003_formatted"
ARTIFACTS_PATH = 'artifacts/models_05112023/conv1d'
EXPERIMENT_NAME = 'Conv1d'


sharing_strategy = "file_system"
torch.multiprocessing.set_sharing_strategy(sharing_strategy)

def set_worker_sharing_strategy(worker_id: int) -> None:
    torch.multiprocessing.set_sharing_strategy(sharing_strategy)


def _task(config, train_ds, val_ds, device):

    exp = mlflow.set_experiment(experiment_name=EXPERIMENT_NAME)

    with mlflow.start_run(experiment_id=exp.experiment_id):
        for k, v in config.items():
            mlflow.log_param(k, v)

        train_loader = DataLoader(
            train_ds,
            batch_size=config['batch_size'],
            shuffle=True, num_workers=4,
            worker_init_fn=set_worker_sharing_strategy
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=config['batch_size'],
            num_workers=4,
            worker_init_fn=set_worker_sharing_strategy
        )

        model = ModelsTrainer(
            config, device, artifacts_path=ARTIFACTS_PATH, model_type=EXPERIMENT_NAME
        )
        trainer = pl.Trainer(logger=TensorBoardLogger(
                f'{ARTIFACTS_PATH}/tbl',
                name='cls',
            ), accelerator=str(device))
        trainer.fit(
            model=model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader
        )

        mdl_ckpt = trainer.checkpoint_callback.best_model_path

        lr = config['lr']
        bs = config['batch_size']
        #hd = config['hidden_dim']
        #nl = config['n_layers']
        if EXPERIMENT_NAME in ['gru', 'lstm', 'LSTM - v1']:
            dir_path = (f"lr{config['lr']}_bs"
                        f"{config['batch_size']}_hd{config['hidden_dim']}"
                        f"_nl{config['n_layers']}")
        elif EXPERIMENT_NAME in ('ConvLSTM'):
            con = config['convs']
            dir_path = (f"lr{config['lr']}_bs"
                        f"{config['batch_size']}_kz_{config['kernel_size']}_"
                        f"nl{config['n_layers']}_hd{config['hidden_dim']}_"
                        f"convs{con[0]}_{con[1]}_{con[2]}")
        else:
            lin = config["linears"]
            con = config["convs"]
            dir_path = (f"lr{config['lr']}_bs"
                        f"{config['batch_size']}_lins{lin[0]}_{lin[1]}_{lin[2]}_"
                        f"convs{con[0]}_{con[1]}_{con[2]}")

        trues, preds = infer(
            model_path=mdl_ckpt,
            test_path=DS_PATH,
            output_path=f'{ARTIFACTS_PATH}/{dir_path}/{exp.experiment_id}.csv',
            model_type=EXPERIMENT_NAME,
            config=config
        )

        r2 = r2_score(trues, preds)
        mse = mean_squared_error(trues, preds)
        rmse = mean_squared_error(trues, preds, squared=False)
        mae = mean_absolute_error(trues, preds)
        r_per, p_val = pearsonr(trues, preds)

        metrics = {
            f'r2_test': r2,
            f'mse_test': mse,
            f'rmse_test': rmse,
            f'mae_test': mae,
            f'r_pearson_test': r_per 
        }

        for k, v in metrics.items():
            mlflow.log_metric(k, v)
    #train_loader = None
    #val_loader = None
    #trainer = None


def _get_ds(ds_path: str = DS_PATH):
    df = pd.read_csv(f'{ds_path}/mapped_paths.csv')

    df_train = df[df['ds_type'] == 'train']
    train_ds = MultiTSDataset(df_train, use_rel=True)
    df_val = df[df['ds_type'] == 'val']
    val_ds = MultiTSDataset(df_val, use_rel=True)

    return train_ds, val_ds


class ModelsGBManager:

    @classmethod
    def lstm_n_gru(cls, lr: float, batch_size: int, hidden_dim: int, nl: int):
        device = torch.device('cpu')

        config = {
            "lr": float(lr),
            "batch_size": int(batch_size),
            "hidden_dim": int(hidden_dim),
            "n_layers": int(nl),
        }

        train_ds, val_ds = _get_ds()

        _task(config, train_ds, val_ds, device)

    @classmethod
    def conv1d(
        cls,
        lr: float,
        batch_size: int,
        convs: int,
        kernel_size: int
    ):
        device = torch.device('cpu')

        lin_dim = int((365 - int(kernel_size - 1))/2)
        lin_dim = int((lin_dim - int(kernel_size - 1))/2)
    
        config = {
            "lr": float(lr),
            "batch_size": int(batch_size),
            "convs": convs,
            "linears": [lin_dim, 32, 1],
            "kernel_size": kernel_size
        }

        train_ds, val_ds = _get_ds()

        _task(config, train_ds, val_ds, device)

    @classmethod
    def convlstm(
        cls,
        lr: float,
        batch_size: int,
        convs: int,
        kernel_size: int,
        hidden_dim: int,
        nl: int
    ):
        device = torch.device('cpu')
    
        config = {
            "lr": float(lr),
            "batch_size": int(batch_size),
            "convs": convs,
            "kernel_size": kernel_size,
            "hidden_dim": hidden_dim,
            "n_layers": nl
        }

        train_ds, val_ds = _get_ds()

        _task(config, train_ds, val_ds, device)


if __name__=="__main__":
    fire.Fire(ModelsGBManager)
    #device_name = 'mps' if torch.backends.mps.is_available() else 'cpu'
    # device = torch.device('cpu')
    # configs_choices = {
    #     "lr": [1e-1, 2e-1, 5e-2, 1e-2],
    #     "batch_size": [32, 64, 128],
    #     "hidden_dim": [8, 16, 32, 64],
    #     "n_layers": [2, 1],
    # }
    # train_ds = MultiTSDataset(f'{DS_PATH}/train_df_mapping.csv')
    # val_ds = MultiTSDataset(f'{DS_PATH}/val_df_mapping.csv')
    # for config in list(ParameterGrid(configs_choices)):
    #     with ClearCache():
    #         main(config, train_ds, val_ds, device)
