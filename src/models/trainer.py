from typing import Any
from collections import OrderedDict

import torch
from torch import Tensor
from torch import optim, nn
import pytorch_lightning as pl
from sklearn.metrics import r2_score

from src.models.conv1d import Conv1DModel
from src.models.conv_lstm import ConvLSTMModel
from src.models.gru import GRUModel
from src.models.lstm import LSTMModel


# define the LightningModule
class GRUTrainer(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim, output_dim, 
                 n_layers, model_type = 'gru'):
        super().__init__()
        if model_type == 'gru':
            self.model = GRUModel(input_dim, hidden_dim, n_layers, output_dim)
        else:
            self.model = LSTMModel(input_dim, hidden_dim, n_layers, output_dim)
        self.eval_loss = []
        self.eval_r2 = []
        self._loss = nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def _step_and_log(self, data, idx, mode):
        x, y = data
        y_hat = self.forward(x)
        loss = self._loss(y_hat, y)
        r2 = r2_score(y.detach().numpy().flatten(), y_hat.detach().numpy().flatten())
        self.log(f"{mode}_loss", loss,
                on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log(f"{mode}_r2", r2,
                on_step=True, on_epoch=True, prog_bar=True, logger=True)
        if mode == 'val':
            self.eval_loss.append(loss)
            self.eval_r2.append(r2)
            return {'val_loss': loss, 'val_r2': r2}
        return loss

    def training_step(self, data: Any, index: int) -> Tensor: # pylint: disable=all
        '''training step'''
        return self._step_and_log(data, index, mode='train')

    def validation_step(self, data: Any, index: int) -> Tensor: # pylint: disable=all
        '''val step'''
        return self._step_and_log(data, index, mode='val')

    def on_validation_epoch_end(self):
        avg_loss = torch.Tensor(self.eval_loss).mean()
        avg_r2 = torch.Tensor(self.eval_r2).mean()
        self.log("ptl/val_loss", avg_loss)
        self.log("ptl/val_r2", avg_r2)
        self.eval_loss.clear()
        self.eval_r2.clear()
        return {'ptl/val_loss': avg_loss, 'ptl/val_r2': avg_r2}

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-1)
        return optimizer


class ModelsTrainer(pl.LightningModule):
    def __init__(self, config, device, artifacts_path, model_type = 'gru'):
        super().__init__()
        self.artifacts_path = artifacts_path
        self.lr = config['lr']
        if model_type == 'gru':
            self.model = GRUModel(
                8,
                config['hidden_dim'],
                config['n_layers'],
                1,
                device=device
            )
            self.dir_path = (f"{self.artifacts_path}/lr{config['lr']}_bs"
                             f"{config['batch_size']}_hd{config['hidden_dim']}"
                             f"_nl{config['n_layers']}")
        elif model_type == 'Conv1d':
            self.model = Conv1DModel(
                convs=config["convs"],
                linears=config["linears"],
                kernel_size=config["kernel_size"]
            )
            lin = config['linears']
            con = config['convs']
            self.dir_path = (f"{self.artifacts_path}/lr{config['lr']}_bs"
                             f"{config['batch_size']}_lins{lin[0]}_{lin[1]}_{lin[2]}_"
                             f"convs{con[0]}_{con[1]}_{con[2]}")
        elif model_type == 'ConvLSTM':
            self.model = ConvLSTMModel(
                convs=config['convs'],
                kernel_size=config['kernel_size'],
                hidden_dim=config['hidden_dim'],
                layer_dim=config['n_layers']
            )
            con = config['convs']
            self.dir_path = (f"{self.artifacts_path}/lr{config['lr']}_bs"
                             f"{config['batch_size']}_kz_{config['kernel_size']}_"
                             f"nl{config['n_layers']}_hd{config['hidden_dim']}_"
                             f"convs{con[0]}_{con[1]}_{con[2]}")
        else:
            self.model = LSTMModel(
                8,
                config['hidden_dim'],
                config['n_layers'],
                1
            )
            self.dir_path = (f"{self.artifacts_path}/lr{config['lr']}_bs"
                             f"{config['batch_size']}_hd{config['hidden_dim']}"
                             f"_nl{config['n_layers']}")
        self.patience = 6
        self._loss = nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def _step_and_log(self, data, idx, mode):
        x, y = data
        y_hat = self.forward(x)
        loss = self._loss(y_hat, y)
        if y.shape[0] == 1:
            r2 = torch.tensor(0, dtype=torch.float32)
        else:
            r2 = r2_score(
                y.detach().numpy().flatten(),
                y_hat.detach().numpy().flatten()
            )
        self.log(f"{mode}_loss", loss)
        self.log(f"{mode}_r2", r2)
        return loss

    def training_step(self, data: Any, index: int) -> Tensor:
        '''training step'''
        return self._step_and_log(data, index, mode='train')

    def validation_step(self, data: Any, index: int) -> Tensor:
        '''val step'''
        return self._step_and_log(data, index, mode='val')

    def configure_optimizers(self):
        opt = optim.Adam(self.parameters(), lr=self.lr)
        return {
            'optimizer': opt,
            'lr_scheduler': {
                'scheduler': torch.optim.lr_scheduler.\
                    ReduceLROnPlateau(opt, mode='min', factor=0.1),
                'monitor': 'val_loss',
            }
        }

    def configure_callbacks(self):
        cbs = [
            pl.callbacks.ModelCheckpoint(
                dirpath=self.dir_path,
                filename='{epoch:02d}-{val_loss:.4f}-{val_r2:.4f}',
                monitor='val_r2',
                mode='max',
                save_top_k=3,
                verbose=True
            ),
            pl.callbacks.EarlyStopping(
                monitor="val_r2",
                min_delta=1e-3,
                patience=self.patience,
                mode='max'
            )
        ]
        return cbs
