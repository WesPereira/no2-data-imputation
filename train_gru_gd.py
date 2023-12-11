import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
#from pytorch_lightning.loggers import TensorBoardLogger
from ray.tune.schedulers import ASHAScheduler
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray import tune, air

from src.models.dataset import MultiTSDataset
from src.models.trainer import GRUTrainerGD


DS_PATH = "/Users/wesleypereira/Documents/tcc/exploration/gee_exploration/data/processed/dataset6"


def main(config, ds_dir = DS_PATH, num_epochs = 2):
    ds = MultiTSDataset(ds_dir)
    train_size = int(0.8 * len(ds))
    val_size = len(ds) - train_size
    train_ds, val_ds = torch.utils.data.random_split(ds, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True, num_workers=8)
    val_loader = DataLoader(val_ds, batch_size=config['batch_size'], num_workers=8)

    callback = TuneReportCallback(["val_loss", "val_r2"], on='validation_epoch_end')

    model = GRUTrainerGD(
        config=config
    )
    trainer = pl.Trainer(accelerator="cpu", max_epochs=num_epochs, enable_progress_bar=False, callbacks=[callback])
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__=="__main__":
    config = {
        "hidden_dim": tune.choice([16, 32, 64]),
        "n_layers": tune.choice([1, 2, 3]),
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([32, 64, 128])
    }

    trainable = tune.with_parameters(
        main,
        ds_dir=DS_PATH,
        num_epochs=15)

    scheduler = ASHAScheduler(
        max_t=15,
        grace_period=1,
        reduction_factor=2
    )

    resources_per_trial = {"cpu": 0.8, "gpu": 0}

    tuner = tune.Tuner(
        tune.with_resources(
            trainable,
            resources=resources_per_trial
        ),
        tune_config=tune.TuneConfig(
            metric="val_loss",
            mode="min",
            scheduler=scheduler,
            num_samples=5,
        ),
        run_config=air.RunConfig(
            name="test"
        ),
        param_space=config,
    )
    results = tuner.fit()
    # analysis = tune.run(
    #     trainable,
    #     resources_per_trial={
    #         "cpu": 0.8,
    #         "gpu": 0
    #     },
    #     metric="_val_loss",
    #     mode="min",
    #     config=config,
    #     num_samples=5,
    #     name="tune_gru")

    print("Best hyperparameters found were: ", results.get_best_result().config)

    #print(analysis.best_config)
