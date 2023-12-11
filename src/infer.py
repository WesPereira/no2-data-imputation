import logging

import torch
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib.pyplot as plt

from src.models.trainer import ModelsTrainer
from src.models.dataset import MultiTSDataset


logging.basicConfig(datefmt='%d-%b-%y %H:%M:%S', level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def infer(
    model_path: str,
    test_path: str,
    output_path: str,
    model_type: str,
    config = None
):
    logging.info('Starting infer step...')

    if not config:
        parts = model_path.split('/')[-2]
        hidden_dim = int(parts.split('_')[-2].split('hd')[-1])
        nl = int(parts.split('_')[-1].split('nl')[-1])
        bs = int(parts.split('_')[-3].split('bs')[-1])
        lr = float(parts.split('_')[-4].split('lr')[-1])

        config = {
            "lr": lr,
            "batch_size": bs,
            "hidden_dim": hidden_dim,
            "n_layers": nl,
        }

    model = ModelsTrainer.load_from_checkpoint(
        model_path,
        device=torch.device('cpu'),
        config=config,
        artifacts_path='test',
        model_type=model_type
    )

    df = pd.read_csv(f'{test_path}/mapped_paths.csv')

    df_test = df[df['ds_type'] == 'test']
    test_ds = MultiTSDataset(df_test, use_rel=True)
    #test_ds = MultiTSDataset(test_path)
    test_dl = DataLoader(test_ds, batch_size=32)

    logging.info('Loader check.')

    yts = []
    yps = []
    batch = 0

    for x, yt in test_dl:
        yp = model.forward(x)
        yps.extend(torch.squeeze(yp).tolist())
        yts.extend(torch.squeeze(yt).tolist())

        batch += 1
        logging.info(f'At {batch=}...')

    df = pd.DataFrame(data={'yt': yts, 'yp': yps})

    df.to_csv(output_path, index=False)

    logging.info('Fertig.')

    return yts, yps
