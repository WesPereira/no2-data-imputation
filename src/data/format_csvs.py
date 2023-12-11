import glob
import pathlib as pl
import logging

import fire
import numpy as np
import pandas as pd

from dataset import MultiTS


logging.basicConfig(datefmt='%d-%b-%y %H:%M:%S', level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


NO2_FACTOR = 10**6
np.random.seed(42)


def format_df(df: pd.DataFrame, col: str) -> pd.DataFrame:
    match col:
        case 'Column_WV' | 'Optical_Depth_047':
            df['AOD_QA'] = df['AOD_QA'].copy().apply(lambda x: bin(int(x)))
            df = df[df['AOD_QA'].str[-2:] != '11']
            df = df.drop(columns=['AOD_QA'])
            df = df.groupby(['date'], as_index=False).mean()
            return df
        case 'precipitationCal':
            return df.groupby(['date'], as_index=False).mean()
        case 'sm_surface':
            return df.groupby(['date'], as_index=False).mean()
        case 'tropospheric_NO2_column_number_density':
            df = df[df['cloud_fraction'] >= 0.5]
            df['tropospheric_NO2_column_number_density'] = \
                df['tropospheric_NO2_column_number_density'].copy().apply(lambda x: x*NO2_FACTOR)
            return df.drop(columns=['cloud_fraction'])
        case _:
            return df


def main(data_path: str, output_path: str):

    logging.info(f'Starting to process path: {data_path}')

    points_paths = glob.glob(f'{data_path}/*')

    logging.info(f'Found {len(points_paths)} point paths')

    for path in points_paths:
        logging.info(f'Starting to process path {path}')
        pt_hex = path.split('/')[-1]
        out_path_pt = f'{output_path}/{pt_hex}'

        vars_paths = glob.glob(f'{path}/*.csv')
        for vp in vars_paths:
            df = pd.read_csv(vp)
            target_var = vp.split('/')[-1].split('.csv')[0]
            formatted_df = format_df(df, target_var)
            pl.Path(out_path_pt).mkdir(parents=True, exist_ok=True)
            formatted_df.to_csv(f'{out_path_pt}/{target_var}.csv', index=False)


if __name__=="__main__":
    fire.Fire(main)
