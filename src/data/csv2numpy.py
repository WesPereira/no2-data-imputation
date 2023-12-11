import glob
import pathlib as pl
import datetime as dt
import functools as ft
import logging

import fire
import numpy as np
import pandas as pd
from shapely.wkb import loads

from dataset import MultiTS


logging.basicConfig(datefmt='%d-%b-%y %H:%M:%S', level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


NUMBER_OF_DAYS_IN_WINDOW = 365
FRAC_OF_NANS_THRESHOLD = 0.7
NO2_NAME = 'tropospheric_NO2_column_number_density'


def main(data_path: str, output_path: str):

    logging.info(f'Starting to process path: {data_path}')

    points_paths = glob.glob(f'{data_path}/*')

    logging.info(f'Found {len(points_paths)} point paths')

    df_mapped_dates_data = []
    total_count = 0
    for path in points_paths:
        no2_df = None
        logging.info(f'Starting to process path {path}')
        pt_dfs = []
        pt_hex = path.split('/')[-1]
        out_path_pt = f'{output_path}/{pt_hex}'
        # TODO: Include point coords in the model
        # pt = loads(pt_hex, hex=True)
        vars_paths = glob.glob(f'{path}/*.csv')
        for vp in vars_paths:
            df = pd.read_csv(vp)
            df = df.drop(columns=['longitude', 'latitude'])

            # Convert datetime to days since epoch
            df['date'] = df['date'].apply(
                lambda x: (pd.to_datetime(x, format="%Y-%m-%d") \
                    - dt.datetime(1970,1,1)).days)

            target_var = vp.split('/')[-1].split('.csv')[0]

            if target_var != "tropospheric_NO2_column_number_density":
                pt_dfs.append(df)
            else:
                no2_df = df

        df_pt = ft.reduce(
            lambda left, right: pd.merge(left, right, on='date', how='outer'),
            pt_dfs
        )

        all_dates = list(set(no2_df['date'].to_list()))

        logging.info(
            f'Finished to create df with shape {df_pt.shape}.'
            ' Starting to create numpy arrays.')

        count = 0
        for _date in all_dates:

            df6 = df_pt[
                (df_pt['date'] > _date - NUMBER_OF_DAYS_IN_WINDOW) & \
                (df_pt['date'] <= _date)
            ]

            no2_target = no2_df[no2_df['date'] == _date]

            try:
                features_size = df6.shape[0] * df6.shape[1]
                num_of_nans = df6.isnull().sum().sum()
                frac_of_nans = num_of_nans / features_size
            except:
                frac_of_nans = 1

            if (frac_of_nans < FRAC_OF_NANS_THRESHOLD) and (no2_target.shape[0] == 1):
                pl.Path(out_path_pt).mkdir(parents=True, exist_ok=True)
                df_dates = pd.DataFrame({
                    'date': list(range(_date - NUMBER_OF_DAYS_IN_WINDOW+1, _date+1))
                })
                df_data = df_dates.merge(df6, how='left', on='date')
                ts = MultiTS(
                    series = df_data.drop(columns=['date']).to_numpy(),
                    dse=df_data['date'].to_numpy(),
                    masks=(df_data.drop(columns=['date']).isnull()).to_numpy()
                )
                uuid_ = ts.save_numpy(out_path_pt)
                no2_value = float(no2_target[NO2_NAME].iloc[0])
                no2_days = int(float(no2_target['date'].iloc[0]))
                target_date = dt.datetime(1970,1,1) + dt.timedelta(no2_days)
                no2_array = np.array([no2_value])
                no2_path = f'{out_path_pt}/{uuid_}_no2.npz'
                np.savez(no2_path, no2=no2_array)
                df_mapped_dates_data.append(
                    (no2_path, f'{out_path_pt}/{uuid_}.npz', target_date)
                )
                count += 1

        pd.DataFrame(
            df_mapped_dates_data,
            columns=['no2_path', 'features_path', 'date']
        ).to_csv(f'{output_path}/mapped_paths.csv', index=False)
        logging.info(f'{count} samples were created.')
        total_count += count
    logging.info(f'Final number of samples created: {total_count}')


if __name__=="__main__":
    fire.Fire(main)