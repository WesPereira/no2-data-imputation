import glob
import logging
import functools as ft
import datetime as dt

import pandas as pd
from shapely.wkb import loads


logging.basicConfig(datefmt='%d-%b-%y %H:%M:%S', level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def apply_rules(target_var: str, df: pd.DataFrame):
    if target_var == "tropospheric_NO2_column_number_density":
        _vars = [target_var, "date", "cloud_fraction"]
        df = df[_vars]
        df = df[df["cloud_fraction"] < 0.4]
        df[target_var] = df[target_var] * 10**8
        df = df.groupby(['date'], as_index=False)[target_var].mean()
    elif target_var == "Column_WV":
        _vars = [target_var, "date", "AOD_QA"]
        df = df[_vars]
        df['AOD_QA'] = df['AOD_QA'].apply(lambda x: bin(int(x)))
        df = df[df['AOD_QA'].str[-2:] != '11']
        df = df.groupby(['date'], as_index=False)[target_var].mean()
    else:
        df = df[['date', target_var]]
        df = df.groupby(['date'], as_index=False)[target_var].mean()
    return df


def get_dataset(ds_path: str):
    points_paths = glob.glob(f'{ds_path}/*')

    dfs_pts = []
    for path in points_paths:
        pt_dfs = []
        pt_hex = path.split('/')[-1]
        pt = loads(pt_hex, hex=True)
        vars_paths = glob.glob(f'{path}/*.csv')
        for vp in vars_paths:
            df = pd.read_csv(vp)
            df['date'] = df['date'].apply(lambda x: (pd.to_datetime(x, format="%Y-%m-%d") - dt.datetime(1970,1,1)).days)
            #df['date'] = (pd.to_datetime(df['date'], format="%Y-%m-%d") - dt.datetime(1970,1,1))
            target_var = vp.split('/')[-1].split('.csv')[0]
            #df = df[['date', target_var]]
            #df = df.groupby(['date'], as_index=False)[target_var].max()
            df = apply_rules(target_var, df)
            pt_dfs.append(df)

        df_pt = ft.reduce(
            lambda left, right: pd.merge(left, right, on='date', how='inner'),
            pt_dfs
        )

        df_pt['long'] = [pt.x] * df_pt.shape[0]
        df_pt['lat'] = [pt.y] * df_pt.shape[0]

        dfs_pts.append(df_pt)

    final_df = pd.concat(dfs_pts)
    return final_df


def main():
    ds_path = 'gee_datasets'

    logging.info(f'Starting to get dataset from {ds_path}')
    df = get_dataset(ds_path)

    logging.info(f'Final df with shape {df.shape} and columns {df.columns}.')

    df.to_csv('final_ds.csv', index=False)


if __name__=="__main__":
    main()
