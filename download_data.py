import json
import yaml
import logging
import datetime as dt
import pathlib as pl
from typing import List
from multiprocessing import Pool
from copy import deepcopy

import ee
from ee.ee_exception import EEException
import pandas as pd
from shapely.geometry import shape, Point


logging.basicConfig(datefmt='%d-%b-%y %H:%M:%S', level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class Params:
    TARGET = "COPERNICUS/S5P/OFFL/L3_NO2"
    MAX_RESOLUTION = None
    BASE_DS_PATH_TO_SAVE = None
    MIN_DATE = None
    MAX_DATE = None
    BASE_CRS = None

def construct_params():
    with open("datasets.yml", 'r') as stream:
        ds_params = yaml.safe_load(stream)
    params = Params()

    params.MIN_DATE = ds_params['parameters']['start_date']
    params.MAX_DATE = ds_params['parameters']['end_date']
    params.BASE_DS_PATH_TO_SAVE = ds_params['parameters']['output_file']
    max_resolution_ds = max(ds_params['data'], key=lambda x: x['resolution'])
    params.MAX_RESOLUTION = max_resolution_ds['resolution']
    ee.Initialize()
    params.BASE_CRS = ee.ImageCollection(max_resolution_ds["image_ref"]) \
                    .select([max_resolution_ds["bands"][0]["name"]]) \
                    .first().projection().crs()
    return ds_params, params

ds_params, params = construct_params()


def init_gee():
    #ee.Authenticate()
    ee.Initialize()


def convert_df_types(df: pd.DataFrame, list_of_bands: List[str]) -> pd.DataFrame:

    # Convert the time field into a datetime.
    df['date'] = pd.to_datetime(df['time'], unit='ms').dt.date

    # Dropping NaNs
    df = df[['date', 'longitude', 'latitude'] + list_of_bands].dropna()

    # Convert the data to numeric values.
    for band in list_of_bands:
        df[band] = pd.to_numeric(df[band], errors='coerce')

    return df


def _download_chunks(img_coll, u_poi, params):
    start_date = dt.datetime.strptime(params.MIN_DATE, "%Y-%m-%d").date()
    end_date = dt.datetime.strptime(params.MAX_DATE, "%Y-%m-%d").date()
    time_delta = dt.timedelta(days=180)
    mid_date = start_date + time_delta
    dfs = []
    while mid_date < end_date:
        logging.info(f'Starting chunk for dates: {str(start_date)} - {str(mid_date)}.')
        coll_data = img_coll.filterDate(str(start_date), str(mid_date))
        coll_data = coll_data.map(convert_crs)
        coll_data = coll_data.getRegion(u_poi, params.MAX_RESOLUTION).getInfo()
        coll_df = pd.DataFrame(coll_data[1:], columns=coll_data[0])
        dfs.append(coll_df)
        start_date = mid_date
        mid_date = start_date + time_delta
    coll_data = img_coll.filterDate(str(start_date), str(end_date)) \
        .getRegion(u_poi, params.MAX_RESOLUTION).getInfo()
    coll_df = pd.DataFrame(coll_data[1:], columns=coll_data[0])
    dfs.append(coll_df)
    return pd.concat(dfs)


def _task(geom, img_coll, dataset_bands, output_path, params):
    ee.Initialize()
    u_poi = ee.Geometry.Point(geom.x, geom.y)
    try:
        coll_data = img_coll.filterDate(params.MIN_DATE, params.MAX_DATE)
        coll_data = coll_data.map(convert_crs)
        coll_data = coll_data.getRegion(geometry=u_poi, scale=params.MAX_RESOLUTION).getInfo()
        coll_df = pd.DataFrame(coll_data[1:], columns=coll_data[0])
    except EEException as e:
        logging.info(f'{e}. Trying to download in chunks')
        coll_df = _download_chunks(img_coll, u_poi, params)
    converted_df = convert_df_types(coll_df, dataset_bands)
    save_path = f'{params.BASE_DS_PATH_TO_SAVE}{geom.wkb_hex}'
    pl.Path(save_path).mkdir(parents=True, exist_ok=True)
    converted_df.to_csv(f'{save_path}/{output_path}', index=False)
    logging.info(f'Finished for geom: {geom}. Saved at {save_path}/{output_path}.')

def convert_crs(img):
    img2 = img.reduceResolution(
      reducer=ee.Reducer.mean(),
      bestEffort=True
    ).reproject(
    crs=params.BASE_CRS)
    return img2


def download_data_to_folder(gee_dataset: str, list_of_geoms: str,
                            dataset_bands: List[str], output_path: str, params: Params):

    logging.info(f'Starting to get img coll: {gee_dataset} and bands {dataset_bands}')
    img_coll = ee.ImageCollection(gee_dataset).select(dataset_bands)
    logging.info('Collection fetched.')

    with open(list_of_geoms) as f:
        features = json.load(f)["features"]

    geoms = [shape(feature["geometry"]) for feature in features]

    logging.info(f'Starting to get data for {len(geoms)} points.')

    args = [(g, img_coll, dataset_bands, output_path, params) for g in geoms]

    with Pool(processes=3) as pool:
        pool.starmap(_task, args)


def main(gee_dataset: str, dataset_bands: List[str],
         list_of_geoms: str, output_path: str, params: Params):

    logging.info("Starting to auth.")
    init_gee()
    logging.info('Auth check.')

    download_data_to_folder(gee_dataset, list_of_geoms,
                            dataset_bands, output_path, params)


if __name__=="__main__":
    for ds in ds_params["data"]:
        if ds["downloaded"] == 1:
            logging.info(f'Starting to process from source {ds["dataset_name"]}, max res: {params.BASE_CRS}')
            gee_dataset = ds["image_ref"]
            if gee_dataset != params.TARGET:
                correct_params = deepcopy(params)
                correct_params.MIN_DATE = "2016-05-01"
            else:
                correct_params = deepcopy(params)
            for band in ds["bands"]:
                bands = [band["name"]]
                output_path = f"{band['name']}.csv"
                if ds["quality_band"] != "":
                    bands.append(ds["quality_band"])
                main(
                    gee_dataset=gee_dataset,
                    dataset_bands=bands,
                    list_of_geoms="points_2.json",
                    output_path=output_path,
                    params=correct_params
                )
