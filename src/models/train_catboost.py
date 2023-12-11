import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.linear_model import Lasso, Ridge
import catboost as cb
import lightgbm as lgb
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from dataset import MultiTSDataset


DS_PATH = "/Users/wesleypereira/Documents/tcc/exploration/gee_exploration/data/processed/dataset5"


def main():
    ds = MultiTSDataset(DS_PATH)
    x_data = []
    target_data = []
    for x, y in ds:
        x_data.append(x.flatten().tolist())
        target_data.append(y.flatten().tolist())

    features = np.array(x_data)
    target = np.array(target_data).flatten()

    X_train, X_test, Y_train, Y_test = train_test_split(features, target, test_size = 0.25, random_state = 42)

    train_dataset = cb.Pool(X_train, Y_train)
    test_dataset = cb.Pool(X_test, Y_test)

    model = cb.CatBoostRegressor(loss_function='RMSE')

    grid = {'iterations': [100, 150, 200],
        'learning_rate': [0.03, 0.1],
        'depth': [2, 4, 6, 8],
        'l2_leaf_reg': [0.2, 0.5, 1, 3]}

    print('Starting gs.')

    model.grid_search(grid, train_dataset)

    pred = model.predict(X_test)
    rmse = (np.sqrt(mean_squared_error(Y_test, pred)))
    r2 = r2_score(Y_test, pred)
    print("Testing performance")
    print("RMSE: {:.2f}".format(rmse))
    print("R2: {:.2f}".format(r2))


if __name__=="__main__":
    main()
