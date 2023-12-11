import json
import pathlib as pl

import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def get_metrics(yt: np.ndarray, yp: np.ndarray, out_path: pl.Path):
    metrics = {}
    error = yt - yp
    error_frac = np.abs(yt - yp)/yt
    metrics['df_test_len'] = int(yt.shape[0])
    metrics['r2_score'] = float(r2_score(yt, yp))
    metrics['mse'] = float(mean_squared_error(yt, yp))
    metrics['mae'] = float(mean_absolute_error(yt, yp))
    metrics['rmse'] = float(mean_squared_error(yt, yp, squared=False))
    metrics['smape'] = np.mean(np.abs(yt - yp)/((np.abs(yt) + np.abs(yp))/2))
    metrics['y_true_mean'] = float(yt.mean())
    metrics['y_pred_mean'] = float(yp.mean())
    metrics['y_true_median'] = float(np.median(yt))
    metrics['y_pred_median'] = float(np.median(yp))
    metrics['std_error'] = float(np.std(error))
    metrics['mean_error_frac'] = float(np.mean(error_frac))
    metrics['std_error_frac'] = float(np.std(error_frac))
    r_per, p_val = pearsonr(yt, yp)
    metrics['r_pearson'] = {
        'r_val': float(r_per),
        'p-val': float(p_val)
    }

    metrics_path = out_path.joinpath('metrics.json')
    infers_path = out_path.joinpath('infers.csv')
    with open(str(metrics_path), "w", encoding ="utf-8") as outfile:
        json.dump(metrics, indent=4, fp=outfile)
    df = pd.DataFrame({'yt': yt, 'yp': yp})
    df.to_csv(infers_path, index=False)


def _plot_scatter(yt, yp, out_path):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

    # Scatter plot
    max_val = np.percentile(yt + yp, 98) + 5
    min_val = np.percentile(yt + yp, 0) - 5
    r2 = r2_score(yt, yp)
    axes[0].scatter(yt, yp)
    axes[0].set_xlabel('Valores reais ' r'($\dfrac{mol}{m^2}$ $\times 10^6$)')
    axes[0].set_ylabel('Valores preditos ' r'($\dfrac{mol}{m^2}$ $\times 10^6$)')
    axes[0].set_title('Gráfico de dispersão das previsões - ' r'$R^2$' f' = {r2:.2f}')
    axes[0].axis([min_val, max_val, min_val, max_val])
    axes[0].plot(
        [min_val, max_val, min_val, max_val],
        [min_val, max_val, min_val, max_val],
        color='gray', linestyle='dashed'
    )
    axes[0].grid(True)

    # Boxplots
    axes[1].boxplot([yt, yp], labels=['Real', 'Predito'], widths=0.7)
    axes[1].set_ylabel('Valores ' r'($\dfrac{mol}{m^2}$ $\times 10^6$)')
    axes[1].set_title('Boxplots das distribuições real e predita')

    plt.tight_layout()
    metrics_path = out_path.joinpath('scatter_plot.pdf')
    plt.savefig(str(metrics_path), bbox_inches='tight')
    plt.show()


def _plot_error(error, out_path):
    fig = plt.figure(figsize=(8, 8))

    mu = error.mean()
    sigma = np.std(error)
    n, bins, patches = plt.hist(error, 50, density=True, linewidth=1.2, edgecolor='black',)
    y = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
     np.exp(-0.5 * (1 / sigma * (bins - mu))**2))
    plt.plot(bins, y, '--')
    plt.ylabel('Fraction of values')
    plt.xlabel('Error ' r'($\dfrac{mol}{m^2}) \times 10^6$')
    plt.title('Histogram of the distribution of error')
    plt.grid(True)
    metrics_path = out_path.joinpath('error_plot.pdf')
    plt.savefig(str(metrics_path), bbox_inches='tight')


def _plot_error_abs(error, out_path):
    fig = plt.figure(figsize=(8, 8))

    error = np.abs(error)

    mu = error.mean()
    sigma = np.std(error)
    n, bins, patches = plt.hist(error, 50, density=True, linewidth=1.2, edgecolor='black',)
    y = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
     np.exp(-0.5 * (1 / sigma * (bins - mu))**2))
    plt.plot(bins, y, '--')
    plt.ylabel('Fraction of values')
    plt.xlabel('Absolute error' r'($\dfrac{mol}{m^2} \times 10^6$)')
    plt.title('Histogram of the distribution of absolute error')
    plt.grid(True)
    metrics_path = out_path.joinpath('error_abs_plot.pdf')
    plt.savefig(str(metrics_path), bbox_inches='tight')


def plot_distributions(yt, yp, out_path):
    fig = plt.figure(figsize=(10,10))
    grid = plt.GridSpec(2, 2, figure=fig, wspace=0.2, hspace=0.3)

    ax = plt.subplot(grid[:1, :])
    n, bins, patches = ax.hist(
        yt,
        alpha=0.7,
        linewidth=1.2,
        edgecolor='black',
        bins=30,
        label=r'$y_{true}$'
    )

    ax.hist(
        yp,
        alpha=0.7,
        linewidth=1.2,
        edgecolor='black',
        bins=30,
        range=(bins[0], bins[-1]),
        label=r'$y_{pred}$'
    )
    ax.set_xlabel('Concentração 'r'$NO_2$' ' ' r'($\dfrac{mol}{m^2} \times 10^6$)')
    ax.set_ylabel('Quantidade')
    ax.set_title('Histograma com ambas as distribuições ' r'$y_{true}$ e $y_{pred}$')
    ax.legend()

    ax = plt.subplot(grid[1:, :1])
    n, bins, patches = ax.hist(
        yt,
        alpha=0.7,
        linewidth=1.2,
        edgecolor='black',
        bins=30
    )
    ax.set_xlabel('Concentração 'r'$NO_2$' ' ' r'($\dfrac{mol}{m^2} \times 10^6$)')
    ax.set_ylabel('Quantidade')
    ax.set_title('Histograma da distribuição real ' r'($y_{true}$)')

    ax = plt.subplot(grid[1:, 1:])
    ax.hist(
        yp,
        alpha=0.7,
        linewidth=1.2,
        edgecolor='black',
        bins=30,
        range=(bins[0], bins[-1])
    )
    ax.set_xlabel('Concentração 'r'$NO_2$' ' ' r'($\dfrac{mol}{m^2} \times 10^6$)')
    ax.set_ylabel('Quantidade')
    ax.set_title('Histograma da distribuição estimada ' r'($y_{pred}$)')
    metrics_path = out_path.joinpath('distributions_plot.pdf')
    plt.savefig(str(metrics_path), bbox_inches='tight')


def get_plots(yt: np.ndarray, yp: np.ndarray, out_path: pl.Path):
    metrics = {}
    error = yt - yp
    _plot_scatter(yt, yp, out_path)
    _plot_error(error, out_path)
    _plot_error_abs(error, out_path)
    plot_distributions(yt, yp, out_path)


def get_metrics_n_plots(
    infer_path: str,
    output_folder: str,
    yt: np.ndarray = None,
    yp: np.ndarray = None
):

    if (yt is None) or (yp is None):
        print('Preds and Trues did not send.')
        df = pd.read_csv(infer_path)

        yt = df['yt'].to_numpy()
        yp = df['yp'].to_numpy()

    out_p = pl.Path(output_folder)
    out_p.mkdir(exist_ok=True, parents=True)

    get_metrics(yt, yp, out_p)
    get_plots(yt, yp, out_p)
