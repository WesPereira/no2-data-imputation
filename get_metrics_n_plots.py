import fire

from src.metrics_and_plots import get_metrics_n_plots


def main(infer_path: str, output_folder: str):
    get_metrics_n_plots(infer_path, output_folder)


if __name__=="__main__":
    fire.Fire(main)
