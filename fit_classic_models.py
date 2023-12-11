import fire

from src.models.classic_models import fit_classic_models


def main(ds_path: str, pca_path: str, out_path: str):
    fit_classic_models(
        ds_path=ds_path,
        pca_path=pca_path,
        out_path=out_path
    )


if __name__=="__main__":
    fire.Fire(main)
