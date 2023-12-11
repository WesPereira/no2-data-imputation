import fire

from src.infer import infer


def main(model_path: str, test_path: str, output_path: str):
    infer(
        model_path=model_path,
        test_path=test_path,
        output_path=output_path
    )


if __name__=="__main__":
    fire.Fire(main)
