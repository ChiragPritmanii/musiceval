import logging
import argparse

from musiceval.metrics.metrics import CLAPScore, FADScore

SCORES = ["clap", "fad"]
ENCODERS = ["clap", "mert"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--score", type=str, default="fad", help="Objective Metric")
    parser.add_argument(
        "--model_name", type=str, default="musicgen-small", help="Model Name"
    )
    parser.add_argument("--encoder_name", type=str, default="clap", help="Encoder Name")
    parser.add_argument("--dataset", type=str, default="fma-caps", help="Dataset")
    parser.add_argument(
        "--data_dir", type=str, default="musicdata", help="Data Directory"
    )
    parser.add_argument(
        "--gen_data_dir",
        type=str,
        default="musicdata/generations",
        help="Generated Data Directory",
    )
    parser.add_argument(
        "--encoder_dir",
        type=str,
        default="pretrained",
        help="Encoder Checkpoint Directory",
    )

    args = parser.parse_args()

    assert args.score in SCORES, (
        f"Score {args.score} is not supported. Choose from {SCORES}."
    )
    assert args.encoder_name in ENCODERS, (
        f"Encoder {args.encoder_name} is not supported. Choose from {ENCODERS}."
    )

    if args.score == "clap":
        scorer = CLAPScore(
            dataset=args.dataset,
            data_dir=args.data_dir,
            gen_data_dir=args.gen_data_dir,
            model_name=args.model_name,
            encoder_dir=args.encoder_dir,
        )
        score = scorer.calculate_clap_score()
        logging.info(f"CLAP Score: {score}")
    elif args.score == "fad":
        scorer = FADScore(
            dataset=args.dataset,
            data_dir=args.data_dir,
            gen_data_dir=args.gen_data_dir,
            model_name=args.model_name,
            encoder_name=args.encoder_name,
            encoder_dir=args.encoder_dir,
        )
        score = scorer.calculate_fad_score()
        logging.info(f"FAD Score: {score}")


if __name__ == "__main__":
    # python generate.py --score fad  model_name musicgen-small --encoder_name clap --dataset fma-caps --data_dir musicdata --gen_data_dir musicdata/generations --encoder_dir pretrained
    main()
