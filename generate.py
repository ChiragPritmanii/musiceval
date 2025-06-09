import argparse

import pytorch_lightning as pl
from huggingface_hub import login

from musiceval.models.models import EvalPipeline

class ExceptionCallback(pl.Callback):
    def on_exception(self, trainer, module, err):
        print(f"{type(err).__name__}: {err}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", type=str, default="musicgen-small", help="Model name"
    )
    parser.add_argument("--batch_size", type=int, default=4, help="Batch Size")
    parser.add_argument("--dataset", type=str, default="fma-caps", help="Dataset")
    parser.add_argument("--data_dir", type=str, default="musicdata", help="Data Directory")
    parser.add_argument("--gen_data_dir", type=str, default="musicdata/generations", help="Generated Data Directory")
    parser.add_argument("--hf_token", type=str, default="", help="HF Token")

    args = parser.parse_args()
    login(token=args.hf_token)

    model = EvalPipeline(
        model_name=args.model_name,
        batch_size=args.batch_size,
        dataset=args.dataset,
        data_dir=args.data_dir,
        gen_data_dir=args.gen_data_dir
    )
    exc_callback = ExceptionCallback()
    trainer = pl.Trainer(
        devices="auto",
        accelerator="gpu",
        num_nodes=1,
        callbacks=[exc_callback],
        log_every_n_steps=1,
    )

    trainer.predict(model)


if __name__ == "__main__":
    # python generate.py --model_name musicgen-small --batch_size 4 --dataset fma-caps --data_dir musicdata
    main()
