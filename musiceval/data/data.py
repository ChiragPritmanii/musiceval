import os
import json
import random
import logging
import subprocess
from typing import Dict, List

import torch
import torchaudio
from torch.utils.data import Dataset

SEED = 42
FILELIMIT = 10
DATASETS = ["fma-caps", "music-bench"]

random.seed(SEED)

class EvalDataset(Dataset):
    def __init__(self, dataset: str, data_dir: str):
        assert dataset in DATASETS, f"Dataset must be one of {DATASETS}"
        self.dataset = dataset
        self.data_dir = os.path.join(os.getcwd(), data_dir)
        self._load_dataset()

    def _load_dataset(self):
        if self.dataset == "fma-caps":
            url = "https://huggingface.co/datasets/amaai-lab/MusicBench/resolve/main/FMACaps_eval_set.tar.gz"
            json_path = os.path.join(self.data_dir, "FMACaps_eval_set/FMACaps_A.json")
        elif self.dataset == "music-bench":
            url = "https://huggingface.co/datasets/amaai-lab/MusicBench/resolve/main/MusicBench.tar.gz"
            url_json = "https://huggingface.co/datasets/amaai-lab/MusicBench/resolve/main/MusicBench_test_A.json"
            json_path = os.path.join(self.data_dir, "MusicBench/MusicBench_test_A.json")
        else:
            raise ValueError(f"Dataset must be one of {DATASETS}")

        self.tar_name = url.split("/")[-1]
        self.dataset_name = url.split("/")[-1].split(".")[0]
        tar_path = os.path.join(self.data_dir, self.tar_name)
        dataset_path = os.path.join(self.data_dir, self.dataset_name)

        if self.dataset == "music-bench" and not os.path.exists(json_path):
            logging.info(
                f"Downloading JSON metadata for {self.dataset_name} to {json_path}"
            )
            subprocess.run(["wget", url_json, "-O", json_path], check=True)

        if not os.path.exists(dataset_path):
            if not os.path.exists(tar_path):
                logging.info(
                    f"Downloading dataset {self.dataset_name} to {self.data_dir}"
                )
                subprocess.run(["wget", url, "-O", tar_path], check=True)
            else:
                logging.info(
                    f"Dataset {self.dataset_name} already downloaded at {tar_path}"
                )

            logging.info(f"Extracting dataset {self.dataset_name} to {self.data_dir}")
            subprocess.run(["tar", "-xzf", tar_path, "-C", self.data_dir], check=True)
        else:
            logging.info(
                f"Dataset {self.dataset_name} already downloaded and extracted at {self.data_dir}"
            )

        with open(json_path, "r") as f:
            self.samples = [json.loads(line) for line in f]
        
        if len(self.samples) > FILELIMIT:
            logging.info(
                f"Limiting dataset to {FILELIMIT} samples for evaluation. Original size: {len(self.samples)}"
            )
            self.samples = random.sample(self.samples, FILELIMIT)

    def __len__(self):
        return len(self.samples)

    def _load_audio(self, audio_path: str):
        try:
            audio_path = os.path.join(self.data_dir, audio_path)
            wav, sr = torchaudio.load(audio_path, normalize=True)
            return wav, sr
        except Exception as e:
            print(f"Error loading item {audio_path}: {e}")
            return None, None

    def __getitem__(self, index: int):
        sample = self.samples[index]
        sample_processed = {"index": index}
        sample_processed["item_name"] = sample["location"].split("/")[-1].split(".")[0]

        prompt = sample.get("main_caption", None)
        if type(prompt) is str:
            prompt = prompt.strip()

        if prompt is None or prompt == "":
            prompt = sample.get("alt_caption", None)
            if type(prompt) is str:
                prompt = prompt.strip()

        if prompt is None or prompt == "":
            prompt = None

        sample_processed["prompt"] = prompt
        sample_processed["gt_audio"], sample_processed["gt_audio_sr"] = (
            self._load_audio(sample["location"])
        )

        if any(
            x is None
            for x in (sample_processed["gt_audio"], sample_processed["prompt"])
        ):
            logging.info(
                f"Skipping item {sample['item_name']} due to missing audio or prompt data."
            )
            return None

        return sample_processed

    def collator(self, samples: List[Dict]):
        samples = [sample for sample in samples if sample is not None]
        indices = [sample["index"] for sample in samples]
        item_names = [sample["item_name"] for sample in samples]
        prompts = [sample["prompt"] for sample in samples]

        gt_audios = []
        gt_audios_sr = []
        for sample in samples:
            gt_audios.append(sample["gt_audio"])
            gt_audios_sr.append(sample["gt_audio_sr"])

        gt_audios = torch.stack(gt_audios, dim=0)

        net_input = {
            "id": indices,
            "item_names": item_names,
            "prompts": prompts,
            "gt_audios": gt_audios,
            "gt_audios_sr": gt_audios_sr,
            "batch_size": len(samples),
        }

        return net_input
