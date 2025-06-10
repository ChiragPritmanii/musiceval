import os
import logging

import torch
import torchaudio
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from diffusers import MusicLDMPipeline
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond
from transformers import AutoProcessor, MusicgenForConditionalGeneration

from musiceval.data.data import EvalDataset

MODELS = ["musicgen-small", "stable-audio-open-small", "musicldm"]


class EvalPipeline(pl.LightningModule):
    def __init__(
        self,
        model_name: str,
        batch_size: int,
        dataset: str,
        data_dir: str,
        gen_data_dir: str,
    ):
        super().__init__()
        assert model_name in MODELS, (
            f"Model {model_name} is not supported. Choose from {MODELS}."
        )
        self.model_name = model_name
        self.batch_size = batch_size
        self.dataset = dataset
        self.data_dir = os.path.join(os.getcwd(), data_dir)
        self.gen_data_dir = os.path.join(os.getcwd(), gen_data_dir)

    def setup(self, stage=None):
        if self.model_name == "musicgen-small":
            hf_model_id = "facebook/musicgen-small"
            self.processor = AutoProcessor.from_pretrained(hf_model_id)
            self.model = MusicgenForConditionalGeneration.from_pretrained(hf_model_id)
            self.model_sr = self.model.config.sampling_rate
            self.output_dir = os.path.join(
                self.gen_data_dir, self.dataset, "musicgen-small"
            )
            self.model = self.model.to(self.device)
            self.model.eval()
        elif self.model_name == "stable-audio-open-small":
            hf_model_id = "stabilityai/stable-audio-open-small"
            self.model, self.model_config = get_pretrained_model(
                "stabilityai/stable-audio-open-small"
            )
            self.model_sr = self.model_config["sample_rate"]
            self.output_dir = os.path.join(
                self.gen_data_dir, self.dataset, "stable-audio-open-small"
            )
            self.model = self.model.to(self.device)
            self.model.eval()
        elif self.model_name == "musicldm":
            hf_model_id = "ucsd-reach/musicldm"
            self.model = MusicLDMPipeline.from_pretrained(
                hf_model_id, torch_dtype=torch.float16
            )
            self.output_dir = os.path.join(self.gen_data_dir, self.dataset, "musicldm")
            self.model_sr = self.model.vocoder.config.sampling_rate
            self.model = self.model.to(self.device)
        else:
            raise ValueError(f"Model {self.model_name} is not supported.")

        logging.info(f"Loaded model {self.model_name}")

    def predict_dataloader(self):
        eval_data = EvalDataset(dataset=self.dataset, data_dir=self.data_dir)
        eval_dl = DataLoader(
            eval_data,
            batch_size=self.batch_size,
            num_workers=0,
            shuffle=False,
            collate_fn=eval_data.collator,
            prefetch_factor=None,
            pin_memory=True,
        )
        return eval_dl

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        if self.model_name == "musicgen-small":
            inputs = self.processor(
                text=batch["prompts"], return_tensors="pt", padding=True
            ).to(self.device)
            audios = self.model.generate(**inputs, max_new_tokens=512)
        elif self.model_name == "stable-audio-open-small":
            inputs = [
                {"prompt": prompt, "seconds_total": 10} for prompt in batch["prompts"]
            ]
            audios = generate_diffusion_cond(
                self.model,
                conditioning=inputs,
                device=self.device,
                steps=8,
                cfg_scale=1.0,
                batch_size=len(inputs),
                sample_size=self.model_config["sample_size"],
                sampler_type="pingpong",
            )
        elif self.model_name == "musicldm":
            self.model = self.model.to("cuda")
            audios = self.model(
                batch["prompts"], num_inference_steps=200, audio_length_in_s=10.0
            ).audios
            audios = torch.tensor(audios).unsqueeze(1)

        processed_audios = []
        for i in range(audios.shape[0]):
            wav = audios[i]
            wav = wav.to(torch.float32)
            wav = wav / wav.abs().max()
            wav = wav.clamp(-1, 1)
            wav = (wav * 32767).to(torch.int16).cpu()
            processed_audios.append(wav)
        print("pass")
        audios = torch.stack(processed_audios, dim=0)
        logging.info(f"Generated {len(audios)} audio samples for batch {batch_idx}")

        return audios

    def on_predict_batch_end(self, outputs, batch, batch_idx):
        for i, item in enumerate(batch["item_names"]):
            out_path = os.path.join(self.output_dir, f"{item}.wav")
            os.makedirs(self.output_dir, exist_ok=True)
            torchaudio.save(out_path, outputs[i], self.model_sr)
        logging.info(f"Generated audio saved to {self.output_dir}")
        return
