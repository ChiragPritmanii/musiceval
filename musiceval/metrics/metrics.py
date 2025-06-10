import os
import logging
import random
from glob import glob
from tqdm import tqdm

import torch
import numpy as np
from scipy import linalg

from musiceval.data.data import EvalDataset
from musiceval.encoders.encoders import CLAPLaionModel, MERTModel


class CLAPScore:
    def __init__(self, dataset, data_dir, gen_data_dir, model_name, encoder_dir):
        ref_data = EvalDataset(dataset=dataset, data_dir=data_dir)
        self.ref_samples = ref_data.samples
        self.ref_texts = [sample["main_caption"] for sample in self.ref_samples]
        self.ref_audio_paths = [
            os.path.join(
                os.getcwd(), data_dir, ref_data.dataset_name, sample["location"]
            )
            for sample in self.ref_samples
        ]
        self.gen_audio_paths = [
            os.path.join(
                os.getcwd(),
                gen_data_dir,
                ref_data.dataset,
                model_name,
                path.split("/")[-1],
            )
            for path in self.ref_audio_paths
        ]
        self.encoder = CLAPLaionModel(encoder_dir)
        self.encoder.load_model()

    def calculate_clap_score(self):
        logging.info("Encoding Reference Texts")
        batch_size = 64
        text_emb = []
        for i in tqdm(range(0, len(self.ref_texts), batch_size)):
            batch_texts = self.ref_texts[i : i + batch_size]
            with torch.no_grad():
                embeddings = self.encoder.model.get_text_embedding(
                    batch_texts, use_tensor=True
                )
            text_emb.append(embeddings.cpu())
        text_emb = torch.cat(text_emb, dim=0)

        score = 0
        count = 0
        logging.info("Encoding Generated Audios")
        for i in tqdm(range(0, len(self.gen_audio_paths))):
            with torch.no_grad():
                audio = self.encoder.load_wav(self.gen_audio_paths[i])
                audio_embedding = self.encoder._get_embedding(audio=audio)
            cosine_sim = torch.nn.functional.cosine_similarity(
                audio_embedding.cpu(), text_emb[i].unsqueeze(0), dim=1, eps=1e-8
            )[0]
            score += cosine_sim
            count += 1

        clap_score = score / count if count > 0 else 0

        return clap_score


class FADScore:
    def __init__(
        self, dataset, data_dir, gen_data_dir, model_name, encoder_name, encoder_dir
    ):
        ref_data = EvalDataset(dataset=dataset, data_dir=data_dir, limit=10)
        self.ref_samples = ref_data.samples
        self.ref_samples = ref_data.samples
        self.ref_audio_paths = [
            os.path.join(
                os.getcwd(), data_dir, ref_data.dataset_name, sample["location"]
            )
            for sample in self.ref_samples
        ]
        self.gen_audio_paths = [
            os.path.join(
                os.getcwd(),
                gen_data_dir,
                ref_data.dataset,
                model_name,
                path.split("/")[-1],
            )
            for path in self.ref_audio_paths
        ]
        if encoder_name == "clap":
            self.encoder = CLAPLaionModel(encoder_dir)
        elif encoder_name == "mert":
            self.encoder = MERTModel(encoder_dir)
        self.encoder.load_model()

    def calculate_embd_statistics(self, embd_lst):
        if isinstance(embd_lst, list):
            embd_lst = np.array(embd_lst)
        mu = np.mean(embd_lst, axis=0)
        sigma = np.cov(embd_lst, rowvar=False)
        return mu, sigma

    def calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
        """
        Adapted from: https://github.com/mseitzer/pytorch-fid/blob/master/src/pytorch_fid/fid_score.py
        Adapted from: https://github.com/gudgud96/frechet-audio-distance/blob/main/frechet_audio_distance/fad.py

        Numpy implementation of the Frechet Distance.

        The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
        and X_2 ~ N(mu_2, C_2) is
                d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

        Params:
        -- mu1: Embedding's mean statistics for generated samples.
        -- mu2: Embedding's mean statistics for reference samples.
        -- sigma1: Covariance matrix over embeddings for generated samples.
        -- sigma2: Covariance matrix over embeddings for reference samples.
        Returns:
        --  Fréchet Distance.
        """

        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert mu1.shape == mu2.shape, (
            "Training and test mean vectors have different lengths"
        )
        assert sigma1.shape == sigma2.shape, (
            "Training and test covariances have different dimensions"
        )

        diff = mu1 - mu2

        # product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = (
                "fid calculation produces singular product; "
                "adding %s to diagonal of cov estimates"
            ) % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError("Imaginary component {}".format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

    def extract_embeddings(self):
        audio_gen_embeddings = []
        audio_ref_embeddings = []

        logging.info("Encoding Generated and Reference Audios")

        for i in tqdm(range(0, len(self.gen_audio_paths))):
            with torch.no_grad():
                audio_gen = self.encoder.load_wav(self.gen_audio_paths[i])
                audio_ref = self.encoder.load_wav(self.ref_audio_paths[i])

                audio_gen_embedding = self.encoder._get_embedding(audio=audio_gen)
                audio_ref_embedding = self.encoder._get_embedding(audio=audio_ref)

                audio_gen_embeddings.append(audio_gen_embedding.cpu().numpy())
                audio_ref_embeddings.append(audio_ref_embedding.cpu().numpy())
        
        audio_gen_embeddings = np.concatenate(audio_gen_embeddings, axis=0)
        audio_ref_embeddings = np.concatenate(audio_ref_embeddings, axis=0)

        return audio_gen_embeddings, audio_ref_embeddings

    def calculate_fad_score(self):
        audio_gen_embeddings, audio_ref_embeddings = self.extract_embeddings()
        mu_gen, cov_gen = self.calculate_embd_statistics(audio_gen_embeddings)
        mu_ref, cov_ref = self.calculate_embd_statistics(audio_ref_embeddings)
        fad_score = self.calculate_frechet_distance(mu_gen, cov_gen, mu_ref, cov_ref)
        return fad_score
    