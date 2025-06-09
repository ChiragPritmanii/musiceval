import os
import logging
from pathlib import Path
import importlib.metadata
from abc import ABC, abstractmethod

import torch
import torchaudio
import numpy as np
from hypy_utils.downloader import download_file


class ModelLoader(ABC):
    """
    Abstract class for loading a model and getting embeddings from it. The model should be loaded in the `load_model` method.
    """
    def __init__(self, name: str, model_dir: Path, num_features: int, sr: int, min_len: int = -1):
        """
        Args:
            name (str): A unique identifier for the model.
            num_features (int): Number of features in the output embedding (dimensionality).
            sr (int): Sample rate of the audio.
            min_len (int, optional): Enforce a minimal length for the audio in seconds. Defaults to -1 (no minimum).
        """
        self.model = None
        self.model_dir = os.path.join(os.getcwd(), model_dir)
        self.sr = sr
        self.num_features = num_features
        self.name = name
        self.min_len = min_len
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def get_embedding(self, audio: np.ndarray):
        embd = self._get_embedding(audio)
        if self.device == torch.device('cuda'):
            embd = embd.cpu()
        embd = embd.detach().numpy()
        
        # If embedding is float32, convert to float16 to be space-efficient
        if embd.dtype == np.float32:
            embd = embd.astype(np.float16)

        return embd

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def _get_embedding(self, audio: np.ndarray):
        """
        Returns the embedding of the audio file. The resulting vector should be of shape (n_frames, n_features).
        """
        pass

    def load_wav(self, wav_file: str):

        wav, _ = torchaudio.load(wav_file, normalize=True) # normalize to [-1.0, +1.0]
        wav = wav.squeeze(0).numpy()
        
        # Enforce minimum length
        wav = self.enforce_min_len(wav)

        return wav
    
    def enforce_min_len(self, audio: np.ndarray) -> np.ndarray:
        """
        Enforce a minimum length for the audio. If the audio is too short, output a warning and pad it with zeros.
        """
        if self.min_len < 0:
            return audio
        if audio.shape[1] < self.min_len * self.sr:
            logging.warning(
                f"Audio is too short for {self.name}.\n"
                f"The model requires a minimum length of {self.min_len}s, audio is {audio.shape[0] / self.sr:.2f}s.\n"
                f"Padding with zeros."
            )
            audio = np.pad(audio, (0, int(np.ceil(self.min_len * self.sr - audio.shape[0]))))
        return audio

class CLAPLaionModel(ModelLoader):    
    def __init__(self, model_dir):
        super().__init__(name="clap-laion-music", num_features=512, sr=48000, min_len=10, model_dir=model_dir)
        url = 'https://huggingface.co/lukewys/laion_clap/resolve/main/music_audioset_epoch_15_esc_90.14.pt'

        self.model_file = os.path.join(self.model_dir, url.split('/')[-1])

        # Download file if it doesn't exist
        if not self.model_file.exists():
            self.model_file.parent.mkdir(parents=True, exist_ok=True)
            download_file(url, self.model_file)
            
        # Patch the model file to remove position_ids (will raise an error otherwise)
        # This key must be removed for CLAP version <= 1.1.5
        # But it must be kept for CLAP version >= 1.1.6
        package_name = "laion_clap"
        from packaging import version
        ver = version.parse(importlib.metadata.version(package_name))
        if ver < version.parse("1.1.6"):
            self.patch_model_430(self.model_file)
        else:
            self.unpatch_model_430(self.model_file)


    def patch_model_430(self, file: Path):
        """
        Patch the model file to remove position_ids (will raise an error otherwise)
        This is a new issue after the transformers 4.30.0 update
        Please refer to https://github.com/LAION-AI/CLAP/issues/127
        """
        # Create a "patched" file when patching is done
        patched = file.parent / f"{file.name}.patched.430"
        if patched.exists():
            return
        
        logging.warning("Patching LAION-CLAP's model checkpoints")
        
        # Load the checkpoint from the given path
        ck = torch.load(file, map_location="cpu")

        # Extract the state_dict from the checkpoint
        unwrap = isinstance(ck, dict) and "state_dict" in ck
        sd = ck["state_dict"] if unwrap else ck

        # Delete the specific key from the state_dict
        sd.pop("module.text_branch.embeddings.position_ids", None)

        # Save the modified state_dict back to the checkpoint
        if isinstance(ck, dict) and "state_dict" in ck:
            ck["state_dict"] = sd

        # Save the modified checkpoint
        torch.save(ck, file)
        logging.warning(f"Saved patched checkpoint to {file}")
        
        # Create a "patched" file when patching is done
        patched.touch()
            

    def unpatch_model_430(self, file: Path):
        """
        Since CLAP 1.1.6, its codebase provided its own workarounds that isn't compatible
        with our patch. This function will revert the patch to make it compatible with the new
        CLAP version.
        """
        patched = file.parent / f"{file.name}.patched.430"
        if not patched.exists():
            return
        
        # The below is an inverse operation of the patch_model_430 function, so comments are omitted
        logging.warning("Unpatching LAION-CLAP's model checkpoints")
        ck = torch.load(file, map_location="cpu")
        unwrap = isinstance(ck, dict) and "state_dict" in ck
        sd = ck["state_dict"] if unwrap else ck
        sd["module.text_branch.embeddings.position_ids"] = 0
        if isinstance(ck, dict) and "state_dict" in ck:
            ck["state_dict"] = sd
        torch.save(ck, file)
        logging.warning(f"Saved unpatched checkpoint to {file}")
        patched.unlink()
        
        
    def load_model(self):
        import laion_clap

        self.model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base')
        self.model.load_ckpt(self.model_file)
        self.model.to(self.device)
        self.model.eval()

    def load_wav(self, wav_file: str):
        new = wav_file.replace(wav_file.split("/")[-1], f"convert/{self.sr}/{wav_file.split('/')[-1]}")
        new = Path(new)
        if not os.path.exists(new):
            x, fsorig = torchaudio.load(wav_file)
            x = torch.mean(x,0).unsqueeze(0) # convert to mono
            resampler = torchaudio.transforms.Resample(
                fsorig,
                self.sr,
                lowpass_filter_width=64,
                rolloff=0.9475937167399596,
                resampling_method="sinc_interp_kaiser",
                beta=14.769656459379492,
            )
            y = resampler(x)
            torchaudio.save(str(new), y, self.ml.sr, encoding="PCM_S", bits_per_sample=16)
        else:
            wav, _ = torchaudio.load(new, normalize=True) # normalize to [-1.0, +1.0]
            wav = wav.squeeze(0).numpy()
            # Enforce minimum length
            wav = self.enforce_min_len(wav)
        return wav

    def _get_embedding(self, audio: np.ndarray) -> np.ndarray:
        audio = audio.reshape(1, -1)

        # The int16-float32 conversion is used for quantization
        audio = self.int16_to_float32(self.float32_to_int16(audio))

        # Split the audio into 10s chunks with 1s hop
        chunk_size = 10 * self.sr  # 10 seconds
        hop_size = self.sr  # 1 second
        chunks = [audio[:, i:i+chunk_size] for i in range(0, audio.shape[1], hop_size)]

        # Calculate embeddings for each chunk
        embeddings = []
        for chunk in chunks:
            with torch.no_grad():
                chunk = chunk if chunk.shape[1] == chunk_size else np.pad(chunk, ((0,0), (0, chunk_size-chunk.shape[1])))
                chunk = torch.from_numpy(chunk).float().to(self.device)
                emb = self.model.get_audio_embedding_from_data(x = chunk, use_tensor=True)
                embeddings.append(emb)

        # Concatenate the embeddings
        emb = torch.cat(embeddings, dim=0) # [timeframes, 512]
        return emb

    def int16_to_float32(self, x):
        return (x / 32767.0).astype(np.float32)

    def float32_to_int16(self, x):
        x = np.clip(x, a_min=-1., a_max=1.)
        return (x * 32767.).astype(np.int16)

class MERTModel(ModelLoader):
    def __init__(self, model_dir, size='v1-95M', layer=12, limit_seconds=10):
        super().__init__(name=f"MERT-{size}" + ("" if layer == 12 else f"-{layer}"), num_features=768, sr=24000, model_dir=model_dir)
        self.huggingface_id = f"m-a-p/MERT-{size}"
        self.layer = layer
        self.limit = limit_seconds * self.sr
        
    def load_model(self):
        from transformers import Wav2Vec2FeatureExtractor, AutoModel, AutoConfig
        
        cfg = AutoConfig.from_pretrained(self.huggingface_id, trust_remote_code=True)
        cfg.conv_pos_batch_norm = False
        self.model = AutoModel.from_pretrained(self.huggingface_id, trust_remote_code=True, config=cfg)
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(self.huggingface_id, trust_remote_code=True)
        self.model.to(self.device)
        self.model.eval()
    
    def load_wav(self, wav_file: str):
        new = wav_file.replace(wav_file.split("/")[-1], f"convert/{self.sr}/{wav_file.split("/")[-1]}")
        new = Path(new)
        if not os.path.exists(new):
            x, fsorig = torchaudio.load(wav_file)
            x = torch.mean(x,0).unsqueeze(0) # convert to mono
            resampler = torchaudio.transforms.Resample(
                fsorig,
                self.sr,
                lowpass_filter_width=64,
                rolloff=0.9475937167399596,
                resampling_method="sinc_interp_kaiser",
                beta=14.769656459379492,
            )
            y = resampler(x)
            torchaudio.save(str(new), y, self.ml.sr, encoding="PCM_S", bits_per_sample=16)
        else:
            wav, _ = torchaudio.load(new, normalize=True) # normalize to [-1.0, +1.0]
            wav = wav.squeeze(0).numpy()
            # Enforce minimum length
            wav = self.enforce_min_len(wav)
        return wav

    def _get_embedding(self, audio: np.ndarray) -> np.ndarray:
        # Limit to 9 minutes
        if audio.shape[0] > self.limit:
            logging.warning("Audio is too long. Truncating...")
            audio = audio[:self.limit]

        inputs = self.processor(audio, sampling_rate=self.sr, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model(**inputs, output_hidden_states=True)
            out = torch.stack(out.hidden_states).squeeze() # [13 layers, timeframes, 768]
            out = out[self.layer] # [timeframes, 768]
        return out