### Setup Environment
```shell
pip install -r requirements.txt
```

### Supported Values
- Supported Scores: fad, clap
- Supported Encoders: clap, mert
- Supported Datasets: fma-caps, music-bench
- Supported Model Names: musicgen-small, stable-audio-open-small, musicldm

### Generate Audios 
```shell
python generate.py --model_name model_name \
                   --batch_size batch_size \
                   --dataset dataset \
                   --data_dir musicdata \ 
                   --gen_data_dir musicdata/generations \
                   --hf_token [HF_TOKEN]
```

### Evaluate Models
```shell
python evaluate.py --score score \
                   --encoder_name encoder_name \
                   --model_name model_name \
                   --batch_size batch_size \
                   --dataset dataset \
                   --data_dir musicdata \ 
                   --gen_data_dir musicdata/generations \
                   --encoder_dir pretrained \
                   --hf_token [HF_TOKEN]
```

### Objective Metrics

| Model           | FAD(CLAP) | FAD(MERT) | CLAP(CLAP) |
|-----------------|-----------|:----------|------------|
| MusicGen-Small  | 0.41      | 15.62     | 0.29       |
| SAOpen-Small    | 0.43      | 18.47     | 0.20       |
| MusicLDM        | 0.26      | 9.53      | 0.32       |