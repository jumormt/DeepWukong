# DeepWukong

[中文文档](README_zh.md)

> (TOSEM'21) DeepWukong: Statically Detecting Software Vulnerabilities Using Deep Graph Neural Network

## Results

Results on CWE119 dataset (GCN encoder, RTX A5000 x2):

| Metric | Value |
|--------|-------|
| Accuracy | 98.30% |
| F1 | 94.56% |
| Precision | 95.10% |
| Recall | 94.02% |
| FPR | 0.90% |

## Docker (Recommended)

### Build

```shell
docker build -t deepwukong .
```

### Train with preprocessed data

```shell
# Download and extract data first
wget -O Data.7z "<link below>"
7z x Data.7z -odata/

# Train on GPU
docker run --gpus all -v $(pwd)/data:/workspace/data deepwukong

# Train with custom config
docker run --gpus all \
  -v $(pwd)/data:/workspace/data \
  -v $(pwd)/configs:/workspace/configs \
  deepwukong src/run.py -c configs/dwk.yaml
```

### Run other pipeline steps

```shell
# Evaluate a trained model
docker run --gpus all \
  -v $(pwd)/data:/workspace/data \
  -v $(pwd)/ts_logger:/workspace/ts_logger \
  deepwukong src/evaluate.py ts_logger/DeepWuKong/CWE119/version_0/checkpoints/<checkpoint>.ckpt

# Generate XFGs (requires joern CSV output in data/)
docker run --gpus all -v $(pwd)/data:/workspace/data \
  deepwukong src/data_generator.py -c configs/dwk.yaml

# Symbolize and split dataset
docker run --gpus all -v $(pwd)/data:/workspace/data \
  deepwukong src/preprocess/dataset_generator.py -c configs/dwk.yaml

# Train word embeddings
docker run --gpus all -v $(pwd)/data:/workspace/data \
  deepwukong src/preprocess/word_embedding.py -c configs/dwk.yaml
```

### GPU Requirements

- NVIDIA GPU with CUDA 12.4+ support
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed
- Use `--gpus all` to enable GPU access (CPU fallback is automatic)

## Local Setup

- Requirements: Python >= 3.9, PyTorch >= 2.0 (CUDA recommended)

- Install dependencies

    ```shell
    bash env.sh
    ```

    Or manually:

    ```shell
    # Install PyTorch (use CUDA version for GPU support)
    # CPU only:
    pip install torch
    # CUDA 12.4:
    pip install torch --index-url https://download.pytorch.org/whl/cu124

    pip install torch-geometric
    pip install -r requirements.txt
    ```

## Data

- **Preprocessed CWE119 data:** Download from [data](https://bupteducn-my.sharepoint.com/:u:/g/personal/jackiecheng_bupt_edu_cn/EalnVAYC8zZDgwhPmGJ034cBYNZ8zB7-mNSNm-a7oYXkcw?e=eRUc50) (7z, ~1.8GB), extract under `data/`:

    ```shell
    7z x Data.7z -odata/
    ```

- **Custom datasets:** See [Data Preparation Guide](docs/DATA_PREPARATION.md) ([中文版](docs/DATA_PREPARATION_zh.md)) for the full pipeline from source code to training data.

- **Expected structure:**
    ```
    data/
    ├── sensiAPI.txt
    └── CWE119/
        ├── train.json, val.json, test.json
        ├── w2v.wv
        ├── XFG/          # XFG pickle files
        └── source-code/  # Original source (needed for preprocessing only)
    ```

---

## One-Step Evaluation

- From Pretrained model

  - Download from [pretrained model](https://bupteducn-my.sharepoint.com/:u:/g/personal/jackiecheng_bupt_edu_cn/EesTvivx1UlEo9THYRSCYkMBMsZqKXgNVYx9wTToYnDwxg?e=Z4nz23).
  - **Note:** Pretrained checkpoints saved with pytorch-lightning 1.x may not be compatible. Retraining is recommended.
  - `PYTHONPATH="." python src/evaluate.py <path to the pretrained model>`

- Training and Testing

  ```shell
  PYTHONPATH="." python src/run.py
  ```

  GPU is auto-detected. Training on CWE119 takes ~9 minutes on dual RTX A5000.

---

**Run from scratch:**

## Data preparation

See [Data Preparation Guide](docs/DATA_PREPARATION.md) for detailed instructions on preparing custom datasets.

### Use joern to Generate PDG

joern is included in `joern/` directory (requires Java 8+).

```shell
PYTHONPATH="." python src/joern/joern-parse.py -c <config file>
```

### Generate raw XFG

```shell
PYTHONPATH="." python src/data_generator.py -c <config file>
```

### Symbolize and Split Dataset

```shell
PYTHONPATH="." python src/preprocess/dataset_generator.py -c <config file>
```

### Word Embedding Pretraining

```shell
PYTHONPATH="." python src/preprocess/word_embedding.py -c <config file>
```

## Evaluation

```shell
PYTHONPATH="." python src/run.py -c <config file>
```


## Acknowledgements

The modernization of this project (dependency upgrades, Docker support, and data preparation documentation) was done with [Claude Code](https://claude.ai/claude-code).

## Citation

Please kindly cite our paper if it benefits:

```bib
@article{xiao2021deepwukong,
author = {Cheng, Xiao and Wang, Haoyu and Hua, Jiayi and Xu, Guoai and Sui, Yulei},
title = {DeepWukong: Statically Detecting Software Vulnerabilities Using Deep Graph Neural Network},
year = {2021},
publisher = {ACM},
volume = {30},
number = {3},
url = {https://doi.org/10.1145/3436877},
doi = {10.1145/3436877},
journal = {ACM Trans. Softw. Eng. Methodol.},
articleno = {38},
numpages = {33}
}
```
