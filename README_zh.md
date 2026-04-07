# DeepWukong

[English](README.md)

> (TOSEM'21) DeepWukong: Statically Detecting Software Vulnerabilities Using Deep Graph Neural Network

## Docker（推荐）

### 构建

```shell
docker build -t deepwukong .
```

### 使用预处理数据训练

```shell
# 先下载并解压数据
# 下载链接见下方"数据"部分
wget -O Data.7z "https://bupteducn-my.sharepoint.com/:u:/g/personal/jackiecheng_bupt_edu_cn/EalnVAYC8zZDgwhPmGJ034cBYNZ8zB7-mNSNm-a7oYXkcw?e=eRUc50&download=1"
7z x Data.7z -odata/

# GPU 训练
docker run --gpus all -v $(pwd)/data:/workspace/data deepwukong

# 使用自定义配置
docker run --gpus all \
  -v $(pwd)/data:/workspace/data \
  -v $(pwd)/configs:/workspace/configs \
  deepwukong src/run.py -c configs/dwk.yaml
```

### 从源代码开始训练（完整流程）

将源代码放在 `data/<数据集名>/source-code/` 下，并准备 `manifest.xml`（格式见[数据准备指南](docs/DATA_PREPARATION_zh.md)），然后逐步执行：

```shell
# 第 1 步：用 joern 从源代码生成 PDG
docker run --gpus all -v $(pwd)/data:/workspace/data \
  deepwukong src/joern/joern-parse.py -c configs/dwk.yaml

# 第 2 步：从 PDG 提取 XFG
docker run --gpus all -v $(pwd)/data:/workspace/data \
  deepwukong src/data_generator.py -c configs/dwk.yaml

# 第 3 步：符号化并划分训练/验证/测试集
docker run --gpus all -v $(pwd)/data:/workspace/data \
  deepwukong src/preprocess/dataset_generator.py -c configs/dwk.yaml

# 第 4 步：训练词向量
docker run --gpus all -v $(pwd)/data:/workspace/data \
  deepwukong src/preprocess/word_embedding.py -c configs/dwk.yaml

# 第 5 步：训练模型
docker run --gpus all -v $(pwd)/data:/workspace/data \
  deepwukong src/run.py -c configs/dwk.yaml

# 第 6 步（可选）：评估已保存的模型
docker run --gpus all \
  -v $(pwd)/data:/workspace/data \
  -v $(pwd)/ts_logger:/workspace/ts_logger \
  deepwukong src/evaluate.py <checkpoint 路径>
```

使用不同数据集时，复制 `configs/dwk.yaml` 并修改 `dataset.name`：

```shell
docker run --gpus all \
  -v $(pwd)/data:/workspace/data \
  -v $(pwd)/configs:/workspace/configs \
  deepwukong src/run.py -c configs/my_dataset.yaml
```

### GPU 要求

- 支持 CUDA 12.4+ 的 NVIDIA GPU
- 已安装 [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- 使用 `--gpus all` 启用 GPU（无 GPU 时自动回退到 CPU）

## 本地安装

- 环境要求：Python >= 3.10，PyTorch >= 2.0（推荐 CUDA 版本）

- 安装依赖

    ```shell
    bash env.sh
    ```

    或手动安装：

    ```shell
    # 安装 PyTorch
    # 仅 CPU：
    pip install torch
    # CUDA 12.4：
    pip install torch --index-url https://download.pytorch.org/whl/cu124

    pip install torch-geometric
    pip install -r requirements.txt
    ```

## 数据

- **预处理 CWE119 数据：** 从 [data](https://bupteducn-my.sharepoint.com/:u:/g/personal/jackiecheng_bupt_edu_cn/EalnVAYC8zZDgwhPmGJ034cBYNZ8zB7-mNSNm-a7oYXkcw?e=eRUc50)（7z，约 1.8GB）下载，解压到 `data/` 目录：

    ```shell
    7z x Data.7z -odata/
    ```

- **自定义数据集：** 参见[数据准备指南](docs/DATA_PREPARATION_zh.md)（[English](docs/DATA_PREPARATION.md)），了解从源代码到训练数据的完整流程。

- **目录结构：**
    ```
    data/
    ├── sensiAPI.txt
    └── CWE119/
        ├── train.json, val.json, test.json
        ├── w2v.wv
        ├── XFG/          # XFG pickle 文件
        └── source-code/  # 源代码（仅预处理阶段需要）
    ```

---

## 快速评估

- 使用预训练模型

  - 从 [pretrained model](https://bupteducn-my.sharepoint.com/:u:/g/personal/jackiecheng_bupt_edu_cn/EesTvivx1UlEo9THYRSCYkMBMsZqKXgNVYx9wTToYnDwxg?e=Z4nz23) 下载。
  - **注意：** 使用 pytorch-lightning 1.x 保存的预训练模型可能不兼容，建议重新训练。
  - `PYTHONPATH="." python src/evaluate.py <预训练模型路径>`

- 训练与测试

  ```shell
  PYTHONPATH="." python src/run.py
  ```

  自动检测 GPU。在双 RTX A5000 上训练 CWE119 约需 9 分钟。

---

**从零开始：**

## 数据准备

详细说明请参见[数据准备指南](docs/DATA_PREPARATION_zh.md)。

### 使用 joern 生成 PDG

joern 已包含在 `joern/` 目录中（需要 Java 8+）。

```shell
PYTHONPATH="." python src/joern/joern-parse.py -c <配置文件>
```

### 生成 XFG

```shell
PYTHONPATH="." python src/data_generator.py -c <配置文件>
```

### 符号化与数据集划分

```shell
PYTHONPATH="." python src/preprocess/dataset_generator.py -c <配置文件>
```

### 词向量预训练

```shell
PYTHONPATH="." python src/preprocess/word_embedding.py -c <配置文件>
```

## 训练与评估

```shell
PYTHONPATH="." python src/run.py -c <配置文件>
```

## 致谢

本项目的现代化升级（依赖更新、Docker 支持、数据准备文档）由 [Claude Code](https://claude.ai/claude-code) 辅助完成。

## 引用

如果本项目对您有帮助，请引用我们的论文：

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
