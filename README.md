# DeepWukong

> (TOSEM'21) DeepWukong: Statically Detecting Software Vulnerabilities Using Deep Graph Neural Network

## Setup

```shell
bash env.sh
```

---

## One-Step Evaluation

```shell
bash run.sh
```

---

**Run from beginning:**

## Data preparation

### Use joern to Generate PDG

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
