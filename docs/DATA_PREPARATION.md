# Data Preparation Guide

This guide explains how to prepare custom training data for DeepWukong.

## Overview

DeepWukong works on **XFG** — a program slice centered around security-sensitive operations (API calls, array accesses, pointer dereferences, arithmetic). The pipeline is:

```
Source Code → [joern] → PDG (CSV) → [data_generator] → XFG (pickle)
    → [dataset_generator] → Symbolized XFG + train/val/test split
    → [word_embedding] → Word2Vec model
    → [run.py] → Train & Evaluate
```

## Quick Start (Preprocessed Data)

If you just want to train on the provided CWE119 dataset:

```shell
# Download preprocessed data (~1.8GB)
wget -O Data.7z "<OneDrive link from README>"
7z x Data.7z -odata/

# Train
PYTHONPATH="." python src/run.py
```

## Directory Structure

After preparation, your data directory should look like:

```
data/
├── sensiAPI.txt                          # Sensitive API list (provided in repo)
└── <CWE_ID>/                             # e.g., CWE119
    ├── source-code/
    │   ├── manifest.xml                  # Test case manifest with vulnerability labels
    │   └── <testcase dirs>/              # Source files organized by test case
    ├── csv/                              # joern output (intermediate, Step 1)
    │   └── .../<file>.c/
    │       ├── nodes.csv
    │       └── edges.csv
    ├── XFG/                              # Generated XFGs (Step 2-3)
    │   └── <testcaseid>/
    │       ├── call/                     # XFGs for API call sites
    │       │   └── <line>.xfg.pkl
    │       ├── array/                    # XFGs for array accesses
    │       │   └── <line>.xfg.pkl
    │       ├── ptr/                      # XFGs for pointer dereferences
    │       │   └── <line>.xfg.pkl
    │       └── arith/                    # XFGs for arithmetic operations
    │           └── <line>.xfg.pkl
    ├── train.json                        # List of XFG paths for training (Step 3)
    ├── val.json                          # List of XFG paths for validation
    ├── test.json                         # List of XFG paths for testing
    └── w2v.wv                            # Pretrained Word2Vec (Step 4)
```

## Step-by-Step Pipeline

### Step 1: Generate PDG with joern

Use the [old version of joern](https://github.com/ives-nx/dwk_preprocess/tree/main/joern_slicer/joern) to parse source code into CSV format (nodes and edges).

```shell
PYTHONPATH="." python src/joern/joern-parse.py -c configs/dwk.yaml
```

**Input:** Source code under `data/<CWE_ID>/source-code/`

**Output:** CSV files under `data/<CWE_ID>/csv/` with `nodes.csv` and `edges.csv` per source file.

**Note:** joern is a Java tool and is NOT included in the Docker image. Install it separately.

### Step 2: Generate XFGs from PDG

Extract XFG (program slices) from the PDG. Each XFG is centered on a security-sensitive operation.

```shell
PYTHONPATH="." python src/data_generator.py -c configs/dwk.yaml
```

**Input:** CSV files from Step 1 + `manifest.xml` (vulnerability labels) + `data/sensiAPI.txt`

**Output:** XFG pickle files under `data/<CWE_ID>/XFG/<testcaseid>/{call,array,ptr,arith}/`

### Step 3: Symbolize and Split Dataset

Symbolize variable/function names (e.g., `myVar` → `VAR1`, `myFunc` → `FUN1`), deduplicate XFGs, and split into train/val/test.

```shell
PYTHONPATH="." python src/preprocess/dataset_generator.py -c configs/dwk.yaml
```

**Input:** XFG pickle files from Step 2

**Output:**
- Symbolized XFGs (modified in place)
- `train.json`, `val.json`, `test.json` — JSON arrays of XFG file paths

### Step 4: Train Word Embeddings

Train a Word2Vec model on the symbolized tokens from the training set.

```shell
PYTHONPATH="." python src/preprocess/word_embedding.py -c configs/dwk.yaml
```

**Input:** `train.json` + symbolized XFG files

**Output:** `w2v.wv` — gensim Word2Vec KeyedVectors file

### Step 5: Train and Evaluate

```shell
PYTHONPATH="." python src/run.py -c configs/dwk.yaml
```

## Custom Dataset Configuration

Edit `configs/dwk.yaml` to point to your dataset:

```yaml
# Change dataset name to your CWE ID
dataset:
  name: CWE119          # ← your dataset folder name under data/

# Data location
data_folder: "data"      # ← root data folder

# Adjust if needed
gnn:
  w2v_path: "${data_folder}/${dataset.name}/w2v.wv"
  embed_size: 256        # Word2Vec embedding dimension

hyper_parameters:
  n_epochs: 50
  patience: 10           # Early stopping patience
  batch_size: 64
  learning_rate: 0.002
```

For a new CWE type, create a copy of the config:

```shell
cp configs/dwk.yaml configs/my_dataset.yaml
# Edit dataset.name and other parameters
PYTHONPATH="." python src/run.py -c configs/my_dataset.yaml
```

## XFG Data Format

Each XFG is a `networkx.DiGraph` saved as a Python pickle file with:

**Node attributes:**
- Key: line number (int)
- `code_sym_token`: list of symbolized tokens, e.g., `["if", "(", "VAR1", "!=", "0", ")"]`

**Edge attributes:**
- `c/d`: `"c"` for control dependency, `"d"` for data dependency

**Graph attributes:**
- `label`: `0` (non-vulnerable) or `1` (vulnerable)
- `file_paths`: list of source file paths
- `key_line`: the security-sensitive line number this XFG is centered on

**Split files** (`train.json`, `val.json`, `test.json`):
- JSON arrays of file paths to XFG pickle files, e.g.:
  ```json
  ["data/CWE119/XFG/153086/array/1019.xfg.pkl", ...]
  ```

## manifest.xml Format

The `manifest.xml` provides vulnerability labels for each test case:

```xml
<container>
  <testcase id="12345">
    <file path="relative/path/to/file.c">
      <flaw line="42"/>           <!-- vulnerable line -->
      <mixed line="55"/>          <!-- mixed vulnerable/fixed line -->
      <fix line="60"/>            <!-- fixed line -->
    </file>
  </testcase>
</container>
```

## Sensitive API List

`data/sensiAPI.txt` contains a comma-separated list of security-sensitive C/C++ API functions (e.g., `memcpy`, `strcpy`, `malloc`, `free`). These are used to identify key lines for XFG extraction.

To add custom sensitive APIs, append them to this file:
```
...,my_custom_api,another_api
```

