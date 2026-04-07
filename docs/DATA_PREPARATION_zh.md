# 数据准备指南

本文档介绍如何为 DeepWukong 准备自定义训练数据。

## 概述

DeepWukong 基于 **XFG（扩展流图）** 进行漏洞检测。XFG 是以安全敏感操作（API 调用、数组访问、指针解引用、算术运算）为中心的程序切片。数据准备流程如下：

```
源代码 → [joern] → PDG (CSV) → [data_generator] → XFG (pickle)
    → [dataset_generator] → 符号化 XFG + 训练/验证/测试集划分
    → [word_embedding] → Word2Vec 模型
    → [run.py] → 训练与评估
```

## 快速开始（使用预处理数据）

如果只需要在提供的 CWE119 数据集上训练：

```shell
# 下载预处理数据（约 1.8GB）
wget -O Data.7z "<README 中的 OneDrive 链接>"
7z x Data.7z -odata/

# 训练（自动检测 GPU）
PYTHONPATH="." python src/run.py

# 或使用 Docker
docker run --gpus all -v $(pwd)/data:/workspace/data deepwukong
```

## 目录结构

数据准备完成后，目录结构如下：

```
data/
├── sensiAPI.txt                          # 敏感 API 列表（仓库自带）
└── <CWE_ID>/                             # 例如 CWE119
    ├── source-code/
    │   ├── manifest.xml                  # 测试用例清单（含漏洞标签）
    │   └── <testcase dirs>/              # 按测试用例组织的源代码文件
    ├── csv/                              # joern 输出（中间产物，步骤 1）
    │   └── .../<file>.c/
    │       ├── nodes.csv
    │       └── edges.csv
    ├── XFG/                              # 生成的 XFG 文件（步骤 2-3）
    │   └── <testcaseid>/
    │       ├── call/                     # API 调用点的 XFG
    │       │   └── <line>.xfg.pkl
    │       ├── array/                    # 数组访问的 XFG
    │       │   └── <line>.xfg.pkl
    │       ├── ptr/                      # 指针解引用的 XFG
    │       │   └── <line>.xfg.pkl
    │       └── arith/                    # 算术运算的 XFG
    │           └── <line>.xfg.pkl
    ├── train.json                        # 训练集 XFG 路径列表（步骤 3）
    ├── val.json                          # 验证集 XFG 路径列表
    ├── test.json                         # 测试集 XFG 路径列表
    └── w2v.wv                            # 预训练 Word2Vec 模型（步骤 4）
```

## 分步流程

### 步骤 1：使用 joern 生成 PDG

使用[旧版 joern](https://github.com/ives-nx/dwk_preprocess/tree/main/joern_slicer/joern) 将源代码解析为 CSV 格式（节点和边）。

```shell
PYTHONPATH="." python src/joern/joern-parse.py -c configs/dwk.yaml
```

**输入：** `data/<CWE_ID>/source-code/` 下的源代码

**输出：** `data/<CWE_ID>/csv/` 下每个源文件对应的 `nodes.csv` 和 `edges.csv`

**注意：** joern 是 Java 工具，Docker 镜像中未包含，需单独安装。

### 步骤 2：从 PDG 生成 XFG

从程序依赖图中提取 XFG（程序切片），每个 XFG 以一个安全敏感操作为中心。

```shell
PYTHONPATH="." python src/data_generator.py -c configs/dwk.yaml
```

**输入：** 步骤 1 的 CSV 文件 + `manifest.xml`（漏洞标签）+ `data/sensiAPI.txt`

**输出：** `data/<CWE_ID>/XFG/<testcaseid>/{call,array,ptr,arith}/` 下的 XFG pickle 文件

### 步骤 3：符号化与数据集划分

将变量名和函数名符号化（如 `myVar` → `VAR1`，`myFunc` → `FUN1`），去重，然后划分为训练集/验证集/测试集（8:1:1）。

```shell
PYTHONPATH="." python src/preprocess/dataset_generator.py -c configs/dwk.yaml
```

**输入：** 步骤 2 的 XFG pickle 文件

**输出：**
- 符号化后的 XFG（原地修改）
- `train.json`、`val.json`、`test.json` — 包含 XFG 文件路径的 JSON 数组

### 步骤 4：训练词向量

使用训练集中的符号化 token 训练 Word2Vec 模型。

```shell
PYTHONPATH="." python src/preprocess/word_embedding.py -c configs/dwk.yaml
```

**输入：** `train.json` + 符号化 XFG 文件

**输出：** `w2v.wv` — gensim Word2Vec KeyedVectors 文件

### 步骤 5：训练与评估

```shell
PYTHONPATH="." python src/run.py -c configs/dwk.yaml
```

## 自定义数据集配置

编辑 `configs/dwk.yaml` 指向你的数据集：

```yaml
# 修改数据集名称为你的 CWE 编号
dataset:
  name: CWE119          # ← data/ 下的数据集文件夹名

# 数据根目录
data_folder: "data"      # ← 数据根文件夹

# 按需调整
gnn:
  w2v_path: "${data_folder}/${dataset.name}/w2v.wv"
  embed_size: 256        # Word2Vec 词向量维度

hyper_parameters:
  n_epochs: 50
  patience: 10           # 早停耐心值
  batch_size: 64
  learning_rate: 0.002
```

使用新数据集时，复制一份配置文件：

```shell
cp configs/dwk.yaml configs/my_dataset.yaml
# 修改 dataset.name 和其他参数
PYTHONPATH="." python src/run.py -c configs/my_dataset.yaml
```

## XFG 数据格式

每个 XFG 是一个 `networkx.DiGraph`，以 Python pickle 格式保存，包含以下内容：

**节点属性：**
- 键：行号（int）
- `code_sym_token`：符号化 token 列表，例如 `["if", "(", "VAR1", "!=", "0", ")"]`

**边属性：**
- `c/d`：`"c"` 表示控制依赖，`"d"` 表示数据依赖

**图属性：**
- `label`：`0`（非漏洞）或 `1`（漏洞）
- `file_paths`：源文件路径列表
- `key_line`：该 XFG 中心的安全敏感代码行号

**数据集划分文件**（`train.json`、`val.json`、`test.json`）：
- XFG pickle 文件路径的 JSON 数组，例如：
  ```json
  ["data/CWE119/XFG/153086/array/1019.xfg.pkl", ...]
  ```

## manifest.xml 格式

`manifest.xml` 为每个测试用例提供漏洞标签：

```xml
<container>
  <testcase id="12345">
    <file path="relative/path/to/file.c">
      <flaw line="42"/>           <!-- 漏洞行 -->
      <mixed line="55"/>          <!-- 混合漏洞/修复行 -->
      <fix line="60"/>            <!-- 修复行 -->
    </file>
  </testcase>
</container>
```

## 敏感 API 列表

`data/sensiAPI.txt` 包含以逗号分隔的安全敏感 C/C++ API 函数（如 `memcpy`、`strcpy`、`malloc`、`free`）。这些函数用于识别 XFG 提取的关键代码行。

添加自定义敏感 API：
```
...,my_custom_api,another_api
```

