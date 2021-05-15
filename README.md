# About
This repo contains datasets and code for **On the Interplay Between Fine-tuning and Composition in Transformers** (camera-ready coming soon), by Lang Yu and Allyson Ettinger.

# Dependencies
- The code is implemented with [Huggingface's transformer implementation](https://github.com/huggingface/transformers).
- The evaluation code is reused from Yu&Ettinger paper **"Assessing Phrasal Representation and Composition in Transformers"**. Code can be found here: https://github.com/yulang/phrasal-composition-in-transformers

# Repo structure
- `src/` contains source code to fine-tune transformers and perform analysis
- `scripts/` contains example scripts to run fine-tuning and evaluation jobs

# Dataset
## Fine-tuning dataset
- The Quora Question Pairs dataset in Paraphrase Adversaries from Word Scrambling (PAWS-QQP). Full dataset can be downloaded here: https://github.com/google-research-datasets/paws
- The Stanford Sentiment Treebank (SST): https://nlp.stanford.edu/sentiment/treebank.html
## Evaluation dataset
- Similarity correlation dataset: As mentioned in the paper, the full dataset can be downloaded here: http://saifmohammad.com/WebPages/BiRD.html. Please refer to the original paper for details about the dataset.
- Paraphrase classification dataset: You can download ppdb-2.0-tldr from http://paraphrase.org.

# Code
- `config.py`: configs for fine-tuning process
- `eval_config.py`: configs for evaluation
- `model_io.py`: utilities to save and load models
- `model_compare.py`: functions to compare models
- `train.py`: fine-tuning logics
- `analyze_model.py`: paws analysis code
- `fine_tune.py`: entrance for fine-tuning
- `eval_model.py`: entrance for evaluation
- `baseline_predictor.py`: linear classifier for paws analysis

# Usage
- update "/path/to/phrasal/composition/src" in `eval_model.py`, `model_compare.py` and `preprocessing.py` to be the location of src folder of https://github.com/yulang/phrasal-composition-in-transformers.
- `config.py` contains configurations for fine-tuning process. Update the file to specify: a) DATA_FOLDER (data location) and b) other fine-tuning configurations
- `eval_config.py` contains configurations for evaluation tasks. Update the file to specify: a) DATA_FOLDER (data location); b) model (type of model to run) and c) other test configs
- To run fine-tuning, refer to `run-fine-tune.sh`; `run-eval-model.sh` contains evaluation command; `run-clf.sh` contains command of training and testing classifiers (for paraphrase classification)