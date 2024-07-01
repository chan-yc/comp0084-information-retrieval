**University College London**

**COMP0084 Information Retrieval and Data Mining (2023/24)**

**Coursework 2**

# Files Description

## Data files

Put the following data files in `data/`. See `Coursework_2.pdf` for the instructions to download these files.

- `train_data.tsv`
- `validation_data.tsv`
- `test-queries.tsv`
- `passage_collection_new.txt`
- `candidate_passages_top100.tsv`

## Submission files

- `parameters.py` - parameters for algorithms in task 2 - 4
- `utils.py` - utility functions for all tasks
- `task1.py` - main functions for task 1
- `task2.py` - main functions for task 2
- `task3.py` - main functions for task 3
- `task4.py` - main functions for task 4
- `LR.txt` - re-rank results of `candidate_passages_top1000.tsv` using Logistic Regression
- `LM.txt` - re-rank results of `candidate_passages_top1000.tsv` using LambdaMART
- `NN.txt` - re-rank results of `candidate_passages_top1000.tsv` using Neural Network
- `report.pdf` - report for coursework 2
- `environment.yml` - specification of the python environment for conda
- `README.md` - this instruction file

## Intermediate files to be created during code execution

- `assets/` - all intermediate files will be store in this directory
    - `BM25_metrics.json` - BM25 model results for reporting
    - `LR_metrics.json` - Logistic Regression model results for reporting
    - `LM_metrics.json` - LambdaMART model results for reporting
    - `NN_metrics.json` - Neural Network model results for reporting
    - `LR_learning_curves.pdf` - plot of learning curves for task 2
    - `processed_test.pkl` - processed testing data to avoid repetitive processing in tasks
    - `processed_valid.pkl` - processed validation data to avoid repetitive processing in tasks
    - `processed_train_5.pkl` - processed training data to avoid repetitive processing in tasks
    - `.vector_cache/` - directory that store downloaded word embeddings

# Python Environment Managed by Conda

Create the environment:

    $ conda env create -f environment.yml

Activate the environment:

    $ conda activate comp0084_cw2

# Code Execution

Make sure all [data files](#data-files) are in the directory `data/`.

## Execution Commands

- A single can be executed by running the corresponding main `.py` script, e.g.

      $ python task1.py

- To run all the tasks, one can use

      $ python task1.py; python task2.py; python task3.py; python task4.py;

## Execution Time

The time taken to run each script on my Mac (M2 Max 12 core) is provided as a reference.

The first time of running task 2 - 4 will perform the following one-time processes and takes extra time
1. download the GloVe embeddings (1.88GB, would takes ~6mins for a download speed of 42Mb/s)
2. compute the text embdeddings for training, validation and testing datasets (~6mins)

The following table records the time taken for each task, assuming that at least one of task 2 - 4 has been run for once.

| Command           | Time taken |
| ----------------- | ---------- |
| `python task1.py` | 1m 49s     |
| `python task2.py` | 1m 44s     |
| `python task3.py` | 1m 41s     |
| `python task4.py` | 16s        |