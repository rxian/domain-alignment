"""Load dataset for sequence classification. Refactored from `run_text_cls.py`."""

import datasets
from datasets import load_dataset


def load_raw_dataset(
    dataset_name=None,
    dataset_config_name=None,
    train_file=None,
    evaluation_file=None,
    text_column_name=None,
    label_column_name=None,
):

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets for text classification task available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).

    # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.

    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)
    if dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(dataset_name, dataset_config_name)
    else:
        # Loading the dataset from local csv or json file.
        data_files = {}
        if train_file is not None:
            data_files["train"] = train_file
        if evaluation_file is not None:
            data_files["test"] = evaluation_file
        extension = (train_file if train_file is not None else evaluation_file).split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files)
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Labels
    label_list = raw_datasets["train"].unique(label_column_name)
    label_list.sort()  # Let's sort it for determinism
    label_to_id = {label: i for i, label in enumerate(label_list)}
    
    return raw_datasets, label_list, label_to_id


def tokenize_raw_dataset(
    tokenizer,
    raw_datasets,
    label_list,
    label_to_id,
    text_column_name,
    label_column_name,
    pad_to_max_length=False,
    max_length=None,
):

    # Preprocessing the datasets
    padding = "max_length" if pad_to_max_length else False

    def preprocess_function(examples):
        # Tokenize the texts
        texts = [examples[n] for n in text_column_name]
        result = tokenizer(*texts, padding=padding, max_length=max_length, truncation=True)

        if label_column_name in examples:
            if label_to_id is not None:
                # Map labels to IDs (not necessary for GLUE tasks)
                result["labels"] = [label_to_id[l] for l in examples[label_column_name]]
            else:
                # In all cases, rename the column to labels because the model will expect that.
                result["labels"] = examples[label_column_name]
        return result

    processed_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
        desc="Running tokenizer on dataset",
    )

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["test"]

    return train_dataset, eval_dataset
