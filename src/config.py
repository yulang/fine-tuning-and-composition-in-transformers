from os import path

DATA_FOLDER = ""

config = {
    "PAWS-WIKI": path.join(DATA_FOLDER, "paws/wiki/labeled_final/"),
    "PAWS-QQP": path.join(DATA_FOLDER, "paws/qqp/"),
    "balanced": True, # whether to load the label balanced version of paws
    "STANFORD_LOC": path.join(DATA_FOLDER, "stanfordSentimentTreebank"),
    "model_name": "roberta",
    "output_loc": path.join(DATA_FOLDER, "sent-out"),
    "task": "paws", # paws, standsent,
    "phrase_len": None, # config for stanford sent only
    "trained_model_prefix": "fine_tuned_",        
    "trained_model_dir": path.join(DATA_FOLDER, "models/bert-tuned"),
    "trained_model_suffix": "_balanced_paws",
    # training configs
    "n_epochs": 3,
    "print_every": 100,
    "evaluate_every": 200,
}