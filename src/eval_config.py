from os import path
from random import randint

DATA_FOLDER = ""
model = "bert"

eval_config = {
    "model_name": model,
    "trained_model_loc": path.join(DATA_FOLDER, "models/bert-tuned/", "fine_tuned_distillbertfull_early.pt"),

    # --- workload locations
    "BIRD_LOC": path.join(DATA_FOLDER, "BiRD/BiRD.txt"),
    # "BIRD_LOC": path.join(DATA_FOLDER, "BiRD/bird_transposed.txt"),
    "PPDB_LOC": path.join(DATA_FOLDER, "ppdb-2.0-tldr"),
    "KINTSCH_LOC": path.join(DATA_FOLDER, "kintsch.txt"),
    "TEXT_CORPUS": path.join(DATA_FOLDER, "enwiki/enwiki-unidecoded.txt"),
    # --- tmp output locations
    "EMBEDDING_OUT_LOC": path.join(DATA_FOLDER, "bert-embeddings/"),
    "dump_path": path.join(DATA_FOLDER, "bert-dump/"),
    # --- eval configuration
    "embed_in_sent": False,
    "normalize": False,
    "workload": "ppdb", # "bird", "ppdb", "ppdb_exact"
    "sample_size": 15000,
    "rand_seed": str(randint(10000, 99999)),
    "dump_path": path.join(DATA_FOLDER, "bert-dump/"),
    "include_input_emb": True,
    "batch_size": 10,
    "dump_every": 5,
    "classification": True,
    "correlation": False,
    "compare_model": False,
    "negative_sample_mode": "half_neg",  # None, "one_per_source", "half_neg", "all_neg", (ppdb_exact will always has helf neg)
    "overlap_threshold": None,
    "n_layers": 13,
}