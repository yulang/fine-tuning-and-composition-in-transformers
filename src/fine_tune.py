import logging
import torch
from model_io import load_model, get_pretrained_name
from preprocessing import load_paws, load_stanford
from train import train_iter, evaluate
from config import config


from transformers import AdamW


logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


def main():
    holdout_test = True
    fine_tune_task = config["task"]
    logger.info("init model...")
    pretrained_name = get_pretrained_name(config["model_name"])
    if fine_tune_task == "paws":
        num_labels = 2
        train_loader, dev_loader = load_paws(config["PAWS-QQP"], balanced=config["balanced"])
    elif fine_tune_task == "standsent":
        num_labels = 5
        train_loader, dev_loader, test_loader = load_stanford(config["STANFORD_LOC"], phrase_len=config["phrase_len"], reserve_test=holdout_test)
    else:
        logger.error("unsupport fine tune task: {}".format(fine_tune_task))

    model, tokenizer = load_model(pretrained_name=pretrained_name, load_tuned=False, num_labels=num_labels)
    optimizer = AdamW(model.parameters(), lr=1e-5)

    logger.info("training...")
    model.train()
    init_prec, init_loss, best_prec, best_loss, best_model = train_iter(model, tokenizer, optimizer, train_loader, dev_loader, task=fine_tune_task, early_stopping=True, max_epochs=config["n_epochs"], print_every=config["print_every"], evaluate_every=config["evaluate_every"])
    logger.info("done training.")

    training_info_str = \
    """ training summary:
    training loss {} -> {}
    test precision {} -> {}
    """.format(init_loss, best_loss, init_prec, best_prec)
    logger.info(training_info_str)

    if holdout_test and (fine_tune_task == "standsent"):
        # evaluate on holdout test set (with phrases of various lengths)
        test_prec = evaluate(best_model, tokenizer, test_loader, fine_tune_task)
        logger.info("precision on holdout test: {}".format(test_prec))


if __name__ == "__main__":
    main()