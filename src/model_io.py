import logging
from config import config
import os
import torch
from transformers import AutoTokenizer, BertForSequenceClassification, RobertaForSequenceClassification, DistilBertForSequenceClassification, XLMRobertaForSequenceClassification, XLNetForSequenceClassification


logger = logging.getLogger(__name__)


def get_pretrained_name(model_name):
    if model_name == "roberta":
        pretrained_name = 'roberta-base'
        # tokenizer = RobertaTokenizer.from_pretrained(pretrained_name)
        # model = RobertaModel.from_pretrained(pretrained_name, output_hidden_states=True, output_attentions=True)
    elif model_name == "transxl":
        # need more adjustment? no cls and sep token
        # not to include right now. way too slow and memory consumption
        logger.error("not support transxl")
        exit(1)
    elif model_name == "xlnet":
        pretrained_name = 'xlnet-base-cased'
    elif model_name == "xlmroberta":
        # same architecture as roberta
        pretrained_name = 'xlm-roberta-base'
    elif model_name == "distillbert":
        pretrained_name = 'distilbert-base-cased'
    elif model_name == "bert":
        pretrained_name = 'bert-base-uncased'
    else:
        logger.error("unsupported model: {}".format(model_name))
        pretrained_name = None

    return pretrained_name


def generate_disk_location():
    model_dump_name = config["trained_model_prefix"] + config["model_name"] + config["trained_model_suffix"] + ".pt"
    model_state_dic_file_name = config["trained_model_prefix"] + config["model_name"] + config["trained_model_suffix"] + "_state_dic" + ".pt"

    model_dump_loc = os.path.join(config["trained_model_dir"], model_dump_name)
    model_state_dic_loc = os.path.join(config["trained_model_dir"], model_state_dic_file_name)

    return model_dump_loc, model_state_dic_loc


def load_model(pretrained_name, model_loc=None, load_tuned=True, num_labels=2):
    assert pretrained_name is not None
    if load_tuned:
        # load previously tuned model from disk
        if model_loc is None:
            model_dump_loc, model_state_dic_loc = generate_disk_location()
        else:
            model_dump_loc = model_loc
        model = torch.load(model_dump_loc)
        logger.info("loading model from {}".format(model_dump_loc))
    else:
        # load pretrained name from hugging face
        model_name = config["model_name"]
        if model_name == "bert":
            model = BertForSequenceClassification.from_pretrained(pretrained_name, num_labels=num_labels)
        elif model_name == "roberta":
            model = RobertaForSequenceClassification.from_pretrained(pretrained_name, num_labels=num_labels)
        elif model_name == "distillbert":
            model = DistilBertForSequenceClassification.from_pretrained(pretrained_name, num_labels=num_labels)
        elif model_name == "xlmroberta":
            model = XLMRobertaForSequenceClassification.from_pretrained(pretrained_name, num_labels=num_labels)
        elif model_name == "xlnet":
            model = XLNetForSequenceClassification.from_pretrained(pretrained_name, num_labels=num_labels)
        else:
            logger.error("unsupported model: {}".format(model_name))
        logger.info("loading pretrained model")

    tokenizer = AutoTokenizer.from_pretrained(pretrained_name)
    logger.info("model config: {}".format(model.config))

    return model, tokenizer


def dump_model(model):
    model_dump_loc, model_state_dic_loc = generate_disk_location()

    torch.save(model, model_dump_loc)
    torch.save(model.state_dict(), model_state_dic_loc)

