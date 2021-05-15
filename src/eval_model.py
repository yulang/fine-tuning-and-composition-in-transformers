from eval_config import eval_config
from model_io import load_model, get_pretrained_name
from train import encode_batch
from preprocessing import EvalDataset
from model_compare import *

from transformers import *
import logging
import sys
import os
from torch.utils.data import DataLoader, TensorDataset
import torch
from numpy import save, load

sys.path.append("/path/to/phrasal/composition/src")

from workload_generator import *
from classifier import *
from kintsch_exp import kintsch_preprocess
from analyzer import TransformerAnalyzer
from utilities import adjust_transformer_range, analyze_correlation_by_layer, print_stats_by_layer, \
    concact_hidden_states



logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


def init_base_model(model_name):
    if model_name == "roberta":
        pretrained_name = 'roberta-base'
        tokenizer = RobertaTokenizer.from_pretrained(pretrained_name)
        model = RobertaModel.from_pretrained(pretrained_name, output_hidden_states=True, output_attentions=True)
    elif model_name == "xlnet":
        pretrained_name = 'xlnet-base-cased'
        tokenizer = XLNetTokenizer.from_pretrained(pretrained_name)
        model = XLNetModel.from_pretrained(pretrained_name, output_hidden_states=True, output_attentions=True)
    elif model_name == "xlmroberta":
        pretrained_name = 'xlm-roberta-base'
        tokenizer = XLMRobertaTokenizer.from_pretrained(pretrained_name)
        model = XLMRobertaModel.from_pretrained(pretrained_name, output_hidden_states=True, output_attentions=True)
    elif model_name == "distillbert":
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased')
        model = DistilBertModel.from_pretrained('distilbert-base-cased', output_hidden_states=True, output_attentions=True)
    elif model_name == "bert":
        pretrained_name = 'bert-base-uncased'
        tokenizer = BertTokenizer.from_pretrained(pretrained_name)
        model = BertModel.from_pretrained(pretrained_name, output_hidden_states=True, output_attentions=True)
    else:
        logger.error("unsupported model: {}".format(model_name))

    return model, tokenizer


# ugly, but easiest way to do it without retraining the model
def remove_clf_head(model):
    cur_rand_seed = eval_config["rand_seed"]
    tmp_dir = os.path.join("./tmp/pytorch_dump", eval_config["model_name"], cur_rand_seed)
    model.save_pretrained(tmp_dir)
    base_model = AutoModel.from_pretrained(tmp_dir, output_hidden_states=True, output_attentions=True)
    return base_model


# replace encode_padded_input in phrasal repo
def encode_input(tokenizer, dataloader, model_name):
    max_len = 250
    input_id_list, attention_mask_list, sequence_length = [], [], []
    pad_id = tokenizer.pad_token_id
    for sequence_batch in dataloader:
        encoded_info = tokenizer(sequence_batch, \
                                return_tensors='pt', padding='max_length', truncation=True, max_length=max_len)
        input_ids, attention_masks = encoded_info['input_ids'], encoded_info['attention_mask']
        input_id_list.append(input_ids)
        attention_mask_list.append(attention_masks)
        batch_max_len = input_ids.shape[1]
        length_list = [batch_max_len - list(x).count(pad_id) for x in input_ids]
        sequence_length.extend(length_list)

    input_id_list = torch.cat(input_id_list, 0)
    attention_mask_list = torch.cat(attention_mask_list, 0)
    return input_id_list, attention_mask_list, sequence_length


def forward_input(model, model_name, input_ids, input_mask):
    if model_name in ["roberta", "bert", "xlmroberta"]:
        last_hidden_state, pooler_output, hidden_states, attentions = model(input_ids, attention_mask=input_mask)
    elif model_name in ["distillbert"]:
        last_hidden_state, hidden_states, attentions = model(input_ids, attention_mask=input_mask)
    elif model_name in ['xlnet']:
        last_hidden_state, mems, hidden_states, attentions = model(input_ids, attention_mask=input_mask)
    else:
        logger.error("unsupported model: {}".format(model_name))
        exit(1)
    return hidden_states


def eval_and_dump_embeddings(model, model_name, data_loader, dump_path):
    assert os.path.exists(dump_path) is False
    dump_write_handler = open(dump_path, "ab")
    accumulated_hidden_states = None
    cached_count = 0

    for input_ids, input_mask in data_loader:
        hidden_states = forward_input(model, model_name, input_ids, input_mask)

        if accumulated_hidden_states is None:
            accumulated_hidden_states = list(hidden_states)
        else:
            accumulated_hidden_states = concact_hidden_states(accumulated_hidden_states, hidden_states)
        cached_count += 1

        if cached_count == eval_config["dump_every"]:
            save(dump_write_handler, accumulated_hidden_states)  # note: dump accumulated hidden states
            cached_count = 0
            accumulated_hidden_states = None

    if cached_count != 0:
        # dump remaining segments
        save(dump_write_handler, accumulated_hidden_states)
        accumulated_hidden_states = None

    dump_write_handler.close()


def main():
    random_seed = eval_config["rand_seed"]
    logger.info("current random seed: {}".format(random_seed))
    model_name = eval_config["model_name"]
    logger.info("preprocessing input...")
    if eval_config["workload"] == "bird":
        input_filename, score_dic, score_range, phrase_pos, phrase_text = bird_preprocess(eval_config["BIRD_LOC"],
                                                                                          random_seed,
                                                                                          eval_config["sample_size"],
                                                                                          normalize=eval_config["normalize"], out_folder="./out")
        phrase_dic = score_dic
    elif eval_config["workload"] == "ppdb":
        input_filename, score_dic, score_range, phrase_pos, phrase_text, samples_dic = \
            ppdb_preprocess(eval_config["PPDB_LOC"], random_seed, eval_config["sample_size"],
                            negative_sampling_mode=eval_config["negative_sample_mode"],
                            overlap_threshold=eval_config["overlap_threshold"], out_folder="./out/")
        phrase_dic = score_dic
    elif eval_config["workload"] == "ppdb_exact":
        input_filename, exact_label_dic, phrase_pos, phrase_text = ppdb_exact_preprocess(eval_config["PPDB_LOC"],
                                                                                         random_seed,
                                                                                         eval_config["sample_size"], out_folder="./out")
        phrase_dic = exact_label_dic
    elif eval_config["workload"] == "stanford_sent":
        input_filename, phrase_pos, phrase_text, phrase_labels, phrase_scores = stanfordsent_preprocess(random_seed,
                                                                                                        eval_config[
                                                                                                            "sample_size"])
        # TODO embed in sents not support
        phrase_dic = None
    elif eval_config["workload"] == "kintsch":
        input_filename, landmark_samples, inference_samples, phrase_pos, phrase_text = kintsch_preprocess(random_seed)
        # TODO embed in sents not support
        phrase_dic = None
    else:
        print("unsupport workload " + eval_config["workload"])
        exit(1)

    logger.info("current eval_configuration: {}".format(eval_config))

    if eval_config["embed_in_sent"]:
        logger.info("Embedding phrase in wiki text")
        if eval_config["workload"] == "ppdb_exact":
            logger.info("Before truncating: {}".format(len(phrase_text)))
            sentence_texts, phrase_text, exact_label_dic = embed_phrase_and_truncate(phrase_dic, phrase_text, eval_config["TEXT_CORPUS"])
            logger.info("After truncating: {}".format(len(sentence_texts)))
        else:
            sentence_texts = embed_phrase_transformer(phrase_dic, phrase_text, eval_config["TEXT_CORPUS"])

        sents_loc = "out/embedded_sents_" + random_seed + ".txt"
        sent_out = open(sents_loc, "w")
        for sentence in sentence_texts:
            sent_out.write(sentence)
        sent_out.close()
    
    logger.info("loading model...")
    pretrained_name = get_pretrained_name(eval_config["model_name"])
    model, tokenizer = load_model(model_loc=eval_config["trained_model_loc"], load_tuned=True, pretrained_name=pretrained_name)
    model = remove_clf_head(model)
    if eval_config["compare_model"]:
        base_model, base_tokenizer = init_base_model(model_name)

    logger.info("model being evaluated: {}".format(model.config))
    
    model_config = model.config
    n_layers, n_heads = model_config.num_hidden_layers, model_config.num_attention_heads

    logger.info("encoding input...")
    if eval_config["embed_in_sent"]:
        eval_text_dataset = EvalDataset(sentence_texts)
    else:
        eval_text_dataset = EvalDataset(phrase_text)

    # shuffling has to be turned off. need to keep the order to adjust phrase position etc.
    eval_text_loader = DataLoader(dataset=eval_text_dataset, shuffle=False, batch_size=eval_config["batch_size"])
    # input_id_list, attention_mask_list, phrase_length_list = encode_input(tokenizer, eval_text_loader, eval_config["model_name"])
    input_id_list, attention_mask_list, input_sequence_length_list = encode_input(tokenizer, eval_text_loader, eval_config["model_name"])

    logger.info("adjusting phrase position & genreating label dic")
    if (model_name in ['roberta']) and (eval_config["embed_in_sent"] is True):
        # tokenizer is space sensitive. 'access' has different id than ' access'
        add_space_before_phrase = True
    else:
        add_space_before_phrase = False

    phrase_pos = adjust_transformer_range(phrase_text, input_id_list, tokenizer, model_name, space_before_phrase=add_space_before_phrase)

    if eval_config["classification"] and (eval_config["workload"] in ["bird", "ppdb"]):
        # generate label dic for classification task
        if eval_config["negative_sample_mode"] is None:
            label_dic = nontrivial_score_to_label(score_dic, score_range)
        else:
            label_dic = trivial_score_to_label(score_dic)

    #----------------------------- evaluation -------------------------------#
    logger.info("evaluating model")
    model.eval()
    dump_filename = "{}-dump-{}.npy".format(model_name, random_seed)
    dump_path = os.path.join(eval_config["dump_path"], dump_filename)
    batch_size = eval_config["batch_size"]

    eval_data = TensorDataset(input_id_list, attention_mask_list)
    data_loader = DataLoader(eval_data, batch_size=batch_size, shuffle=False)

    eval_and_dump_embeddings(model, model_name, data_loader, dump_path)

    if eval_config["compare_model"]:
        base_model.eval()
        base_dump_filename = "{}-dump-{}-base.npy".format(model_name, random_seed)
        base_dump_path = os.path.join(eval_config["dump_path"], base_dump_filename)
        eval_and_dump_embeddings(base_model, model_name, data_loader, base_dump_path)

    logger.info("dumping segment size: {} samples per segment".format(batch_size * eval_config["dump_every"]))

    logger.info("working on downstream task")
    analyzer = TransformerAnalyzer(dump_path, n_layers, phrase_text, phrase_text, input_sequence_length_list, model_name,
                                   eval_config["include_input_emb"])
    if eval_config["compare_model"]:
        base_analyzer = TransformerAnalyzer(base_dump_path, n_layers, phrase_text, phrase_text, input_sequence_length_list, model_name, eval_config["include_input_emb"])
        embedding_sims = modelwise_compare([analyzer, base_analyzer], phrase_text, phrase_pos, model_name)
        analyze_embedding_dic(embedding_sims)
        analyze_max_changes(embedding_sims)

        analyzer.reset_handler()
        base_analyzer.reset_handler()
        
        max_change_pairs_by_layer = modelwise_phrase_pair_analysis([analyzer, base_analyzer], score_dic, phrase_pos, phrase_text)
        
        # finished comparison. no need to run other tasks
        return

    if eval_config["workload"] == "kintsch":
        logger.info("writing out kintsch embeddings")
        out_embedding_dir = os.path.join(eval_config["EMBEDDING_OUT_LOC"], model_name)
        if os.path.exists(out_embedding_dir) is False:
            os.mkdir(out_embedding_dir)

        dump_read_handler = open(dump_path, "rb")
        generate_kintsch_embeddings_transformer(dump_read_handler, out_embedding_dir, phrase_pos, input_sequence_length_list,
                                                landmark_samples, inference_samples, n_layers,
                                                eval_config["include_input_emb"])
        dump_read_handler.close()

        logger.info("evaluating kintsch embeddings")
        evaluate_kintsch_embeddings(os.path.join(out_embedding_dir, "kintsch"), landmark_samples, inference_samples,
                                    n_layers, eval_config["include_input_emb"])
    elif eval_config["workload"] in ["bird", "ppdb"]:
        if eval_config["correlation"]:
            logger.info("analyzing correlation...")
            coe_by_layer, cos_sim_by_layer, target_score = analyze_correlation_by_layer(analyzer, score_dic, phrase_pos,
                                                                                        eval_config["include_input_emb"])
            print_stats_by_layer(coe_by_layer, is_list=False, stat_type="cor", out_folder="./out")
            analyzer.reset_handler()

        if eval_config["classification"]:
            logger.info("generating classification workloads...")
            generate_classifier_workloads(analyzer, eval_config, random_seed, phrase_text, label_dic, phrase_pos,
                                          eval_config["include_input_emb"])
    elif eval_config["workload"] == "ppdb_exact":
        generate_classifier_workloads(analyzer, eval_config, random_seed, phrase_text, exact_label_dic, phrase_pos,
                                      eval_config["include_input_emb"])
    elif eval_config["workload"] == "stanford_sent":
        generate_stanford_classifier_workloads(analyzer, eval_config, random_seed, phrase_text, phrase_labels, phrase_pos,
                                               eval_config["include_input_emb"])
    else:
        logger.error("unsupport task {}".format(eval_config["workload"]))

    #----------------------------- training classifiers (if workload is classification) -------------------------------#
    if eval_config["classification"]:
        logger.info("training classifiers on embeddings...")
        n_layers = eval_config["n_layers"]

        working_dir = os.path.join(eval_config["EMBEDDING_OUT_LOC"], str(eval_config["rand_seed"]))

        verify_embeddings(n_layers, working_dir)
        label_handler = open(os.path.join(working_dir, "label.txt"), "r")
        configure_handler = open(os.path.join(working_dir, "config.txt"), "r")
        labels = []
        core_count = mp.cpu_count()
        pool = mp.Pool(core_count)
        logger.info("Using {} cores".format(core_count))
        logger.info("Current configurations:")
        text = configure_handler.readlines()
        logger.info(text)

        for line in label_handler:
            line = line.strip()
            labels.append(line)

        # logger.info("classification by layer...")
        # classify_by_layer(n_layers, labels, pool, working_dir)
        logger.info("classification by token...")
        classify_by_token(n_layers, labels, pool, working_dir)

        label_handler.close()
        configure_handler.close()


if __name__ == "__main__":
    main()