from torch.nn import CosineSimilarity
from model_io import load_model, get_pretrained_name
from eval_model import remove_clf_head
from numpy import var
import pdb
import sys

sys.path.append("/path/to/phrasal/composition/src")

from workload_generator import extract_related_embedding

def normalized_cos_sim(emb1, emb2):
    cos_sim = CosineSimilarity(dim=0)
    sim = cos_sim(emb1, emb2)
    sim = (sim + 1) / 2.0
    return sim.item()


def modelwise_compare(analyzer_tuple, phrase_text, phrase_pos, model_name):
    # phrase pos should be a dic: {phrase_index: (phrase_start_pos, phrase_end_pos)}
    assert len(analyzer_tuple) == 2
    assert len(phrase_text) == len(phrase_pos)
    
    # we don't actually need two phrase_pos? same tokenizer, same phrase list. phrase pos should be consistent
    analyzer1, analyzer2 = analyzer_tuple
    n_layers = analyzer1.n_layers + 1
    size = len(phrase_pos)
    embedding_sims = [{"cls": [], "pivot_token": [], "sep": [], "avg_all": [], "avg_phrase": []} for _ in range(n_layers)]

    for phrase_index in range(size):
        phrase = phrase_text[phrase_index]
        start_pos, end_pos = phrase_pos[phrase_index]
        sequence_length = analyzer1.lookup_sequence_length(phrase_index)
        for layer_id in range(n_layers):
            embedding1 = analyzer1.lookup_embedding(phrase_index, layer_id)
            embedding2 = analyzer2.lookup_embedding(phrase_index, layer_id)
            related_embeddings1 = extract_related_embedding(embedding1, start_pos, end_pos, sequence_length, model_name)
            related_embeddings2 = extract_related_embedding(embedding2, start_pos, end_pos, sequence_length, model_name)

            cls1, head1, sep1, avg_all1, avg_phrase1 = related_embeddings1
            cls2, head2, sep2, avg_all2, avg_phrase2 = related_embeddings2

            cls_sim = normalized_cos_sim(cls1, cls2)
            head_sim = normalized_cos_sim(head1, head2)
            sep_sim = normalized_cos_sim(sep1, sep2)
            avg_all_sim = normalized_cos_sim(avg_all1, avg_all2)
            avg_phrase_sim = normalized_cos_sim(avg_phrase1, avg_phrase2)

            embedding_sims[layer_id]["cls"].append((cls_sim, phrase))
            embedding_sims[layer_id]["pivot_token"].append((head_sim, phrase))
            embedding_sims[layer_id]["sep"].append((sep_sim, phrase))
            embedding_sims[layer_id]["avg_all"].append((avg_all_sim, phrase))
            embedding_sims[layer_id]["avg_phrase"].append((avg_phrase_sim, phrase))

    return embedding_sims


def analyze_embedding_dic(embedding_sims):
    n_layers = len(embedding_sims)
    rep_types = ["cls", "pivot_token", "sep", "avg_all", "avg_phrase"]

    mean_by_rep_type = {"cls": [], "pivot_token": [], "sep": [], "avg_all": [], "avg_phrase": []}
    variance_by_rep_type = {"cls": [], "pivot_token": [], "sep": [], "avg_all": [], "avg_phrase": []}

    for layer_id, sim_dic in enumerate(embedding_sims):
        for rep_type in rep_types:
            sim_list = sim_dic[rep_type]
            sim_list = [x[0] for x in sim_list] # extract similarity from the tuple
            mean = sum(sim_list) / len(sim_list)
            variance = var(sim_list)
            mean_by_rep_type[rep_type].append(mean)
            variance_by_rep_type[rep_type].append(variance)

    # output logic
    title = "Layer\tMean\tVariance"
    output_format = "{}\t{}\t{}"

    for rep_type in rep_types:
        mean_list = mean_by_rep_type[rep_type]
        variance_list = variance_by_rep_type[rep_type]

        assert len(mean_list) == n_layers
        assert len(variance_list) == n_layers

        print("token {}".format(rep_type))
        print(title)
        for layer_id in range(n_layers):
            mean, variance = mean_list[layer_id], variance_list[layer_id]
            print(output_format.format(layer_id, mean, variance))


def analyze_max_changes(embedding_sims):
    n_layers = len(embedding_sims)
    rep_types = ["cls", "pivot_token", "sep", "avg_all", "avg_phrase"]
    # output logic
    title = "Layer\tPhrases"
    output_format = "{}\t{}"
    for rep_type in rep_types:
        print("token {}".format(rep_type))
        print(title)
        for layer_id in range(n_layers):
            sim_list = embedding_sims[layer_id][rep_type]
            sim_list.sort(key = lambda x: x[0])
            max_changes = sim_list[:5]
            phrase_list = [x[1] for x in max_changes]
            print(output_format.format(layer_id, "///".join(phrase_list)))


def modelwise_phrase_pair_analysis(analyzer_tuple, score_dic, phrase_pos, phrase_text):
    analyzer1, analyzer2 = analyzer_tuple
    n_layers = analyzer1.n_layers + 1
    model_name = analyzer1.model_name

    max_changes_by_layer = [{"cls": [], "pivot_token": [], "sep": [], "avg_all": [], "avg_phrase": []} for _ in range(n_layers)]

    for source_phrase_index in score_dic:
        second_phrase_list = score_dic[source_phrase_index]
        source_phrase_length = analyzer1.lookup_sequence_length(source_phrase_index)
        src_start_pos, src_end_pos = phrase_pos[source_phrase_index]

        for layer_id in range(n_layers):
            source_embedding1 = analyzer1.lookup_embedding(source_phrase_index, layer_id)
            source_embedding2 = analyzer2.lookup_embedding(source_phrase_index, layer_id)
            out1 = extract_related_embedding(source_embedding1, src_start_pos, src_end_pos, source_phrase_length,
                                            model_name)
            out2 = extract_related_embedding(source_embedding2, src_start_pos, src_end_pos, source_phrase_length,
                                            model_name)
            src_cls1, src_pivot1, src_sep1, src_avg_all1, src_avg_phrase1 = out1
            src_cls2, src_pivot2, src_sep2, src_avg_all2, src_avg_phrase2 = out2

            for second_phrase_index, score in second_phrase_list:
                target_phrase_length = analyzer1.lookup_sequence_length(second_phrase_index)
                target_embedding1 = analyzer1.lookup_embedding(second_phrase_index, layer_id)
                target_embedding2 = analyzer2.lookup_embedding(second_phrase_index, layer_id)
                trg_start_pos, trg_end_pos = phrase_pos[second_phrase_index]

                out1 = extract_related_embedding(target_embedding1, trg_start_pos, trg_end_pos, target_phrase_length,
                                                model_name)
                out2 = extract_related_embedding(target_embedding2, trg_start_pos, trg_end_pos, target_phrase_length,
                                                model_name)
                trg_cls1, trg_pivot1, trg_sep1, trg_avg_all1, trg_avg_phrase1 = out1
                trg_cls2, trg_pivot2, trg_sep2, trg_avg_all2, trg_avg_phrase2 = out2

                cls_sim1 = normalized_cos_sim(src_cls1, trg_cls1)
                head_sim1 = normalized_cos_sim(src_pivot1, trg_pivot1)
                sep_sim1 = normalized_cos_sim(src_sep1, trg_sep1)
                avg_all_sim1 = normalized_cos_sim(src_avg_all1, trg_avg_all1)
                avg_phrase_sim1 = normalized_cos_sim(src_avg_phrase1, trg_avg_phrase1)

                cls_sim2 = normalized_cos_sim(src_cls2, trg_cls2)
                head_sim2 = normalized_cos_sim(src_pivot2, trg_pivot2)
                sep_sim2 = normalized_cos_sim(src_sep2, trg_sep2)
                avg_all_sim2 = normalized_cos_sim(src_avg_all2, trg_avg_all2)
                avg_phrase_sim2 = normalized_cos_sim(src_avg_phrase2, trg_avg_phrase2)

                phrase_pair = phrase_text[source_phrase_index] + "-" + phrase_text[second_phrase_index]

                max_changes_by_layer[layer_id]["cls"].append((abs(cls_sim1 - cls_sim2), phrase_pair))
                max_changes_by_layer[layer_id]["pivot_token"].append((abs(head_sim1 - head_sim2), phrase_pair))
                max_changes_by_layer[layer_id]["sep"].append((abs(sep_sim1 - sep_sim2), phrase_pair))
                max_changes_by_layer[layer_id]["avg_all"].append((abs(avg_all_sim1 - avg_all_sim2), phrase_pair))
                max_changes_by_layer[layer_id]["avg_phrase"].append((abs(avg_phrase_sim1 - avg_phrase_sim2), phrase_pair))

    rep_types = ["cls", "pivot_token", "sep", "avg_all", "avg_phrase"]
    # output logic
    title = "Layer\tPhrases"
    output_format = "{}\t{}"
    for rep_type in rep_types:
        print("token {}".format(rep_type))
        print(title)
        for layer_id in range(n_layers):
            sim_list = max_changes_by_layer[layer_id][rep_type]
            sim_list.sort(key = lambda x: x[0])
            max_changes = sim_list[-5:]
            phrase_list = [x[1] for x in max_changes]
            print(output_format.format(layer_id, "///".join(phrase_list)))

    return max_changes_by_layer



if __name__ == "__main__":
    modelwise_compare()