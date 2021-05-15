from train import *
from preprocessing import load_paws
from model_io import load_model, get_pretrained_name
from collections import Counter
import pickle
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import pdb

DUMP_DIR = "bert-dump/prediction_analysis"


# ----- prediction generation logic ------ #
def generate_predictions(dump_out_loc):
    # pretrained_name = get_pretrained_name("bert")
    paws_location = ""
    models_dir = ""

    model_locs = {
        "bert": os.path.join(models_dir, "fine_tuned_bert_balanced_paws.pt"),
        "roberta": os.path.join(models_dir, "fine_tuned_roberta_balanced_paws.pt"),
        "distillbert": os.path.join(models_dir, "fine_tuned_distillbert_balanced_paws.pt"),
        "xlmroberta": os.path.join(models_dir, "fine_tuned_xlmroberta_balanced_paws.pt"),
        "xlnet": os.path.join(models_dir, "fine_tuned_xlnet_balanced_paws.pt"),
    }
    train_loader, dev_loader = load_paws(paws_location, balanced=True)

    for model_name, model_path in model_locs.items():
        pretrained_name = get_pretrained_name(model_name)
        model, tokenizer = load_model(model_loc=model_path, load_tuned=True, pretrained_name=pretrained_name)
        pickle_out_loc = os.path.join(dump_out_loc, model_name)
        generate_model_prediction(model, tokenizer, dev_loader, pickle_out_loc)


def generate_model_prediction(model, tokenizer, test_loader, pickle_loc):
    precision, (errors, corrects) = evaluate(model, tokenizer, test_loader, "paws", error_analysis=True)
    Path(pickle_loc).mkdir(parents=True, exist_ok=True)
    print(precision)
    err_path = os.path.join(pickle_loc, "errors.pt")
    correct_path = os.path.join(pickle_loc, "corrects.pt")

    error_handler = open(err_path, "wb")
    pickle.dump(errors, error_handler)
    correct_handler = open(correct_path, "wb")
    pickle.dump(corrects, correct_handler)

# ----- prediction generation logic end ------ #

# ----- plotting logic ----- #
def generate_subplots():
    ax1 = plt.subplot2grid(shape=(2,6), loc=(0,0), colspan=2)
    ax2 = plt.subplot2grid((2,6), (0,2), colspan=2)
    ax3 = plt.subplot2grid((2,6), (0,4), colspan=2)
    # ax4 = plt.subplot2grid((2,6), (1,1), colspan=2)
    # ax5 = plt.subplot2grid((2,6), (1,3), colspan=2)
    ax4 = plt.subplot2grid((2,6), (1,0), colspan=2)
    ax5 = plt.subplot2grid((2,6), (1,2), colspan=2)
    ax6 = plt.subplot2grid((2,6), (1,4), colspan=2)
    return [ax1, ax2, ax3, ax4, ax5, ax6]


def plot_histogram(err_pos_counter, err_neg_counter, correct_pos_counter, correct_neg_counter, ax, normalized=False):
    colors = ['red', 'blue']
    pos_prediction = correct_pos_counter + err_neg_counter
    neg_prediction = correct_neg_counter + err_pos_counter

    pos_prediction_elements = list(pos_prediction.elements())
    neg_prediction_elements = list(neg_prediction.elements())
    pos_counts, pos_bins = np.histogram(pos_prediction_elements)
    neg_counts, neg_bins = np.histogram(neg_prediction_elements)

    if normalized:
        pos_weights = pos_counts / float(sum(pos_counts))
        neg_weights = neg_counts / float(sum(neg_counts))
    else:
        pos_weights = pos_counts
        neg_weights = neg_counts

    n, bins, patches = ax.hist([pos_bins[:-1], neg_bins[:-1]], pos_bins, weights=[pos_weights, neg_weights], color=colors, label=["pos_prediction", "neg_prediction"],fill=True, histtype='bar')

    # plt.setp(patches[0], hatch='/')
    for patch in patches[0].patches:
        patch.set_hatch('/')
    for patch in patches[1].patches:
        patch.set_hatch('*')



def plot_dataset_stats(pos_counter, neg_counter, ax):
    colors = ['red', 'blue']
    pos_elements = list(pos_counter.elements())
    neg_elements = list(neg_counter.elements())

    pos_counts, pos_bins = np.histogram(pos_elements)
    neg_counts, neg_bins = np.histogram(neg_elements)

    ax.hist([pos_bins[:-1], neg_bins[:-1]], pos_bins, weights=[pos_counts, neg_counts], color=colors, label=["pos_sample", "neg_sample"])
    ax.set_title("Test set statistics")


def plot_all_models(models, model_counter_list, axs):
    # model_counter_list should be list of shape [#model, 4]
    # where the second dim corresponds to err_pos, err_neg, correct_pos, correct_neg

    for model_index, model in enumerate(models):
        cur_ax = axs[model_index]
        cur_ax.set_title(model)
        err_pos_counter, err_neg_counter, correct_pos_counter, correct_neg_counter = model_counter_list[model_index]
        plot_histogram(err_pos_counter, err_neg_counter, correct_pos_counter, correct_neg_counter, cur_ax)

# ----- plotting logic end ----- #

# ----- analysis logic ----- #

def get_swapping_pos(sent1, sent2, normalized=False):
    # if normalized is True, the distance is normalized by sentence length
    words1, words2 = sent1.split(), sent2.split()
    size = min(len(words1), len(words2))
    diff_pos = [i for i in range(size) if words1[i] != words2[i]]
    if len(diff_pos) == 0:
        return 0
    anchor_pos = diff_pos[0]
    anchor_word = words1[anchor_pos]

    try:
        distance = words2[anchor_pos + 1:].index(anchor_word) + 1
    except ValueError:
        return 0
        
    assert distance > 0
    if normalized:
        distance = float(distance) / size

    return distance

def get_swapping_pos_v0(sent1, sent2):
    words1, words2 = sent1.split(), sent2.split()
    size = min(len(words1), len(words2))
    diff_pos = [i for i in range(size) if words1[i] != words2[i]]
    return diff_pos


def get_swapping_stat(counter):
    elements = list(counter.elements())
    count = len(elements)
    distance_sum = sum(elements)

    avg_distance = distance_sum / float(count)

    return avg_distance


def analyze_model_prediction(dump_dir):
    error_loc = os.path.join(dump_dir, "errors.pt")
    correct_loc = os.path.join(dump_dir, "corrects.pt")
    error_handler = open(error_loc, "rb")
    correct_handler = open(correct_loc, "rb")

    errors = pickle.load(error_handler)
    corrects = pickle.load(correct_handler)

    # positive/negative count
    err_sent1, err_sent2, err_label = [], [], []
    correct_sent1, correct_sent2, correct_label = [], [], []
    for t in errors:
        err_sent1.append(t[0])
        err_sent2.append(t[1])
        err_label.append(t[2])

    for t in corrects:
        correct_sent1.append(t[0])
        correct_sent2.append(t[1])
        correct_label.append(t[2])
    
    err_pos_count, err_neg_count = err_label.count(1), err_label.count(0)
    correct_pos_count, correct_neg_count = correct_label.count(1), correct_label.count(0)

    print("in errors")
    print("pos count: {}, neg count: {}".format(err_pos_count, err_neg_count))
    print("in correct")
    print("pos count: {}, neg count: {}".format(correct_pos_count, correct_neg_count))

    # swapping distance

    err_pos_counter, err_neg_counter = Counter(), Counter()
    correct_pos_counter, correct_neg_counter = Counter(), Counter()

    for ind in range(len(err_sent1)):
        # swap_pos = get_swapping_pos(err_sent1[ind], err_sent2[ind])
        # distance = swap_pos[1] - swap_pos[0]
        distance = get_swapping_pos(err_sent1[ind], err_sent2[ind], normalized=True)
        if err_label[ind] == 0:
            # negative sample
            err_neg_counter[distance] += 1
        else:
            # positive sample
            err_pos_counter[distance] += 1

    for ind in range(len(correct_sent1)):
        # swap_pos = get_swapping_pos(correct_sent1[ind], correct_sent2[ind])
        # distance = swap_pos[1] - swap_pos[0]
        distance = get_swapping_pos(correct_sent1[ind], correct_sent2[ind], normalized=True)
        if correct_label[ind] == 0:
            # negative sample
            correct_neg_counter[distance] += 1
        else:
            correct_pos_counter[distance] += 1

    overall_avg_swap = get_swapping_stat(err_neg_counter + err_pos_counter + correct_pos_counter + correct_neg_counter)
    err_avg_swap = get_swapping_stat(err_neg_counter + err_pos_counter)
    correct_avg_swap = get_swapping_stat(correct_pos_counter + correct_neg_counter)
    pos_prediction_avg_swap = get_swapping_stat(correct_pos_counter + err_neg_counter)
    neg_prediction_avg_swap = get_swapping_stat(correct_neg_counter + err_pos_counter)

    print(overall_avg_swap, err_avg_swap, correct_avg_swap, pos_prediction_avg_swap, neg_prediction_avg_swap)

    # plot_histogram(err_pos_counter, err_neg_counter, correct_pos_counter, correct_neg_counter)
    return err_pos_counter, err_neg_counter, correct_pos_counter, correct_neg_counter

    
# ----- analysis logic end ----- #

# entrance of the analysis
# to generate prediciton dump, call generate_predictions
def main():
    img_out = "img/histogram.eps"
    dump_dir_name = ["bert", "roberta", "distillbert", "xlmroberta", "xlnet"]
    model_names = ["BERT", "RoBERTa", "DistilBERT", "XLM-RoBERTa", "XLNet"]
    model_counter_list = []
    for index, model_name in enumerate(model_names):
        # print("processing {}".format(model_name))
        dump_dir = os.path.join(DUMP_DIR, dump_dir_name[index])
        err_pos_counter, err_neg_counter, correct_pos_counter, correct_neg_counter = analyze_model_prediction(dump_dir)
        model_counter_list.append([err_pos_counter, err_neg_counter, correct_pos_counter, correct_neg_counter])
    
    print("plotting...")
    plt.clf()
    matplotlib.rc('font', size=40)
    fig = plt.figure(figsize=(35, 20))
    axs = generate_subplots()

    plot_all_models(model_names, model_counter_list, axs[:5])
    plot_dataset_stats(err_pos_counter + correct_pos_counter, err_neg_counter + correct_neg_counter, axs[-1])

    axs[0].set_ylabel("#samples")
    axs[3].set_ylabel("#samples")
    for ax in axs:
        ax.set_xlabel("swap distance / sentence length")
        ax.set_ylim(0, 52)
    fig.tight_layout(pad=1.0)
    handles, labels = axs[1].get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor=(0, 0, 1, 1), ncol=5, loc='lower center', borderaxespad=-0.5)
    # plt.savefig(img_out, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    # generate_predictions(DUMP_DIR)
    main()
    # generate_model_prediction()
    # analyze_model_prediction()
    # print(get_swapping_pos("she is so good .", "good is so she ."))