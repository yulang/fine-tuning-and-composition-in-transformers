from sklearn import linear_model
import numpy as np
import pdb

from preprocessing import load_paws
from analyze_model import get_swapping_pos


def transform_loader(dataloader, include_absolute_distance=False):
    # convert raw data from dataloader to swapping distance input
    input_array, label_array = [], []
    for sentence_batch, label_batch in dataloader:
        for index, label in enumerate(label_batch):
            sent1, sent2 = sentence_batch[0][index], sentence_batch[1][index]
            # distance = get_swapping_pos(sent1, sent2, normalized=True)
            # if include_absolute_distance:
            absolute_distance = get_swapping_pos(sent1, sent2)
            relative_distance = get_swapping_pos(sent1, sent2, normalized=True)
            if include_absolute_distance:
                input_array.append((absolute_distance, relative_distance))
            else:
                input_array.append((relative_distance,))
            label_array.append(label.item())
    return input_array, label_array


paws_location = ""
train_loader, dev_loader = load_paws(paws_location)

print("=== w/ relative distance only ===")
# build training and testing data
train_input, train_label = transform_loader(train_loader)
test_input, test_label = transform_loader(dev_loader)

baseline_clf = linear_model.RidgeClassifier().fit(train_input, train_label)
train_acc = baseline_clf.score(train_input, train_label)
print(train_acc)
test_acc = baseline_clf.score(test_input, test_label)
print(test_acc)

print("=== w/ relative and absolute distance ===")
# build training and testing data
train_input, train_label = transform_loader(train_loader, include_absolute_distance=True)
test_input, test_label = transform_loader(dev_loader, include_absolute_distance=True)

baseline_clf = linear_model.RidgeClassifier().fit(train_input, train_label)
train_acc = baseline_clf.score(train_input, train_label)
print(train_acc)
test_acc = baseline_clf.score(test_input, test_label)
print(test_acc)
