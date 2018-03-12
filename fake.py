# Imports
import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn import tree

from build_sets import *
from naive_bayes import *
from logistic_classifier import *

################################################################################
# Part 1
################################################################################
def part1():
    build_sets()

################################################################################
# Part 2
################################################################################
def part2():
    training_set, validation_set, testing_set, training_label, validation_label, testing_label  = build_sets()

    word_freq_dict = NB_word_freq(training_set, training_label)

    correct = 0
    for i in range(len(training_set)):
        if NB_classifier(word_freq_dict, training_set, training_label, training_set[i]) == training_label[i]: correct += 1

    print("Training Set Accuracy: " + str(100 * correct/float(len(training_set))) + "%")

    correct = 0
    for i in range(len(validation_set)):
        if NB_classifier(word_freq_dict, training_set, training_label, validation_set[i]) == validation_label[i]: correct += 1

    print("Validation Set Accuracy: " + str(100 * correct/float(len(validation_set))) + "%")

    correct = 0
    for i in range(len(testing_set)):
        if NB_classifier(word_freq_dict, training_set, training_label, testing_set[i]) == testing_label[i]: correct += 1

    print("Testing Set Accuracy: " + str(100 * correct/float(len(testing_set))) + "%")

################################################################################
# Part 3
################################################################################
def part3():
    pass

################################################################################
# Part 4
################################################################################
def part4():
    training_set, validation_set, testing_set, training_label, validation_label, testing_label  = build_sets()

    word_index_dict, total_unique_words = word_to_index_builder(training_set, validation_set, testing_set)

    training_set_np, validation_set_np, testing_set_np, training_label_np, validation_label_np, testing_label_np = convert_sets_to_vector(training_set, validation_set, testing_set, training_label, validation_label, testing_label, word_index_dict, total_unique_words)

    model = train_LR_model(training_set_np, training_label_np, validation_set_np, validation_label_np, total_unique_words)

    # Test the model
    x_test = Variable(torch.from_numpy(testing_set_np), requires_grad=False).type(torch.FloatTensor)
    y_pred = model(x_test).data.numpy()
    test_perf = (np.mean(np.argmax(y_pred, 1) == np.argmax(testing_label_np, 1))) * 100
    print("Testing Set Performance   : " + str(test_perf)+ "%")

    # Save model weights
    np.save("LR_model.npy", model.linear.weight.data.numpy())

################################################################################
# Part 5
################################################################################
def part5():
    # Nothing to do here, check the report - fake.pdf
    pass

################################################################################
# Part 6
################################################################################
def part6():
    training_set, validation_set, testing_set, training_label, validation_label, testing_label  = build_sets()

    word_index_dict, total_unique_words = word_to_index_builder(training_set, validation_set, testing_set)

    # Load LR model weights from Part 4
    model_weights = np.load("LR_model.npy")
    model_weights = model_weights[0] - model_weights[1]

    theta_max, theta_min = [], []

    for i in range(50):
        theta_max.append(np.argmax(model_weights))
        theta_min.append(np.argmin(model_weights))
        model_weights[theta_max[-1]] = 0
        model_weights[theta_min[-1]] = 0

    print("Top 10 positive thetas (including stop-words): ")
    count = 0
    for theta in theta_max:
        if count >= 10: break
        word = [word for word, index in word_index_dict.items() if index == theta][0]
        print(str(i+1)+ ": " + word)
        count += 1

    print("\n")

    print("Top 10 negative thetas (including stop-words): ")
    count = 0
    for theta in theta_min:
        if count >= 10: break
        word = [word for word, index in word_index_dict.items() if index == theta][0]
        print(str(i+1)+ ": " + word)
        count += 1

    print("\n")

    print("Top 10 positive thetas (excluding stop-words): ")
    count = 0
    for theta in theta_max:
        if count >= 10: break
        word = [word for word, index in word_index_dict.items() if index == theta][0]
        if word in ENGLISH_STOP_WORDS: continue
        print(str(i+1)+ ": " + word)
        count += 1

    print("\n")

    print("Top 10 negative thetas (excluding stop-words): ")
    count = 0
    for theta in theta_min:
        if count >= 10: break
        word = [word for word, index in word_index_dict.items() if index == theta][0]
        if word in ENGLISH_STOP_WORDS: continue
        print(str(i+1)+ ": " + word)
        count += 1

################################################################################
# Part 7
################################################################################
#def part7():
training_set, validation_set, testing_set, training_label, validation_label, testing_label  = build_sets()

word_index_dict, total_unique_words = word_to_index_builder(training_set, validation_set, testing_set)

training_set_np, validation_set_np, testing_set_np, training_label_np, validation_label_np, testing_label_np = convert_sets_to_vector(training_set, validation_set, testing_set, training_label, validation_label, testing_label, word_index_dict, total_unique_words)

max_depth_val = [2, 5, 10, 20, 50, 75, 100, 150, 200, 500]

for d in max_depth_val:
    clf = tree.DecisionTreeClassifier(max_depth=d)
    clf = clf.fit(training_set_np, training_label)

    print("Depth: " + str(d))
    print("Training Set Accuracy  : " + str(100*clf.score(training_set_np, training_label)))
    print("Validation Set Accuracy: " + str(100*clf.score(validation_set_np, validation_label)))
    print("\n")

# Best performance comes at max_depth=150
clf = tree.DecisionTreeClassifier(max_depth=150)
clf = clf.fit(training_set_np, training_label)

# Visualize first two layers of decision tree
dot_data = tree.export_graphviz(clf, out_file="figures/decision_tree.dot", max_depth=2, filled=True, rounded=True, class_names=['fake', 'real'])

################################################################################
# Part 8
################################################################################
def part8():
    pass
