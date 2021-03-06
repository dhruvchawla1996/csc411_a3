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

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
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
def part3a():
    training_set, validation_set, testing_set, training_label, validation_label, testing_label = build_sets()
    word_freq_dict = NB_word_freq(training_set, training_label)

    #note: all_words are unique
    words = np.array([])
    P_fake_given_word = np.array([])
    P_fake_given_not_word = np.array([])
    P_real_given_word = np.array([])
    P_real_given_not_word = np.array([])
    P_word_given_fake = np.array([])
    P_word_given_real =np.array([])
    P_not_word_given_real = np.array([])
    P_not_word_given_fake = np.array([])
    P_word = np.array([])
    P_not_word = np.array([])

    count_fake = 0
    for label in training_label:
        if label == 0: count_fake = count_fake + 1

    count_real = len(training_label) - count_fake

    for word, freq in word_freq_dict.iteritems():
        P_word_given_fake = np.append(P_word_given_fake , float((freq[0])) / float(count_fake))
        P_word_given_real = np.append(P_word_given_real , float((freq[1])) / float(count_real))
        P_word = np.append(P_word, float((freq[0] + freq[1]))/ len(training_label))

    P_not_word_given_fake = 1 - P_word_given_fake
    P_not_word_given_real = 1 - P_word_given_real
    P_not_word = 1 - P_word #

    ####### compute P( class | ~word) for all words #######
    #P(real | ~word) = (P_~word_| class * P(class) ) / P(~ word)

    P_real_given_not_word = ( P_word_given_real / P_not_word )
    P_fake_given_not_word = ( P_word_given_fake / P_not_word )

    ####### get P(class | word) for all words #######
    for word in word_freq_dict.keys():
        words = np.append(words, word)
        P_fake_given_word = np.append(P_fake_given_word,
                                      NB_probabilities(word_freq_dict, training_set, training_label, [word]))

    P_real_given_word = 1 - P_fake_given_word

    #print(words.shape, P_word_given_fake.shape, P_word_given_real.shape, P_fake_given_word.shape,
          P_real_given_word.shape, P_word.shape, P_not_word.shape, P_real_given_not_word.shape, P_fake_given_not_word.shape)

    #create maps from words to the four probabilities
    fake_under_presence = np.vstack((words, P_fake_given_word)).T
    fake_under_absence = np.vstack((words, P_fake_given_not_word)).T
    real_under_presence = np.vstack((words, P_real_given_word)).T
    real_under_absence = np.vstack((words, P_real_given_not_word)).T


    # print(fake_under_presence.shape, fake_under_absence.shape, real_under_presence.shape, real_under_absence.shape)

    # #sort by 2nd column
    fake_under_presence = fake_under_presence[fake_under_presence[:, 1].argsort()]
    fake_under_absence = fake_under_absence[fake_under_absence[:,1].argsort()]
    real_under_presence = real_under_presence[real_under_presence[:,1].argsort()]
    real_under_absence = real_under_absence[real_under_absence[:,1].argsort()]

    np.savetxt("fake_under_presence.txt", fake_under_presence, fmt = '%s')
    np.savetxt("fake_under_absence.txt", fake_under_absence, fmt = "%s")
    np.savetxt("real_under_presence.txt", real_under_presence, fmt="%s")
    np.savetxt("real_under_absence.txt", real_under_absence, fmt="%s")




def part3b():
    training_set, validation_set, testing_set, training_label, validation_label, testing_label = build_sets()
    word_freq_dict = NB_word_freq(training_set, training_label)

    #TODO: modify
    for word in ENGLISH_STOP_WORDS:
        if word in word_freq_dict: del word_freq_dict[word]

    #TODO: repeat part 3a

    return


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
        print(str(count+1)+ ": " + word)
        count += 1

    print("\n")

    print("Top 10 negative thetas (including stop-words): ")
    count = 0
    for theta in theta_min:
        if count >= 10: break
        word = [word for word, index in word_index_dict.items() if index == theta][0]
        print(str(count+1)+ ": " + word)
        count += 1

    print("\n")

    print("Top 10 positive thetas (excluding stop-words): ")
    count = 0
    for theta in theta_max:
        if count >= 10: break
        word = [word for word, index in word_index_dict.items() if index == theta][0]
        if word in ENGLISH_STOP_WORDS: continue
        print(str(count+1)+ ": " + word)
        count += 1

    print("\n")

    print("Top 10 negative thetas (excluding stop-words): ")
    count = 0
    for theta in theta_min:
        if count >= 10: break
        word = [word for word, index in word_index_dict.items() if index == theta][0]
        if word in ENGLISH_STOP_WORDS: continue
        print(str(count+1)+ ": " + word)
        count += 1

################################################################################
# Part 7
################################################################################
def part7():
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

    index_word_dict = {index: word for word, index in word_index_dict.iteritems()}
    word_list = []
    for i in range(total_unique_words):
        word_list.append(index_word_dict[i])

    # Best performance comes at max_depth=150
    clf = tree.DecisionTreeClassifier(max_depth=150)
    clf = clf.fit(training_set_np, training_label)
    #TODO: experiment with other non-default parameters

    # Visualize first two layers of decision tree
    dot_data = tree.export_graphviz(clf, out_file="figures/decision_tree.dot", max_depth=2, filled=True, rounded=True, class_names=['fake', 'real'], feature_names=word_list)

################################################################################
# Part 8
################################################################################
def part8():
    training_set, validation_set, testing_set, training_label, validation_label, testing_label  = build_sets()

    word_index_dict, total_unique_words = word_to_index_builder(training_set, validation_set, testing_set)

    training_set_np, validation_set_np, testing_set_np, training_label_np, validation_label_np, testing_label_np = convert_sets_to_vector(
        training_set, validation_set, testing_set, training_label, validation_label, testing_label, word_index_dict,
        total_unique_words)


    #print(training_set_np.shape)
    #print(word_index_dict)
    # count_fake = 0
    # for label in training_label:
    #     if label == 0: count_fake = count_fake + 1
    #
    # count_real = len(training_label) - count_fake
    #
    # P_fake = count_fake / float(len(training_label))
    # P_real = 1. - P_fake

    #print("P_fake = %s" %P_fake, "P_real = %s" %P_real)


    #get P(fake | ~word) from training.np (size = 2287 x 5832 = #examples x #uniquewords)
    #P(fake | ~word) = P( fake and ~word) / P(word)

###################### MAIN #############################
part3a()