# Imports
from build_sets import *
from naive_bayes import *

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
    pass

################################################################################
# Part 5
################################################################################
def part5():
    pass

################################################################################
# Part 6
################################################################################
def part6():
    pass

################################################################################
# Part 7
################################################################################
def part7():
    pass

################################################################################
# Part 8
################################################################################
def part8():
    pass