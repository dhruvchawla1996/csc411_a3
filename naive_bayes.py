# Imports

# def NB_classifier(training_set, training_label, all_words, input_words):
#     n = len(words)
#     count_fake = 0
#     for label in training_label:
#         if label == 0: count_fake = count_fake + 1

#     count_real = len(training_label) - count_fake

#     P_fake = count_fake/len(training_label)
#     P_n_fake = 1 - P_fake

#     for word in all_words:
#         count_fake_word, count_real_word = 0, 0

#         for i in range(training_set):
#             if word in training_set[i]: 
#                 if   training_label[i] == 0: count_fake_word = count_fake_word + 1
#                 elif training_label[i] == 1: count_real_word = count_real_word + 1

#             P_word_fake = count_fake_word/count_fake
#             P_n_word_fake = 1 - P_word_fake

#             P_word_real = count_real_word/count_real
#             P_n_word_real = 1 - P_word_real

"""
Build a dictionary of words in fake and real training set examples
With values corresponding to their number of appearences in fake and real training set examples

PARAMETERS
----------

training_set: list of list of strings
    contains headlines broken into words

training_label: list of 0 or 1
    0 = fake news | 1 = real news for corresponding i-th element in training_set

RETURNS
-------
word_freq_dict = {string, (int, int)}
    value[0] = appearances in fake news headlines | value[1] = appearances in real news headlines
"""
def NB_word_freq(training_set, training_label):
    word_freq_dict = {}

    for i in range(len(training_set)):
        for word in training_set[i]:
            if word not in word_freq_dict: 
                if   training_label[i] == 0: word_freq_dict[word] = (1, 0)
                elif training_label[i] == 1: word_freq_dict[word] = (0, 1)
            else:
                if   training_label[i] == 0: word_freq_dict[word] = (word_freq_dict[word][0]+1, word_freq_dict[word][1]  )
                elif training_label[i] == 1: word_freq_dict[word] = (word_freq_dict[word][0]  , word_freq_dict[word][1]+1)

    return word_freq_dict