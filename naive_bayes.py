# Imports
import math

def small_multiply(numbers):
    """
    Multiply a lot of small numbers (values may be close to 0)

    PARAMETERS
    ----------
    numbers: list of floats 
        elements should be in (0, 1]

    RETURNS
    -------
    float
        product of all numbers
    """
    log_sum = 0.

    for n in numbers:
        log_sum += math.log(n)

    return math.exp(log_sum)

def NB_classifier(word_freq_dict, training_set, training_label, words):
    """
    Classify a headline as fake or real news

    PARAMETERS
    ----------
    word_freq_dict: {string, (int, int)}
        value[0] = appearances in fake news headlines | value[1] = appearances in real news headlines

    training_set: list of list of strings
        contains headlines broken into words

    training_label: list of 0 or 1
        0 = fake news | 1 = real news for corresponding i-th element in training_set

    words: list[string]
        headline seperated into list of strings

    RETURNS
    -------
    0 or 1
        0 = fake news | 1 = real news
    """
    n = len(words)
    count_fake = 0
    for label in training_label:
        if label == 0: count_fake = count_fake + 1

    count_real = len(training_label) - count_fake

    P_fake = count_fake/float(len(training_label))
    P_real = 1. - P_fake

    P_word_fake, P_word_real = [], []
    #playwith m =1 and p = 0.1
    for word, freq in word_freq_dict.iteritems():
        P_word_i_fake = (freq[0]+1*0.1)/float(count_fake+1)
        P_word_i_real = (freq[1]+1*0.1)/float(count_real+1)

        if word in words:
            P_word_fake.append(P_word_i_fake)
            P_word_real.append(P_word_i_real)
        elif word not in words:
            P_word_fake.append(1. - P_word_i_fake)
            P_word_real.append(1. - P_word_i_real)

    P_fake_words = P_fake * small_multiply(P_word_fake)
    P_real_words = P_real * small_multiply(P_word_real)

    #/P_fake + P_real?
    P_fake_words = P_fake_words/(P_fake_words + P_real_words)

    return 0 if P_fake_words > 0.5 else 1

def NB_probabilities(word_freq_dict, training_set, training_label, words):
    """
    Classify a headline as fake or real news

    PARAMETERS
    ----------
    word_freq_dict: {string, (int, int)}
        value[0] = appearances in fake news headlines | value[1] = appearances in real news headlines

    training_set: list of list of strings
        contains headlines broken into words

    training_label: list of 0 or 1
        0 = fake news | 1 = real news for corresponding i-th element in training_set

    words: list[string]
        previously, headline seperated into list of strings. Intended to use as just a list of one word

    RETURNS
    -------
    P_fake_words = P(fake | words)
    """
    n = len(words)
    count_fake = 0
    for label in training_label:
        if label == 0: count_fake = count_fake + 1

    count_real = len(training_label) - count_fake

    P_fake = count_fake/float(len(training_label))
    P_real = 1. - P_fake

    P_word_fake, P_word_real = [], []
    #TODO: playwith m =1 and p = 0.1
    for word, freq in word_freq_dict.iteritems():
        P_word_i_fake = (freq[0]+1*0.1)/float(count_fake+1)
        P_word_i_real = (freq[1]+1*0.1)/float(count_real+1)

        if word in words:
            P_word_fake.append(P_word_i_fake)
            P_word_real.append(P_word_i_real)
        elif word not in words:
            P_word_fake.append(1. - P_word_i_fake)
            P_word_real.append(1. - P_word_i_real)

    P_fake_words = P_fake * small_multiply(P_word_fake)
    P_real_words = P_real * small_multiply(P_word_real)

    P_fake_words = P_fake_words/(P_fake_words + P_real_words) #probability that news if fake given collection of words (headline)
    #P_real_words = 1 - P_fake_words
    return P_fake_words

def NB_word_freq(training_set, training_label):
    """
    Build a dictionary of words in fake and real training set examples
    With values corresponding to their number of appearences in fake and real training set examples
    Note: If a word appears more than once in a headline, it's still counted as 1

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
    word_freq_dict = {}

    for i in range(len(training_set)):
        headline = set(training_set[i])
        for word in headline:
            if word not in word_freq_dict: 
                if   training_label[i] == 0: word_freq_dict[word] = (1, 0)
                elif training_label[i] == 1: word_freq_dict[word] = (0, 1)
            else:
                if   training_label[i] == 0: word_freq_dict[word] = (word_freq_dict[word][0]+1, word_freq_dict[word][1]  )
                elif training_label[i] == 1: word_freq_dict[word] = (word_freq_dict[word][0]  , word_freq_dict[word][1]+1)

    return word_freq_dict