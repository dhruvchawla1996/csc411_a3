# Imports
import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np

# Model
class LogisticRegression(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)
    
    def forward(self, x):
        out = self.linear(x)
        return out

def train_LR_model(training_set, training_label, validation_set, validation_label, total_unique_words):
    """
    Trains Logistic Regression Numpy model

    PARAMETERS
    ----------
    training_set, validation_set: numpy arrays [num_examples, total_unique_words]
        For each headline in a set:
        v[k] = 1 if kth word appears in the headline else 0

    training_label, validation_label: numpy arrays [num_examples, [0, 1] or [1, 0]]
        [0, 1] if headline is fake else [1, 0]

    total_unique_words: int
        total number of unique words in training_set, validation_set, testing_set

    RETURNS
    -------
    model: LogisticRegression instance
        fully trained Logistic Regression model

    REQUIRES
    --------
    LogisticRegression: PyTorch class defined
    """
    # Hyper Parameters 
    input_size = total_unique_words
    num_classes = 2
    num_epochs = 800
    learning_rate = 0.001
    reg_lambda = 0.01

    model = LogisticRegression(input_size, num_classes)

    x = Variable(torch.from_numpy(training_set), requires_grad=False).type(torch.FloatTensor)
    y_classes = Variable(torch.from_numpy(np.argmax(training_label, 1)), requires_grad=False).type(torch.LongTensor)

    # Loss and Optimizer
    # Softmax is internally computed.
    # Set parameters to be updated.
    loss_fn = nn.CrossEntropyLoss()  
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  

    # Training the Model
    for epoch in range(num_epochs+1):
        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = model(x)
        l2_reg = Variable(torch.FloatTensor(1), requires_grad=True)
        for W in model.parameters():
            l2_reg = l2_reg + W.norm(2)
        loss = loss_fn(outputs, y_classes) + reg_lambda*l2_reg
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print("Epoch: " + str(epoch))

            # Training Performance
            x_train = Variable(torch.from_numpy(training_set), requires_grad=False).type(torch.FloatTensor)
            y_pred = model(x_train).data.numpy()
            train_perf_i = (np.mean(np.argmax(y_pred, 1) == np.argmax(training_label, 1))) * 100
            print("Training Set Performance  : " + str(train_perf_i) + "%")      

            # Validation Performance  
            x_valid = Variable(torch.from_numpy(validation_set), requires_grad=False).type(torch.FloatTensor)
            y_pred = model(x_valid).data.numpy()
            valid_perf_i = (np.mean(np.argmax(y_pred, 1) == np.argmax(validation_label, 1))) * 100
            print("Validation Set Performance:  " + str(valid_perf_i) + "%\n")

    return model

def convert_sets_to_vector(training_set, validation_set, testing_set, training_label, validation_label, testing_label, word_index_dict, total_unique_words):
    """
    Convert training, validation and testing sets and labels from lists to numpy vectors 

    For each headline in a set:
    v[k] = 1 if kth word appears in the headline else 0

    For each label:
    [0, 1] if headline is fake else [1, 0]

    PARAMETERS
    ----------
    training_set, validation_set, testing_set: list of list of strings
        contains headlines broken into words

    training_label, validation_label, testing_label: list of 0 or 1
        0 = fake news | 1 = real news for corresponding i-th element in training_set

    word_index_dict: {string, int}
        Matches word in training_set, validation_set, testing_set to a unique count number

    total_unique_words: int
        total number of unique words in training_set, validation_set, testing_set

    RETURNS
    -------
    training_set_np, validation_set_np, testing_set_np: 
        numpy arrays representing conversions of training_set, validation_set, testing_set respectively

    training_label_np, validation_label_np, testing_label_np:
        numpy arrays representing conversions of training_label, validation_label, testing_label respectively
    """
    training_set_np, validation_set_np, testing_set_np = np.zeros((0, total_unique_words)), np.zeros((0, total_unique_words)), np.zeros((0, total_unique_words))

    # Training Set ############################################################
    for headline in training_set:
        training_set_i = np.zeros(total_unique_words)

        for word in headline:
            training_set_i[word_index_dict[word]] = 1.

        training_set_i = np.reshape(training_set_i, [1, total_unique_words])
        training_set_np = np.vstack((training_set_np, training_set_i))

    training_label_np = np.asarray(training_label).transpose()
    training_label_np_complement = 1 - training_label_np
    training_label_np = np.vstack((training_label_np, training_label_np_complement)).transpose()

    # Validation Set ############################################################
    for headline in validation_set:
        validation_set_i = np.zeros(total_unique_words)

        for word in headline:
            validation_set_i[word_index_dict[word]] = 1.

        validation_set_i = np.reshape(validation_set_i, [1, total_unique_words])
        validation_set_np = np.vstack((validation_set_np, validation_set_i))

    validation_label_np = np.asarray(validation_label).transpose()
    validation_label_np_complement = 1 - validation_label_np
    validation_label_np = np.vstack((validation_label_np, validation_label_np_complement)).transpose()

    # Testing Set ############################################################
    for headline in testing_set:
        testing_set_i = np.zeros(total_unique_words)

        for word in headline:
            testing_set_i[word_index_dict[word]] = 1.

        testing_set_i = np.reshape(testing_set_i, [1, total_unique_words])
        testing_set_np = np.vstack((testing_set_np, testing_set_i))

    testing_label_np = np.asarray(testing_label).transpose()
    testing_label_np_complement = 1 - testing_label_np
    testing_label_np = np.vstack((testing_label_np, testing_label_np_complement)).transpose()

    return training_set_np, validation_set_np, testing_set_np, training_label_np, validation_label_np, testing_label_np

def word_to_index_builder(training_set, validation_set, testing_set):
    """
    Build a mapping of word -> unique number

    PARAMETERS
    ----------
    training_set: list of list of strings
        contains headlines broken into words

    validation_set: list of list of strings
        contains headlines broken into words

    testing_set: list of list of strings
        contains headlines broken into words

    RETURNS
    -------
    word_dict: {string, int}
        Matches word in training_set, validation_set, testing_set to a unique count number

    total_unique_words: int
        total number of unique words in training_set, validation_set, testing_set
    """
    word_dict = {}
    i = 0

    for headline in training_set+validation_set+testing_set:
        for word in headline:
            if word not in word_dict: 
                word_dict[word] = i
                i += 1

    return word_dict, i
