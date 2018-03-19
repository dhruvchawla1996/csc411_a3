# Imports
from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import time
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
from scipy.ndimage import filters
import urllib
from numpy import random

import os

def plot_learning_curves(part, epoch, train_perf, valid_perf):
    '''Plot learning curves for training and testing set w.r.t epoch
    
    part        String      "part4"
    epoch       list(Int)   epoch
    train_perf  list(Int)   performance on training set in % with each element corresponding to epoch
    train_perf  list(Int)   performance on validation set in % with each element corresponding to epoch

    Plots and saves figure in "figure/part4_learning_curve.png"
    '''
    plt.plot(epoch, train_perf, color='k', linewidth=2, marker="o", label="Training Set")
    plt.plot(epoch, valid_perf, color='b', linewidth=2, marker="o", label="Validation Set")

    plt.title("Learning curve")
    plt.xlabel("Epoch")
    plt.ylabel("Performance (%)")
    plt.legend()
    plt.savefig("figures/" + part + "_learning_curve.png")