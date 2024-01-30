import sys
import time
import random
import argparse
import collections
import numpy as np

import torch
import torch.utils
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from utils import *
from train import *
from operation import *
from mutation import *

des_tensor=torch.load("C:/Users/User/Downloads/processed_data/des_tensor.pt")
tweets_tensor=torch.load("C:/Users/User/Downloads/processed_data/tweets_tensor.pt")
num_prop=torch.load("C:/Users/User/Downloads/processed_data/num_properties_tensor.pt")
category_prop=torch.load("C:/Users/User/Downloads/processed_data/cat_properties_tensor.pt")
edge_index=torch.load("C:/Users/User/Downloads/processed_data/edge_index.pt")
edge_relations=torch.load("C:/Users/User/Downloads/processed_data/edge_type.pt")
labels=torch.load("C:/Users/User/Downloads/processed_data/label.pt")
idx_train=torch.tensor(range(8278))
idx_val=torch.tensor(range(8278,8278+2365))
idx_test=torch.tensor(range(8278+2365,8278+2365+1183))

labels = labels.cuda()
idx_train = idx_train.cuda()
idx_val = idx_val.cuda()
idx_test = idx_test.cuda()
index = idx_train, idx_val, idx_test

data = (des_tensor, tweets_tensor, num_prop, category_prop, edge_index, edge_relations, labels)

class Model(object):
  """A class representing a model."""
  def __init__(self):
    self.arch = None
    self.val_acc = None
    self.test_acc = None
    
  def __str__(self):
    """Prints a readable version of this bitstring."""
    return self.arch

def main(cycles, population_size, sample_size):
    if not torch.cuda.is_available():
        print('no gpu device available')
        sys.exit(1)
    
    """Algorithm for regularized evolution (i.e. aging evolution)."""
    population = collections.deque()
    history = []  # Not used by the algorithm, only used to report results.
    
    # Initialize the population with random models.
    while len(population) < population_size:
        model = Model()
        model.arch = random_architecture(4)
        model.val_acc, model.test_acc = train_and_eval(model.arch, data, index)
        population.append(model)
        history.append(model)
        print(model.arch)
        print(model.val_acc, model.test_acc)
    
    # Carry out evolution in cycles. Each cycle produces a model and removes another.
    while len(history) < cycles:
        # Sample randomly chosen models from the current population.
        sample = []
        while len(sample) < sample_size:
            candidate = random.choice(list(population))
            sample.append(candidate)
        
        # The parent is the best model in the sample.
        parent = max(sample, key=lambda i: i.val_acc)
        
        # Create the child model and store it.
        child = Model()
        child.arch = mutate_arch(parent.arch, np.random.randint(0, 3))
        child.val_acc, child.test_acc = train_and_eval(child.arch, data, index)
        population.append(child)
        history.append(child)
        print(child.arch)
        print(child.val_acc, child.test_acc)
        
        # Remove the oldest model.
        population.popleft()
    
    return history

# store the search history
h = main(80, 15, 3)