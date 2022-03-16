import torch
from ipdb import set_trace
import pandas as pd

class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, dataset):
        # if indices is not provided, all elements in the dataset will be considered
        self.indices = list(range(len(dataset)))

        self.num_samples = len(self.indices)
        
        self.weights = 1/ self.num_samples*torch.ones(self.num_samples)

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))