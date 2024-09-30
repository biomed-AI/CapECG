import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset
from sklearn import datasets
from sklearn import preprocessing
import random


class SNP_ECG_dataset():
    def __init__(self, geno_data, pheno_df, feature):
        self.X_data = geno_data
        self.y_data=pheno_df[feature].tolist()
        self.len = len(self.y_data)
        self.y_data = (self.y_data-np.mean(self.y_data))/np.std(self.y_data)
        q3 = np.percentile(self.y_data, 75)
        self.y_discrete_data= []
        for value in self.y_data:
            if value <= q3:
                self.y_discrete_data.append([0, 1])  
            else:
                self.y_discrete_data.append([1, 0])  
        self.y_discrete_data = np.array(self.y_discrete_data)

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index], self.y_discrete_data[index]

    def __len__(self):
        return self.len

