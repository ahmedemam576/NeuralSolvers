# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 17:01:44 2021

@author: 49174
"""


import sys
import os
import h5py
import torch
import numpy as np
from skimage.restoration import denoise_bilateral
from skimage.filters import sobel
from skimage.segmentation import watershed
from scipy.ndimage import binary_fill_holes
from argparse import ArgumentParser
from torch.utils.data import Dataset

class BoundaryConditionsDataSet(Dataset):
    def __init__(self, nb, lb, ub):
        """
        Constructor of the initial condition dataset
        """
        raise NotImplementedError

    def __getitem__(self, idx):
        """
        Returns data for initial state
        """
        raise NotImplementedError

    def __len__(self):
        """
        Length of the dataset
        """
        raise NotImplementedError
    
class InitalConditionDataset(Dataset):
    def __init__(self,  pData, batchSize, numBatches, nx=640,
                 ny=480, useGPU=False,grid_size=0.0125)
    """
        Constructor of the initial condition dataset
        __getitem()__ returns a batch with x,y,t to compute u_predicted value at as well as u_exact
        """
        self.u = []  # distance
        self.x = []
        self.y = []
        self.t = []
        if not os.path.exists(pData):  # Check if given path to data is valid
            raise FileNotFoundError('Could not find file' + pData)
        data= np.load(pData)
        self.x = data['x'] 
        self.y = data['y']
        self.t = data['t']
        self.u = data['velocity']
        # Convert python lists to numpy arrays
        self.u =
        self.x =
        self.y =
        self.t =