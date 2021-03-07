# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 18:16:33 2021

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
    def __init__(self,  pData, batchSize, numBatches, nt, timeStep, nx=640,
                 ny=480, pixStep=4, shuffle=True, useGPU=False)
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
          for tPoint in range(
                0, nt, timeStep):
              exact_u = np.array(data['seq'][:])
              timing = np.array(data['timing'][:])
         for xi in range(
                    0, nx, pixStep):  # sample only each pixStep-th spatial point from an image
                for yi in range(0, ny, pixStep):
                    if Exact_u[xi, yi] != 0:  # neglect non-cortex data
                        self.u.append(Exact_u[xi, yi])
                        self.x.append(xi)
                        self.y.append(yi)
                        self.t.append(timing)