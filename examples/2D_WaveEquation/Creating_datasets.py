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
import scipy.io
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
    def __init__(self, pData, nx=800,ny=800, useGPU=False, grid_size=0.0125):
        
        
  
        #Constructor of the initial condition dataset
        # __getitem()__ returns a batch with x,y,t to compute u_predicted value at as well as u_exact

        self.u = []  # distance
        self.x = []
        self.y = []
        self.t = []
        if not os.path.exists(pData):  # Check if given path to data is valid
            raise FileNotFoundError('Could not find file' + pData)
        data= scipy.io.loadmat(pData)
        Exact_u = data['velocity']
        time= data['t']
        for xi in range(0,nx,grid_size):
            for yi in range(0,ny,grid_size):
                self.u.append(Exact_u[xi, yi])
                self.t.append(time[xi,yi])
                self.x.append(xi)
                self.y.append(yi)
                        
                
        # Convert python lists to numpy arrays of 1D shape
        self.u = np.array(self.u).reshape(-1)
        self.x = np.array(self.x).reshape(-1)
        self.y = np.array(self.y).reshape(-1)
        self.t = np.array(self.t).reshape(-1)
        # Create dictionary with information about the dataset
        self.cSystem = {
            "x_lb": self.x.min(),
            "x_ub": self.x.max(),
            "y_lb": self.y.min(),
            "y_ub": self.y.max(),
            "nx": nx,
            "ny": ny,
            "t_ub": self.t.max()}
        # Boundaries of spatiotemporal domain
        self.lb = np.array([self.x.min(), self.y.min(), self.t.min()])
        self.ub = np.array([self.x.max(), self.y.max(), self.t.max()])
        if (useGPU):    # send to GPU if requested
            # send to GPU if requested
            self.dtype = torch.cuda.FloatTensor
            self.dtype2 = torch.cuda.LongTensor
        else:
            self.dtype = torch.FloatTensor
            self.dtype2 = torch.LongTensor
        # sending data to the gpu
         
        self.x = self.dtype(self.x)
        self.y = self.dtype(self.y)
        self.t = self.dtype(self.t)
        self.u = self.dtype(self.u)
    def __len__(self):
        """
        Length of the dataset
        """
        return 1
    def __getitem__(self, index):
        """
        Returns item at given index
        """
         # Generate batch for inital solution
        x = (self.x)
        y = (self.y)
        t = (self.t)
        u = (self.u)
        return torch.stack([x, y, t], 1), u
    
class PDEDataset(Dataset):
    def __init__(self, pData,t_ub, nx=800,
                 ny=800, grid_size=0.0125,useGPU=False):
        """
        Constructor of the residual points dataset
        __getitem()__ returns a batch with x,y,t points to compute residuals at
        """
        self.x = []
        self.y = []
        self.t = []
        data = scipy.io.loadmat(pData)
        Exact_u= data['velocity']
        Exact_u= Exact_u.reshape(nx,ny)
        time = data['t'].reshape(nx,ny)
        for xi in range(
                    0, nx, grid_size):  # sample for each xi
                for yi in range(0, ny, grid_size): # sample for each yi
                    self.x.append(xi)
                    self.y.append(yi)
                    self.t.append(time[xi,yi])
        
         # Convert python lists to numpy arrays
        self.x = np.array(self.x).reshape(-1)
        self.y = np.array(self.y).reshape(-1)
        self.t = np.array(self.t).reshape(-1)
        
        if (useGPU):  # send to GPU if requested
            self.dtype = torch.cuda.FloatTensor
            self.dtype2 = torch.cuda.LongTensor
        else:
            self.dtype = torch.FloatTensor
            self.dtype2 = torch.LongTensor
    def __len__(self):
        """
        Length of the dataset
        """
        return 1 #we don't have a patch process
    def __getitem__(self, index):
        x = (self.x)
        y = (self.y)
        t = (self.t)
        return torch.stack([x, y, t], 1)
def derivatives(x, u):
    """
    Calculate the nn output at postion (x,y) at time t
    :param x: position vector containing (x,y,t)
    :param u: the velocity vector
    :return: Approximated solutions and their gradients
    """
    x.requires_grad = True
    # Calculate derivatives with torch automatic differentiation
    # Move to the same device as prediction
    grads = torch.ones(u.shape, device=u.device)
    J_U = torch.autograd.grad(u, x, create_graph=True, grad_outputs=grads)[0]
    u_x = J_U[:, 0].reshape(u.shape)
    u_y = J_U[:, 1].reshape(u.shape)
    u_t = J_U[:, 2].reshape(u.shape)
    u_xx = torch.autograd.grad(
        u_x, x, create_graph=True, grad_outputs=grads)[0]
    u_yy = torch.autograd.grad(
        u_y, x, create_graph=True, grad_outputs=grads)[0]
    u_tt = torch.autograd.grad(
        u_t, x, create_graph=True, grad_outputs=grads)[0]
     
    u_xx = u_xx[:, 0].reshape(u.shape)
    u_yy = u_yy[:, 1].reshape(u.shape)
    u_tt = u_tt[:, 2].reshape(u.shape)
    x, y, t = x.T
    x = x.reshape(u.shape)
    y = y.reshape(u.shape)
    t = t.reshape(u.shape)

    return torch.stack([x, y, t, u, u_xx, u_yy, u_tt], 1).squeeze()
                
                
        
            	
        