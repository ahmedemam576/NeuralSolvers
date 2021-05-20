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
from numpy import arange
import scipy.io
from argparse import ArgumentParser
from torch.utils.data import Dataset

# class BoundaryConditionsDataSet(Dataset):
#     def __init__(self, nb, lb, ub):
#         """
#         Constructor of the initial condition dataset
#         """
#         raise NotImplementedError

#     def __getitem__(self, idx):
#         """
#         Returns data for initial state
#         """
#         raise NotImplementedError

#     def __len__(self):
#         """
#         Length of the dataset
#         """
#         raise NotImplementedError
    
class InitialConditionDataset(Dataset):
    def __init__(self, data_info, batch_size, num_batches):
        
        
  
        #Constructor of the initial condition dataset
        # __getitem()__ returns a batch with x,z,t,u to compute u_predicted value at as well as u_exact

        self.u_values = [] #pressure
        self.x_values = []
        self.z_values = []
        self.t_values = []
        self.x_indices = []
        self.z_indices = []
        if not os.path.exists(
                data_info["path_data"]):
            raise FileNotFoundError(
                'Could not find file' +
                data_info["path_data"]) 
        data= scipy.io.loadmat(data_info["path_data"])
        u_exact = data['pressure']
        u_exact = data['pressure'].reshape(data_info["num_x"], data_info["num_y"])
    
        x = data["x"]
        z = data["z"]
        time = data['t']
        for ti in range(0,len(time),1):
            for xi in range(0,len(x),1):
                for zi in range(0,len(z),1):
                self.u_values.append(u_exact[xi, yi])
                self.x_values.append(x[xi])
                self.z_values.append(z[zi])
                self.t_values.append(time[ti])
                self.x_indices.append(xi)
                self.z_indices.append(zi)
                
            
                        
                
        # Convert python lists to numpy arrays of 1D shape
        self.u_values = np.array(self.u_values).reshape(-1)
        self.x_values = np.array(self.x_values).reshape(-1)
        self.z_values = np.array(self.z_values).reshape(-1)
        self.t_values = np.array(self.t_values).reshape(-1)
        
        self.x_indices = np.array(self.x_indices).reshape(-1)
        self.z_indices = np.array(self.z_indices).reshape(-1)
        
        # Sometimes we are loading less files than we specified by batch_size + num_batches
        # => adapt num_batches to real number of batches for avoiding empty batches
        self.batch_size = batch_size
        num_samples = min((num_batches * batch_size, len(self.x_values)))
        self.num_batches = num_samples // self.batch_size
        
        
        # Create lists with boundary values for spatio-temporal coordinates
        self.low_bound = [
            self.x_values.min(),
            self.z_values.min(),
            self.t_values.min()]
        self.up_bound = [
            self.x_values.max(),
            self.z_values.max(),
            self.t_values.max()]

        dtype1 = torch.FloatTensor
        dtype2 = torch.LongTensor
        
        # Generate random permutation idx
        np.random.seed(1234)
        rand_idx = np.random.permutation(self.x_values.shape[0])
        
        
            
        # Permutate data points
        self.x_values = self.x_values[rand_idx]
        self.z_values = self.y_values[rand_idx]
        self.t_values = self.t_values[rand_idx]
        self.u_values = self.u_values[rand_idx]
        
        self.x_indices = self.x_indices[rand_idx]
        self.z_indices = self.y_indices[rand_idx]
        
        
        
        # Slice data for training and convert to torch tensors
        self.x_values = dtype1(self.x_values[:num_samples])
        self.z_values = dtype1(self.y_values[:num_samples])
        self.t_values = dtype1(self.t_values[:num_samples])
        self.u_values = dtype1(self.u_values[:num_samples])
        self.x_indices = dtype2(self.x_indices[:num_samples])
        self.z_indices = dtype2(self.y_indices[:num_samples])

        self.low_bound = dtype1(self.low_bound)
        self.up_bound = dtype1(self.up_bound)
        
        def cuda(self):
        """
        Sends the dataset to GPU
        """        
        self.x_values = self.x_values.cuda()
        self.z_values = self.y_values.cuda()
        self.t_values = self.t_values.cuda() 
        self.u_values = self.u_values.cuda()
        self.x_indices = self.x_indices.cuda()
        self.z_indices = self.y_indices.cuda()

    def __len__(self):
        """
        Length of the dataset
        """
        return self.num_batches
    
    
    def __getitem__(self, index):
        """
        Returns a mini-batch at given index containing X,u
        Args:
            index(int): index of the mini-batch
        Returns:
            X: spatio-temporal coordinates x,z,t concatenated
            u: real-value function of spatio-temporal coordinates
        """
        # Generate batch for inital solution
        x_values = (
            self.x_values[index * self.batch_size: (index + 1) * self.batch_size])
        z_values = (
            self.z_values[index * self.batch_size: (index + 1) * self.batch_size])
        t_values = (
            self.t_values[index * self.batch_size: (index + 1) * self.batch_size])
        u_values = (
            self.u_values[index * self.batch_size: (index + 1) * self.batch_size])
        x_indices = (
            self.x_indices[index * self.batch_size: (index + 1) * self.batch_size])
        z_indices = (
            self.z_indices[index * self.batch_size: (index + 1) * self.batch_size])
        return torch.stack([x_values, z_values, t_values, x_indices, z_indices], 1),u_values.reshape(-1,1)
    
    
    
    
#____________________________________________________________________________________________________________________#
    
class PDEDataset(Dataset):
    """
    Dataset with points (x,y,t) to train HPM model on: HPM(x,y,t) ≈ du/dt.
    """

    def __init__(self, data_info, batch_size, num_batches):
        """Constructor of the residual poins dataset.
        Args:
            data_info (dict): dictionary with info about the data.
            batch_size (int): size of a mini-batch in the dataset
            num_batches (int): number of mini-batches in the dataset
        """

        self.x_values = []
        self.z_values = []
        self.t_values = []
        self.x_indices = []
        self.z_indices = []
        
        if not os.path.exists(
                data_info["path_data"]):
            raise FileNotFoundError(
                'Could not find file' +
                data_info["path_data"])
            
        data= scipy.io.loadmat(data_info["path_data"])
        u_exact = data['pressure']
        u_exact.reshape(data_info["num_x"], data_info["num_y"])
        
        x = data["x"]
        z = data["z"]
        time = data['t']
        for t_i in range(0,data_info["num_t"],1):
            for x_i in range(0,data_info["num_x"],1):
                for z_i in range(0,data_info["num_z"],1):
                self.u.append(Exact_u[ti,xi, zi])
                self.x_values.append(x[x_i])
                self.y_values.append(z[z_i])
                self.t_values.append(time[t_i])
                self.x_indices.append(xi)
                self.z_indices.append(zi)
          
            
            
        
        
         # Convert python lists to numpy arrays
        self.x_values = np.array(self.x_values).reshape(-1)
        self.z_values = np.array(self.z_values).reshape(-1)
        self.t_values = np.array(self.t_values).reshape(-1)
        self.x_indices = np.array(self.x_indices).reshape(-1)
        self.z_indices = np.array(self.z_indices).reshape(-1)
        
        
        # Sometimes we are loading less files than we specified by batch_size * num_batches
        # => adapt num_batches to real number of batches for avoiding empty batches
        self.batch_size = batch_size
        self.num_batches = num_batches
        num_samples = min((num_batches * batch_size, len(self.x_values)))
        
        dtype1 = torch.FloatTensor
        dtype2 = torch.LongTensor
        
        
        
        # Generate random permutation idx
        np.random.seed(1234)
        rand_idx = np.random.permutation(self.x_values.shape[0])
        
        # Permutate data points
        self.x_values = self.x_values[rand_idx]
        self.z_values = self.z_values[rand_idx]
        self.t_values = self.t_values[rand_idx]
        
        self.x_indices = self.x_indices[rand_idx]
        self.z_indices = self.z_indices[rand_idx]
        
        
        
      
        
        
        
        # Slice data for training and convert to torch tensors
        self.x_values = dtype1(self.x_values[:num_samples])
        self.z_values = dtype1(self.z_values[:num_samples])
        self.t_values = dtype1(self.t_values[:num_samples])
        
        self.x_indices = dtype2(self.x_indices[:num_samples])
        self.z_indices = dtype2(self.z_indices[:num_samples])
        
        
        def cuda(self):
            
        """
        Sends the dataset to GPU
        """        
            self.x_values = self.x_values.cuda()
            self.z_values = self.z_values.cuda()
            self.t_values = self.t_values.cuda() 
            self.x_indices = self.x_indices.cuda()
            self.z_indices = self.z_indices.cuda()
        
        
         def __len__(self):
        """
        Length of the dataset
        """
            return self.num_batches
    
        def __getitem__(self, index):
        """
        Returns a mini-batch at given index containing X
        Args:
            index(int): index of the mini-batch
        Returns:
            X: spatio-temporal coordinates x,y,t concatenated
        """
            # Generate batch with residual points
            x_values = (
                self.x_values[index * self.batch_size: (index + 1) * self.batch_size])
            z_values = (
                self.z_values[index * self.batch_size: (index + 1) * self.batch_size])
            t_values = (
                self.t_values[index * self.batch_size: (index + 1) * self.batch_size])
            x_indices = (
                self.x_indices[index * self.batch_size: (index + 1) * self.batch_size])
            z_indices = (
                self.z_indices[index * self.batch_size: (index + 1) * self.batch_size])
            return torch.stack([x_values, z_values, t_values, x_indices, z_indices], 1)


        
#_______________________________________________________________________________________________________________________#        
        


        
        

def derivatives(x_values, u_values):
    """
    Calculate derivatives of u with respect to x
    Args:
        x_values (torch tensor): concatenated spatio-temporal coordinaties (x,z,t)
        u_values (torch tensor): real-value function to differentiate
    Returns:
        x, z, t, u, d2u/dx2, d2u/dz2, d2u/dt2 concatenated
    """     
    # Save input in variabeles is necessary for gradient calculation
    x_values.requires_grad = True
    
    
    # Calculate derivatives with torch automatic differentiation
    # Move to the same device as prediction
    grads = torch.ones(u_values.shape, device=u_values.device)
    
    
    du_dx_values = torch.autograd.grad(
        u_values,
        x_values,
        create_graph=True,
        grad_outputs=grads)[0]
    #du_dx = [du_dx, du_dy, du_dt]
    u_x_values = du_dx_values[:, 0].reshape(u_values.shape)
    u_z_values = du_dx_values[:, 1].reshape(u_values.shape)
    u_t_values = du_dx_values[:, 2].reshape(u_values.shape)
    
    u_xx_values = torch.autograd.grad(
        u_x_values, x_values, create_graph=True, grad_outputs=grads)[0]
    u_zz_values = torch.autograd.grad(
        u_z_values, x_values, create_graph=True, grad_outputs=grads)[0]
    u_tt_values = torch.autograd.grad(
        u_t_values, x_values, create_graph=True, grad_outputs=grads)[0]
    #u_xx = [u_xx, u_xz, u_xt]
    u_xx_values = u_xx_values[:, 0].reshape(u_values.shape)
    #u_zz = [u_zx, u_zz, u_yt]
    u_zz_values = u_zz_values[:, 1].reshape(u_values.shape)
    #u-tt = [u_tx, u_tz ,u_tt]
     u_tt_values = u_tt_values[:, 2].reshape(u_values.shape)
        
    x_values, z_values, t_values, x_indices, y_indices = x_values.T
    x_values = x_values.reshape(u_values.shape)
    z_values = z_values.reshape(u_values.shape)
    t_values = t_values.reshape(u_values.shape)
    x_indices = x_indices.reshape(u_values.shape)
    z_indices = z_indices.reshape(u_values.shape) 
    return torch.stack([x_values, z_values, t_values, x_indices, z_indices, u_values,
                        u_xx_values, u_zz_values, u_tt_values], 1).squeeze()

                
        
            	
        