# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 20:00:31 2021

@author: 49174
"""

import sys
from Creating_datasets import *
from argparse import ArgumentParser
sys.path.append('../..')  # PINNFramework etc.
import PINNFramework as pf
if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--identifier", type=str, default="UFZ_2DWaveEquationHPM')
    parser.add_argument("--pData", type=str, default="D:\thesis_code\NeuralSolvers\examples\2D_WaveEquation\seismic_data.mat")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--hidden_size", type=int, default=500)
    parser.add_argument("--num_hidden", type=int, default=8)
    parser.add_argument("--hidden_size_alpha", type=int, default=500)
    parser.add_argument("--num_hidden_alpha", type=int, default=8)
    parser.add_argument("--hidden_size_hs", type=int, default=500)
    parser.add_argument("--num_hidden_hs", type=int, default=8)
    args = parser.parse_args()
    
    # Initial condition
    ic_dataset = InitialConditionDataset(pData=args.pData)
    initial_condition = pf.InitialCondition(ic_dataset)
    
    # PDE dataset
    pde_dataset = PDEDataset(pData=args.pData)
    
    # C model 
    # Input: spatiotemporal coordinates of a point x,y,t
    # Output: C value for the point = incompresbility/density
    c_net = pf.models.MLP(input_size=3,
                              output_size=1,
                              hidden_size=args.hidden_size_alpha,
                              num_hidden=args.num_hidden_alpha,
                              lb=ic_dataset.lb[:3], #lb for x,y,t
                              ub=ic_dataset.ub[:3]) #ub for x,y,t
    # PINN model
    # Input: spatiotemporal coordinates of a point x,y,t
    # Output: velocity u at the point
    model = pf.models.MLP(input_size=3,
                          output_size=1,
                          hidden_size=args.hidden_size,
                          num_hidden=args.num_hidden,
                          lb=ic_dataset.lb[:3], #lb for x,y,t
                          ub=ic_dataset.ub[:3]) #ub for x,y,t
    