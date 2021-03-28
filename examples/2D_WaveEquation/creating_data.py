# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 17:00:55 2021

@author: 49174
"""


# data 
import numpy as np
from scipy.spatial.distance import euclidean
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
import scipy.io

import pandas as pd
vptxt =np.loadtxt(r'D:\thesis\data_hendrick\2D_gound_models\vp.txt')
x = np.arange(0,10,0.0125)
y = np.arange(0,10,0.0125)
xx, yy = np.meshgrid(x, y, sparse=True)
distance = np.sqrt(yy**2 + xx**2)
time = np.divide(distance,vptxt)#add the velocity file instead of vp txt
import seaborn as sns; sns.set_theme()
''''
vel.rename(columns=lambda x: str(x*0.0125)[0:3], inplace=True)
vel.rename(index=lambda x: str(x*0.0125)[0:3], inplace=True)

ax = sns.heatmap(vel) # create seaborn heatmap


plt.title('Velocities in m/s', fontsize = 20) # title with fontsize 20
plt.xlabel('X in m', fontsize = 15) # x-axis label with fontsize 15
plt.ylabel('Y in m', fontsize = 15) # y-axis label with fontsize 15

plt.show()

'''




Data= {}
Data['x']=x
Data['y']=y
Data['t']=time
Data['velocity']= vptxt
 
scipy.io.savemat('seismic_data.mat', Data)
#Data2 = mat = scipy.io.loadmat('seismic_data.mat')