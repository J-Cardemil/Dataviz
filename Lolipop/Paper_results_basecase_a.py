# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 12:27:00 2022

@author: jblan
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from itertools import groupby
from pylab import cm,matplotlib
from matplotlib.colors import ListedColormap


path_to_file = 'Resultados Macro v_rhoP paper.xlsx'
df=pd.read_excel(path_to_file,sheet_name='Análisis Base-Sales',header=0,engine='openpyxl')
df1 = df.sort_index(axis=0,ascending=False)
df1 = df.groupby(['Discharge','Charge','Sal']).sum()


#%%
# Definition of required parameters and tools
my_range = list(range(1,len(df.index)+1))

# Font size
fs_t = 12
fs_x = 18

axes, xlim1, xlim2, var1, var2, m  = ([0,1,2,3],[35,0.0,0.0,0],
                                         [57,7.5,5.2,100],
                                   ['RTE','r','rho_P','rho_E'],
                                   ['RTE [%]','$r$',r'$\rho_P$ $[MW/(m^3/s)]$',r'$\rho_E$ $[kWh/m^3]$'],
                                   [100,1,1,1]) 
# For colors
viridisBig = cm.get_cmap('viridis', 512)
cmap = ListedColormap(viridisBig(np.linspace(0.0, 0.9, 5)))
c_v=[]
for i in range(cmap.N):
    rgba = cmap(i)
    # rgb2hex accepts rgb or rgba
    c_v.append(matplotlib.colors.rgb2hex(rgba))
    
#%%
fig, ax = plt.subplots(nrows=1, ncols=4, sharey=True, figsize=(15,15))
plt.subplots_adjust(wspace=0.05)

for i in axes:
    plt.axes(ax[i])
    ax[i].set_xlim(xlim1[i],xlim2[i])
    j=0
    for n in my_range:
        if df1.index.get_level_values('Sal')[j] == 'MS 1':
            c = c_v[0]
        elif df1.index.get_level_values('Sal')[j] == 'MS 2':
            c = c_v[1]
        elif df1.index.get_level_values('Sal')[j] == 'MS 3':
            c = c_v[2]
        elif df1.index.get_level_values('Sal')[j] == 'MS 4':
            c = c_v[3]
        elif df1.index.get_level_values('Sal')[j] == 'MS 5':
            c = c_v[4]
        
        # These two lines are for lollipop
        plt.hlines(y=my_range[j], xmin=0, xmax=df1[var1[i]][j]*m[i], color=c, alpha=0.8)
        plt.plot(df1[var1[i]][j]*m[i], my_range[j], "D", ms=5,color=c)
        #
        
        j+=1
    plt.xlabel(var2[i], fontsize=fs_x)
    plt.grid(axis='x')
    plt.xticks(fontsize=fs_t)
    ax0 = ax[i].twiny()
    ax0.set_xlim(ax[i].get_xlim())
    plt.xlabel(var2[i], fontsize=fs_x)
    plt.xticks(fontsize=fs_t)

h = 5.5
while h < 31.5:
    if h == 15.5 or h ==30.5:
        ax[0].axhline(xmin=-0.4,xmax=0,y=h,c="gray",alpha=0.8,ls='-',zorder=0, clip_on=False)
        h += 5  
    else:   
        ax[0].axhline(xmin=-0.25,xmax=0,y=h,c="gray",alpha=0.8,ls='-',zorder=0, clip_on=False)
        h += 5         
ax[0].axhline(xmin=-0.25,xmax=0,y=34.5,c="gray",alpha=0.8,ls='-',zorder=0, clip_on=False)
h = 39.5
while h < 49.5:
    ax[0].axhline(xmin=-0.25,xmax=0,y=h,c="gray",alpha=0.8,ls='-',zorder=0, clip_on=False)
    h += 5

h = 5.5    
while h < 35.5:
    for i in axes:
        ax[i].axhline(xmin=0,xmax=1,y=h,c="gray",alpha=0.8,ls='--',zorder=0, clip_on=False)
    h += 5    
for i in axes:
    ax[i].axhline(xmin=0,xmax=1,y=34.5,c="gray",alpha=0.8,ls='--',zorder=0, clip_on=False)
h = 39.5
while h < 49.5:
    for i in axes:
        ax[i].axhline(xmin=-0,xmax=1,y=h,c="gray",alpha=0.8,ls='--',zorder=0, clip_on=False)
    h += 5

# plt.yticks(my_range, df['Sal'])
plt.yticks(my_range, '')

# Functions for adding labels and additional lines
def add_line(ax, xpos, ypos):
    line = plt.Line2D([xpos-.1, xpos+ .1], [ypos, ypos],
                       transform=ax.transAxes, color='black')
    line.set_clip_on(False)
    ax.add_line(line)

def label_len(my_index,level):
    labels = my_index.get_level_values(level)
    return [(k, sum(1 for i in g)) for k,g in groupby(labels)]
    
def label_group_bar_table(ax, df):
    xpos = .05
    scale = 0.929/df.index.size
    for level in range(df.index.nlevels)[::-1]:
        pos = 1.7
        for label, rpos in label_len(df.index,level):
            if label in ('MS 1','MS 2','MS 3','MS 4','MS 5'):
               label = '' 
            lypos = (pos + 0.5 * rpos)*scale
            ax.text(xpos, lypos, label, ha='center', transform=ax.transAxes)
            # add_line(ax, xpos, (pos+.15)*scale)
            pos += rpos
        # add_line(ax, xpos, (pos+.15)*scale)
        xpos -= .2

# For grouping labels in the y axis        
label_group_bar_table(ax[0], df1)

# Molten salt legends
S1_p = mpatches.Patch(color=c_v[0],label='MS1: 60 $NaNO_3$ + 40 $KNO_3$')
S2_p = mpatches.Patch(color=c_v[1],label='MS2: 53 $KNO_3$ + 18 $NaNO_3$ + 29 $LiNO_3$')
S3_p = mpatches.Patch(color=c_v[2],label='MS3: 33.4 $Na_2 CO_3$ + 34.5 $K_2 CO_3$ + 32.1 $Li_2 CO_3$')
S4_p = mpatches.Patch(color=c_v[3],label='MS4: 37.5 $MgCl_2$ + 62.5 $KCl$')
S5_p = mpatches.Patch(color=c_v[4],label='MS5: 8.1 $NaCl$ + 31.3 $KCl$ + 60.6 $ZnCl_2$')
plt.legend(handles=[S1_p,S2_p,S3_p,S4_p,S5_p,], loc='lower center', 
           bbox_to_anchor=(-1.15, -0.125), fancybox=True, shadow=True, ncol=3, prop={'size': 11})

plt.show()