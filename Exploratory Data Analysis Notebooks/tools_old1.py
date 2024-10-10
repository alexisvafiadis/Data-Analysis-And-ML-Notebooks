import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np
import pandas as pd
from math import ceil

sns.set(style="white")

#utility functions
def labelsizes(ax):
    ax.yaxis.label.set_size(18)
    ax.xaxis.label.set_size(18)
    ax.title.set_size(20)

def adjustfig(fig):
    fig.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.16, 
                    hspace=0.4)

def plotcorrmatrix(data):
    corrMatrix = data.corr()
    fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(20,20))
    ax = sns.heatmap(corrMatrix, annot=True, cmap="Blues")
    return fig

def cat_plot(data, col, ystr, type):
    col = [x for x in col if (len(data[x].unique()) < 30) & (col != ystr)]
    fig, axes = plt.subplots(nrows = ceil(len(col)/2), ncols=2, figsize=(20,4 * len(col)))
    axs = axes.flatten()
    j = 0
    if type == "regression":
        #fig.suptitle('Distribution of Y with respect to categories', fontsize=22)
        for i in col:
            ax = sns.boxplot(data=data, x=i, y=ystr, ax=axs[j])
            labelsizes(ax)
            if len(data[i].unique()) > 5:
                ax.tick_params(labelrotation=90)
            j = j + 1
    elif type == "classification":
        #fig.suptitle('Count of each class for all categories', fontsize=22)
        for i in col:
            ax = sns.histplot(data=data, x=i, hue=ystr, ax=axs[j], multiple="stack", stat="probability")
            labelsizes(ax)
            j = j + 1
    if len(col) % 2 != 0:
        axs[len(col)].set_axis_off()
    adjustfig(fig)
    return fig  


def num_plot(data, col, ystr, type):
    fig, axes = plt.subplots(nrows = ceil(len(col)/2), ncols=2, figsize=(20,4 * len(col)))
    axs = axes.flatten()
    j = 0
    if type == "regression":
        #fig.suptitle('Scatter plot of Y depending on each feature (+ line of best fit)', fontsize=22)
        for i in col:
            if i != ystr:
                ax = sns.regplot(data=data, x=i, y=ystr, ax=axs[j], scatter_kws={"s": 2}, line_kws={'color': 'red'})
                labelsizes(ax)
                j = j + 1
    elif type == "classification":
        #fig.suptitle('Density evolution of each class depending on numerical features', fontsize=22)
        for i in col:
            if i != ystr:
                ax = sns.kdeplot(data=data, x=i, hue=ystr, ax=axs[j])
                labelsizes(ax)
                j = j+1
    if len(col) % 2 != 0:
        axs[len(col)].set_axis_off()
    adjustfig(fig)
    return fig

def y_distribution(data, col):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,6))
    ax = sns.histplot(data=data, x=col, ax=ax)
    return fig
    
