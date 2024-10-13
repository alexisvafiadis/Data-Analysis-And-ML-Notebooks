import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np
import pandas as pd
from math import ceil
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

#import importlib
#import tools
#importlib.reload(tools)

sns.set(style="white")

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
    ax = sns.histplot(data=data, x=col, ax=ax, stat="probability")
    return fig
    
def distribution(data, col):
    fig, axes = plt.subplots(nrows = ceil(len(col)/2), ncols=2, figsize=(20,4 * len(col)))
    axs = axes.flatten()
    j = 0
    for i in col:
        ax = sns.histplot(data=data, x=i, ax=axs[j], stat="probability")
        labelsizes(ax)
        j = j + 1
    if len(col) % 2 != 0:
        axs[len(col)].set_axis_off()
    adjustfig(fig)
    return fig

def get_unique_values(data):
    unique_values = data.nunique().sort_values()
    unique_values.plot.bar(logy=True, figsize=(15, 4), title="Valeurs uniques par colonne")

def perform_pca(data,col,normalize=True):
    pca = PCA(2)
    if (normalize):
        scaler = StandardScaler() 
        data = scaler.fit_transform(data)
    score = pca.fit_transform(data)
    #On affiche les variance de chaque PC
    exp_var_pca = pca.explained_variance_ratio_
    print("Explained variance for 1st component : ", exp_var_pca[0])
    print("Explained variance for 2nd component : ", exp_var_pca[1])
    #On affiche les coeff pour chaque colonne
    pcadf = pd.DataFrame(pca.components_, columns=col, index=['PC1', 'PC2'])
    return pca, score, pca.components_, pcadf

def biplot(score,coef,labels=None):
    fig, ax = plt.subplots(figsize=(10,10))

    #Cercle de corrélation
    an = np.linspace(0, 2 * np.pi, 100)    
    plt.plot(np.cos(an), np.sin(an), c='g')

    #Paramètres
    ax.set_aspect('equal')
    ax.grid(True, which='both')
    sns.despine(ax=ax, offset=0)
    ax.set_ylim(-1, 1)
    ax.set_xlim(-1, 1)

    #Scatter plot
    xs = score[:,0]
    ys = score[:,1]
    coef = np.transpose(coef)
    n = coef.shape[0]
    scalex = 1.0/(xs.max() - xs.min())
    scaley = 1.0/(ys.max() - ys.min())
    plt.scatter(xs * scalex,ys * scaley,
                s=4, 
                color='b')
 
    #Loading vectors
    for i in range(n):
        plt.arrow(0, 0, coef[i,0], 
                  coef[i,1],color = 'red',
                  alpha = 0.5)
        plt.text(coef[i,0]* 1.15, 
                 coef[i,1] * 1.15, 
                 labels[i], 
                 color = 'r', 
                 ha = 'center', 
                 va = 'center')
 
    #Afficher
    plt.xlabel("PC{}".format(1), fontsize=16)
    plt.ylabel("PC{}".format(2), fontsize=16)    
    plt.figure()
    plt.show()

def check_missing(data):
    percent_missing = data_russia_full.isnull().sum() * 100 / len(data_russia_full)
    missing_value_df = pd.DataFrame({'Nom de colonne': data_russia_full.columns,
                                    'Pourcentage de NA': percent_missing})
    missing_value_df.head()