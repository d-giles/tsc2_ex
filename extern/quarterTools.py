from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE
from sklearn import preprocessing
import pandas as pd
import sys

import numpy as np
np.set_printoptions(threshold=sys.maxsize)


#####################################
### Functions for data processing ###
#####################################


def data_scaler(df, nfeats=60):
    """
    Purpose:
        Data scaler this work, returns a dataframe w/ scaled features
    Args:
        df (pd.DataFrame) - dataframe with features to be scaled
        nfeats (int) - number of features in the dataframe, defaults to 60, should be changed if dataframe is a reduction.
    Returns:
        scaled_df (pd.DataFrame)- the dataframe of scaled data. If there were additional calculated columns in the original datafarme, they will not be present in the scaled dataframe.
    """
    data = df.iloc[:, 0:nfeats]
    scaler = preprocessing.StandardScaler().fit(data)
    scaled_data = scaler.transform(data)
    scaled_df = pd.DataFrame(index=df.index,
                             columns=df.columns[:nfeats],
                             data=scaled_data)
    return scaled_df


def tsne_fit(data, perplexity='auto', scaled=False):
    """
    Performs a t-SNE dimensionality reduction on the data sample generated.
    Uses a PCA initialization and the perplexity given, or defaults to 1/10th the amount of data.

    Returns:
        tsneReduction (pd.DataFrame) - a dataframe containing the coordinates for the tsne reduction.
    """
    assert len(
        data) < 20000, "Dataset too large, t-SNE cannot handle very large datasets. Try sampling a subset."
    if type(perplexity) == str:
        perplexity = len(data)/10
    if scaled:
        scaledData = data
    else:
        scaledData = data_scaler(data, nfeats=len(data.columns))

    tsne = TSNE(n_components=2, perplexity=perplexity,
                init='pca', verbose=True)
    fit = tsne.fit_transform(scaledData)
    # Goal is to minimize the KL-Divergence, so it can be useful to know what the resulting divergence is.
    import sklearn
    if sklearn.__version__ == '0.18.1':
        print("KL-Divergence was %s" % tsne.kl_divergence_)
    tsneReduction = pd.DataFrame(
        index=data.index, data=fit, columns=['tsne_x', 'tsne_y'])
    return tsneReduction


def pca_red(data, var_rat=0.9, scaled=False, verbose=True):
    """
    Returns a pca reduction of the given data that explains a
    given fraction of the variance.
    Args:
        data (pd.DataFrame) - the data to be reduced (the features presumably), culled of irrelevant data (any calculated columns like scores or outlier labels)
        var_rat (float between 0 and 1) - the ratio of variance to be 
            explained by the transformed data
        scaled (boolean) - if False, data will be scaled using sklearn's preprocessing module StandardScaler

    Returns:
        reduced_data (pd.DataFrame) - dataframe of the reduced data with colums being the principle component inds
        prints the number of dimensions and the explained variance
    """
    if scaled:
        scaled_data = data
    else:
        if verbose:
            print("Scaling data using StandardScaler...")
        scaled_data = data_scaler(data, nfeats=len(data.columns))

    if verbose:
        print("Finding minimum number of dimensions to explain {:04.1f}% of the variance...".format(
            var_rat*100))

    for i in range(len(data.columns)):
        pca = PCA(n_components=i)
        pca.fit(scaled_data)
        if sum(pca.explained_variance_ratio_) >= var_rat:
            break
    if verbose:
        print("""
        Dimensions: {:d},
        Variance explained: {:04.1f}%
        """.format(i, sum(pca.explained_variance_ratio_)*100))

    reduced_data = pd.DataFrame(index=data.index,
                                data=pca.transform(scaled_data))
    return reduced_data

