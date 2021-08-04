"""
This code is developed to generate outlier scores objects in databases of
arbitrary dimensionality.
Scores are based on the k-Nearest Neighbor (kNN) distance. This kinship is
inversely proportional to outlier score, as outliers do not relate to other
data.
"""
import numpy as np
from sklearn.neighbors import NearestNeighbors


def dist_scores(data, d2s=None, kmin=4, kmax=None):
    """
    This method scores data based on distance to the k-th neighbor for a range
    of k's from kmin to kmax
    Args:
        data (Numpy array or Pandas dataframe) - The reference data to which
            distances will be calculated
        d2s (Numpy array or Pandas dataframe) - Data to be scored (if None,
            same as the reference data)
        kmin (integer) - minimum neighbor to calculate the distance
        kmax (integer) - maximum neighbor to calculate the distance
        Note: The original DBSCAN paper suggests choosing k=4 and that k
            beyond this doesn't provide additional insight.
            Given the increase of d.ata volume I'm not sure this argument
            holds up, but the heuristic
            continues to enjoy widespread use, we use it as a base.
    Returns:
        scoress (Numpy array) - MinMax scaled scores for data in d2s.
    """
    if isinstance(d2s, type(None)):
        # if d2s is a dataframe, the evaluation of d2s==None tries to compare
        # every value of the dataframe to None, instead we check if it's
        # NoneType
        d2s = data  # will default to full data if d2s is not specified
    kmax = kmax or kmin
    assert kmax >= kmin, "Max value of k given is smaller than the min."
    nbrs = NearestNeighbors(
        n_neighbors=kmax+1, algorithm='ball_tree', n_jobs=-1).fit(data)
    distances, indices = nbrs.kneighbors(d2s)

    # empty numpy array which will contain scores, one additional column for
    # average score
    if kmax == kmin:
        scores = np.zeros((1, len(distances)))
    else:
        scores = np.zeros((kmax-kmin+2, len(distances)))

    # There's probably a better way to do this with numpy arrays, but it's not
    # worth figuring out
    for k in range(kmin, kmax+1):
        k_scores = distances[:, k]
        # not using sklearn's preprocessing module bc scaling 1D at a time.
        k_scores = (k_scores-k_scores.min()) / \
            (k_scores.max()-k_scores.min())  # min max scaled
        scores[k-kmin] = k_scores

    if kmax != kmin:
        k_av_scores = np.sum(distances[:, kmin:kmax], axis=1)
        k_av_scores = (k_av_scores-k_av_scores.min()) / \
            (k_av_scores.max()-k_av_scores.min())  # min max scaled
        scores[-1] = k_av_scores

    # TODO: readjust scaling so that the extreme outliers don't
    # TODO (cont): dominate scores of the rest.
    # Potentially scale 90th percentile, define all beyond that as having a
    # score of 1.

    return scores


def kinship_scores(data, d2s=None, kmin=4, kmax=None, samp_size=-1, n_iter=1):
    """
    Args:
        data (Numpy array or Pandas dataframe) - Full set of data (scaled and
            culled of irrelevant data)
        d2s (Numpy array or Pandas dataframe) - Subset of data to be scored
            (optional if interested in a subset).
            Example case: scoring previously identified outliers only.
        kmin (integer) - Minimum neighbor, the distance to which is analagous
            to the score
        kmax (integer, optional) - Maximum neighbor, if used, to find
            distance to and to average scores over
        n_iter (integer) - Number of iterations,

    Returns:
        scores (Numpy array) - MinMax scaled scores for data in d2s for all
            neighbors
        between kmin and kmax, as well as the average score over the distances
            from kmin to kmax.
        The total size will be len(d2s) x (kmax-kmin+1).

    Purpose:
        Calculate scores for each point within the dataset in an efficient
        manner. The score is based on the distance to the k-Nearest Neighbor,
        but in this implementation a reference sample is generated and
        used to calculate distances instead of the genuine neighbors.
        This is in order to make this process scalable to larger datasets,
        calculating the actual k nearest neighbor distance for all points for
        high density, high dimensional data is not particularly efficient or
        informative.
    """
    if samp_size > 1:
        # Creating a random sample of reference points to which distances will
        # be calculated for all data
        ref_data_sample = data.sample(n=samp_size)
    else:
        # if samp_size isn't specified well, scoring is done relative to the
        # whole dataset
        ref_data_sample = data

    scores_i = dist_scores(ref_data_sample, d2s=d2s, kmin=kmin, kmax=kmax)
    if n_iter > 1:
        for i in range(n_iter-1):
            # Speed improvements: using ball_tree to calculate distance to
            # only k+1 neighbors
            # Using subset of reference points for calculating distances
            if samp_size > 1:
                ref_data_sample = data.sample(n=samp_size)
            scores_i += dist_scores(ref_data_sample,
                                    d2s=data, kmin=kmin, kmax=kmax)
    else:
        scores_i = dist_scores(ref_data_sample, d2s=data, kmin=kmin, kmax=kmax)
        n_iter = 1
        """
        TODO:
        loop until scores converge (not implemented)
        """
        pass
    scores = scores_i/n_iter
    return scores
