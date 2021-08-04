import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
from multiprocessing import Pool, cpu_count
from datetime import datetime
import pickle
import lightkurve
from numba import njit
# import numba  # numba is not currently compatible with python 3.9
from datetime import datetime
np.set_printoptions(threshold=sys.maxsize)

"""
TODO:
    It would make sense to refactor this whole thing into an object
    so that features are individually callable/calculable.
    That's a whole thing though... D:
"""

global tmpfile  # for dumping data during parallel processing
tmpfile = ''
# The master feature list
fts = [
    "longtermtrend", "meanmedrat", "skews", "varss", "coeffvar", "stds",
    "numoutliers", "numnegoutliers", "numposoutliers", "numout1s", "kurt",
    "mad", "maxslope", "minslope", "meanpslope", "meannslope", "g_asymm",
    "rough_g_asymm", "diff_asymm", "skewslope", "varabsslope", "varslope",
    "meanabsslope", "absmeansecder", "num_pspikes", "num_nspikes",
    "num_psdspikes", "num_nsdspikes", "stdratio", "pstrend", "num_zcross",
    "num_pm", "len_nmax", "len_nmin", "mautocorrcoef", "ptpslopes",
    "periodicity", "periodicityr", "naiveperiod", "maxvars", "maxvarsr",
    "oeratio", "amp", "normamp", "mbp", "mid20", "mid35", "mid50", "mid65",
    "mid80", "percentamp", "magratio", "sautocorrcoef", "autocorrcoef",
    "flatmean", "tflatmean", "roundmean", "troundmean", "roundrat",
    "flatrat"
    ]


def fl_files_w_path(fl, fitsDir='./fitsFiles/', fl_as_array=False):
    """
    Return an array with the files to be processed
    in the given filelist with the given path.

    If filelist given as array, it is assumed that the entire array
    needs processed. Whereas if a filelist is given as a txt file,
    the following looks for a completed filelist
    as well to avoid reprocessing.
    """
    if fl_as_array:
        files = np.char.array(fl)

    else:
        with open(fl, 'r') as file:
            files = file.readlines()
        fl_c = fl.replace('.txt', "")+"_completed.txt"
        if os.path.isfile(fl_c):
            """
            While it seems roundabout to convert to empty pandas dataframes,
            it's about 20% faster than using numpy arrays as follows.

            unique,counts = np.unique(np.concatenate([files,completed]),
                                      return_counts=True)
            files = unique[counts==1]
            """

            with open(fl_c, 'r') as file:
                completed = file.readlines()

            dff = pd.DataFrame(index=files)
            dfc = pd.DataFrame(index=completed)
            df = dff.append(dfc)
            df = df[~df.index.duplicated(keep=False)]
            # char.array allows appending the fits directory
            files = np.char.array(files)

        else:
            # char.array allows appending the fits directory
            files = np.char.array(files)
            # Creates a filelist to keep track of processed files.
            fcreate = open(fl_c, 'a')
            fcreate.close()

    return fitsDir+files


def read_fits_data(file):
    """
    Given the path of a fits file, this will extract the light curve and
    normalize it.
    """
    lc = lightkurve.read(file)
    lc = lc.remove_nans().normalize()

    return lc


def import_fits(nfile):
    """A wrapper function for read_fits_data

    Args:
        nfile (string) - file including path of lightcurve (as a fits file)
    Returns:
        todo: update to return lightkurve.LightCurve object
        t (np.array) - time array
        nf (np.array) - normalized flux array
        err (np.array) - normalized error array
    """
    try:
        lc = read_fits_data(nfile)

    except TypeError as err:
        # Files can be truncated by the zipping process.
        print("%s. Try downloading %s again." % (err, nfile))
        return

    # features = featureCalculation(nfile,t,nf,err)
    return lc


def load_filelist(fl, fitsDir, fl_as_array=False, verbose=True, useCpus=1):
    if verbose:
        print(f"Using {useCpus} cpus to calculate features...")
        print(f"Importing lightcurves...")
    files = fl_files_w_path(fl, fitsDir, fl_as_array)
    # Importing all the lightcurves first
    with Pool(useCpus) as p:
        p = Pool(useCpus)
        lcs = p.map(import_fits, files)

    return lcs


def recover(fl, temp_file, fl_as_array=False):
    """Recover from a crash.

    Purpose:
        In the event that processing lightcurves fails in the middle,
        this will read in the failsafe tmp file, evaluate which files were
        successfully processed,
        and modify the files list to be processed.

        Removes files already processed from files array and
        removes the original filelist if all files have been processed.

        Each lightcurve appends the pickle file, so this could take awhile,
        but it's probably (no basis for this statement) faster than
        reprocessing.
    Args:
        fl (str,list,np.array) - filelist for processing. If a string, should
            be the path to the filelist containing each file name on a seperate
            line.
        temp_file - filepath to the temp file generated as a failsafe
        fl_as_array (boolean) - true if the filelist is given as a list or
            numpy array
    Returns:
        files - an array of files remaining to be processed
        df - a datUnpicklingErroraframe of successfully processed files
        """
    if fl_as_array:
        files = fl
    else:
        with open(fl, 'r') as file:
            files = file.readlines()

    df = pd.DataFrame()
    with open(temp_file, 'rb') as fr:
        try:
            while True:
                df = df.append(pickle.load(fr))
        except EOFError:
            pass
    # resave the temp file as a singular entry for easier processing later if
    # need be.
    pickle.dump(df, open(temp_file, 'wb'))
    """
    Dropping files that have already been processed from the files
    """
    # Create a copy of df to manipulate
    dfc = df.copy()
    # create an empty dataframe with all the file names as indices (not just
    # completed)
    dff = pd.DataFrame(index=files)
    dfc = dfc.append(dff)
    files = np.array(dfc[~dfc.index.duplicated(keep=False)].index)

    with open(fl.replace('.txt', "")+"_completed.txt", 'a') as completed:
        completed.writelines(df.index)

    return files, df


@njit  # numba needs to update for python 3.9
def easy_feats(t, nf, err):
    feats_dict = {}
    nf_mean = np.mean(nf)
    nf_med = np.median(nf)
    feats_dict["stds"] = stds = np.std(nf)  # F6
    feats_dict["meanmedrat"] = meanmedrat = nf_mean / nf_med  # F2
    feats_dict["varss"] = varss = np.var(nf)  # F4
    feats_dict["coeffvar"] = coeffvar = stds/nf_mean  # F5

    posthreshold = nf_mean+4*stds
    negthreshold = nf_mean-4*stds

    feats_dict["numout1s"] = numout1s = len(nf[np.abs(nf-nf_mean) > stds])
    feats_dict["numposoutliers"] = numposoutliers = len(nf[nf > posthreshold])
    feats_dict["numnegoutliers"] = numnegoutliers = len(nf[nf < negthreshold])
    # F10
    feats_dict["numoutliers"] = numoutliers = numposoutliers + numnegoutliers

    feats_dict["mad"] = mad = np.median(np.abs(nf-nf_med))

    # delta nf/delta t
    slopes = (nf[1:]-nf[:-1])/(t[1:]-t[:-1])
    meanslope = np.mean(slopes)

    # Separating positive slopes and negative slopes
    # Should both include the 0 slope? It doesn't matter for calculating the
    # means later on...
    pslope = slopes[slopes >= 0]
    nslope = slopes[slopes <= 0]
    # Looking at the average (mean) positive and negative slopes
    if len(pslope) == 0:
        feats_dict["meanpslope"] = meanpslope = 0
    else:
        feats_dict["meanpslope"] = meanpslope = np.mean(pslope)  # F15

    if len(nslope) == 0:
        feats_dict["meannslope"] = meannslope = 0
    else:
        feats_dict["meannslope"] = meannslope = np.mean(nslope)  # F16

    # Quantifying the difference in shape.
    # if meannslope==0 (i.e., if there are no negative slopes), g_asymm is
    # assigned a value of 10
    # This value is chosen such that
    # a) it is positive (where g_asymm is inherently negative),
    # b) it is a factor larger than a random signal would produce (roughly
    #   equal average of positive and negative slopes -> g_asymm=-1)
    # c) it is not orders of magnitude larger than other data, which would
    #   affect outlier analysis
    # ! DG: I don't like this feature
    if meannslope == 0:
        feats_dict["g_asymm"] = g_asymm = 10
    else:
        feats_dict["g_asymm"] = g_asymm = meanpslope / meannslope  # F17

    # ? Won't this be skewed by the fact that both pslope and nslope have all
    # ? the 0's? Eh
    # F18
    if len(nslope) == 0:
        feats_dict["rough_g_asymm"] = rough_g_asymm = 10
    else:
        feats_dict["rough_g_asymm"] = rough_g_asymm = len(pslope) / len(nslope)

    # meannslope is inherently negative, so this is the difference btw the 2
    feats_dict["diff_asymm"] = diff_asymm = meanpslope + meannslope  # F19

    absslopes = np.abs(slopes)
    feats_dict["meanabsslope"] = meanabsslope = np.mean(absslopes)  # F21
    feats_dict["varabsslope"] = varabsslope = np.var(absslopes)  # F22
    feats_dict["varslope"] = varslope = np.var(slopes)  # F23

    """
    secder = Second Derivative
    Reminder for self: the slope is "located" halfway between the flux and
    time points, so the delta t in the denominator is accounting for that.
    secder = delta slopes/delta t, delta t = ((t_j-t_(j-1))+(t_(j+1)-t_j))/2
    secder=[(slopes[j]-slopes[j-1])/((t[j+1]-t[j])/2+(t[j]-t[j-1])/2) for j
    in range(1, len(slopes)-1)]
    after algebraic simplification:
    """
    secder = 2*(slopes[1:]-slopes[:-1])/(t[1:-1]-t[:-2])

    # abssecder=[abs((slopes[j]-slopes[j-1])/((t[j+1]-t[j])/2+(t[j]-t[j-1])/2))
    # for j in range (1, len(slopes)-1)] simplification:

    abssecder = np.abs(secder)
    feats_dict["absmeansecder"] = absmeansecder = np.mean(abssecder)  # F24

    if len(pslope) == 0:
        pslopestds = 0
    else:
        pslopestds = np.std(pslope)

    if len(nslope) == 0:
        nslopestds = 0
        stdratio = 10
    else:
        nslopestds = np.std(nslope)
        stdratio = pslopestds/nslopestds

    sdstds = np.std(secder)
    meanstds = np.mean(secder)

    feats_dict["num_pspikes"] = num_pspikes = (
        len(slopes[slopes >= meanpslope+3*pslopestds]))  # F25
    feats_dict["num_nspikes"] = num_nspikes = (
        len(slopes[slopes <= meannslope-3*nslopestds]))  # F26

    # 5/30/18, discovered a typo here. meanslope was missing an 'n', i.e. all
    # data processed prior to this date has num_nspikes defined as
    # meanslope-3*nslopestds which will overestimate the number of negative
    # spikes since meanslope is inherently greater than meannslope.

    feats_dict["num_psdspikes"] = num_psdspikes = (
        len(secder[secder >= meanstds+4*sdstds]))  # F27
    feats_dict["num_nsdspikes"] = num_nsdspikes = (
        len(secder[secder <= meanstds-4*sdstds]))  # F28
    # ! DG: I don't like ratio features
    if nslopestds == 0:
        feats_dict["stdratio"] = stdratio = 10
    else:
        feats_dict["stdratio"] = stdratio = pslopestds / nslopestds  # F29

    # where positive slopes are followed by another positive slope
    pairs = np.where((slopes[1:] > 0) & (slopes[:-1] > 0))[0]
    # The ratio of postive slopes with a following postive slope to the total
    # number of points.
    feats_dict["pstrend"] = pstrend = len(pairs)/len(slopes)  # F30

    plusminus = np.where((slopes[1:] < 0) & (slopes[:-1] > 0))[0]
    feats_dict["num_pm"] = num_pm = len(plusminus)

    # This looks up the local maximums. Adds a peak if it's the largest within
    # 10 points on either side.
    # ? Q: Is there a way to do this and take into account drastically
    # ? different periodicity scales?
    # TODO: More features with different periodicity scales in mind?

    naivemax, nmax_times, nmax_inds = [], [], []
    naivemins, nmin_times, nmin_inds = [], [], []
    for j in range(len(nf)):
        nfj = nf[j]
        if j-10 < 0:
            jmin = 0
        else:
            jmin = j-10
        if j+10 > len(nf)-1:
            jmax = len(nf-1)
        else:
            jmax = j+10

        max_nf = nf[jmin]
        min_nf = nf[jmin]
        for k in range(jmin, jmax):
            if nf[k] >= max_nf:
                max_nf = nf[k]
            elif nf[k] <= min_nf:
                min_nf = nf[k]

        if nf[j] == max_nf:
            if len(nmax_inds) > 0:
                if j-nmax_inds[-1] > 10:
                    naivemax.append(nf[j])
                    nmax_times.append(t[j])
                    nmax_inds.append(j)
            else:
                naivemax.append(nf[j])
                nmax_times.append(t[j])
                nmax_inds.append(j)
        elif nf[j] == min_nf:
            if len(nmin_inds) > 0:
                if j-nmin_inds[-1] > 10:
                    naivemins.append(nf[j])
                    nmin_times.append(t[j])
                    nmin_inds.append(j)
            else:
                naivemins.append(nf[j])
                nmin_times.append(t[j])
                nmin_inds.append(j)

    naivemax = np.array(naivemax)
    nmax_times = np.array(nmax_times)
    nmax_inds = np.array(nmax_inds)
    naivemins = np.array(naivemins)
    nmin_times = np.array(nmin_times)
    nmin_inds = np.array(nmin_inds)

    feats_dict["len_nmax"] = len_nmax = len(naivemax)  # F33
    feats_dict["len_nmin"] = len_nmin = len(naivemins)  # F34

    ppslopes = (
        np.abs(naivemax[1:]-naivemax[:-1]) / (nmax_times[1:]-nmax_times[:-1])
    )

    if len(ppslopes) == 0:
        feats_dict["ptpslopes"] = ptpslopes = 0
    else:
        feats_dict["ptpslopes"] = ptpslopes = np.mean(ppslopes)  # F36

    maxdiff = nmax_times[1:]-nmax_times[:-1]

    emin = naivemins[::2]  # even indice minimums
    omin = naivemins[1::2]  # odd indice minimums
    meanemin = np.mean(emin)
    if len(omin) == 0:
        meanomin = 0
    else:
        meanomin = np.mean(omin)
    # ! DG: I don't like ratio features, use meanomin and meanemin instead
    feats_dict["oeratio"] = oeratio = meanomin / meanemin  # F42

    # measures the slope before and after the maximums
    # reminder: 1 fewer slopes than fluxes, slopes start after first flux
    # slope[0] is between flux[0] and flux[1]
    # mean of slopes before max will be positive
    # mean of slopes after max will be negative

    nmax_inds_subset = nmax_inds[(nmax_inds > 5) & (nmax_inds < len(slopes)-5)]
    flatness = np.zeros(len(nmax_inds_subset))
    for i, j in enumerate(nmax_inds_subset):
        flatness[i] = np.mean(slopes[j-6:j-1])-np.mean(slopes[j:j+5])

    if len(flatness) == 0:
        feats_dict["flatmean"] = flatmean = 0
    else:
        feats_dict["flatmean"] = flatmean = np.mean(flatness)  # F55

    # measures the slope before and after the minimums
    # trying flatness w slopes and nf rather than "corr" vals, despite orig
    # def in RN's program
    # mean of slopes before min will be negative
    # mean of slopes after min will be positive

    nmin_inds_subset = nmin_inds[(nmin_inds > 5) & (nmin_inds < len(slopes)-5)]
    tflatness = np.zeros(len(nmin_inds_subset))
    for i, j in enumerate(nmin_inds_subset):
        tflatness[i] = -np.mean(slopes[j-6:j-1])+np.mean(slopes[j:j+5])

    # tflatness for mins, flatness for maxes
    if len(tflatness) == 0:
        feats_dict["tflatmean"] = tflatmean = 0
    else:
        feats_dict["tflatmean"] = tflatmean = np.mean(tflatness)  # F56

    # reminder: 1 less second derivative than slope (2 less than flux).
    # secder starts after first slope.
    # secder[0] is between slope[0] and slope[1], centered at flux[1]

    nmax_inds_subset = nmax_inds[(nmax_inds > 5) & (nmax_inds < len(secder)-5)]

    roundness = np.zeros(len(nmax_inds_subset))
    for i, j in enumerate(nmax_inds_subset):
        roundness[i] = np.mean(secder[j-6:j+6])*2

    if len(roundness) == 0:
        feats_dict["roundmean"] = roundmean = 0
    else:
        feats_dict["roundmean"] = roundmean = np.mean(roundness)  # F57

    nmin_inds_subset = nmin_inds[(nmin_inds > 5) & (nmin_inds < len(secder)-5)]
    troundness = np.zeros(len(nmin_inds_subset))
    for i, j in enumerate(nmin_inds_subset):
        troundness[i] = np.mean(secder[j-6:j+6])*2

    if len(troundness) == 0:
        feats_dict["troundmean"] = troundmean = 0
    else:
        feats_dict["troundmean"] = troundmean = np.mean(troundness)  # F58
    # ! DG: I don't like ratio features
    if troundmean == 0 and roundmean == 0:
        feats_dict["roundrat"] = roundrat = 1
    elif troundmean == 0 and roundmean > 0:
        feats_dict["roundrat"] = roundrat = 123456
    elif troundmean == 0 and roundmean < 0:
        feats_dict["roundrat"] = roundrat = -123456
    else:
        feats_dict["roundrat"] = roundrat = roundmean / troundmean  # F59
    # ! DG: I don't like ratio features.
    if flatmean == 0 and tflatmean == 0:
        feats_dict["flatrat"] = flatrat = 1
    elif tflatmean == 0:
        # Flatness ratio is (almost) inherently positive has an average value
        # of 1. If there are no minima, a flatness ratio of -10 is assigned.
        # See reasoning for g_asymm.
        feats_dict["flatrat"] = flatrat = -10
    else:
        feats_dict["flatrat"] = flatrat = flatmean / tflatmean  # F60

    return feats_dict, naivemax, maxdiff


def fancy_feats(t, nf, err, naivemax, maxdiff):
    feats_dict = {}
    nf_mean = np.mean(nf)
    nf_med = np.median(nf)

    # fancy meaning I can't throw these under a jit decorator.
    # Feature 1 (Abbr. F1) overall slope
    feats_dict["longtermtrend"] = longtermtrend = np.polyfit(t, nf, 1)[0]  # F1
    yoff = np.polyfit(t, nf, 1)[1]  # Not a feature, y-intercept of linear fit
    feats_dict["skews"] = skews = stats.skew(nf)  # F3
    # this removes the linear trend of lc so you can look at just troughs
    corrnf = nf - longtermtrend*t - yoff

    feats_dict["kurt"] = kurt = stats.kurtosis(nf)  # F11

    slopes = (nf[1:]-nf[:-1])/(t[1:]-t[:-1])

    # by looking at where the 99th percentile is instead of just the largest
    # number, I think it avoids the extremes which might not be relevant
    # (might be unreliable data)
    # Is the miniumum slope the most negative one, or the flattest one?
    # Answer: Most negative
    feats_dict["maxslope"] = maxslope = np.percentile(slopes, 99)  # F13
    feats_dict["minslope"] = minslope = np.percentile(slopes, 1)  # F14

    # corrslopes (corrected slopes) removes the longterm linear trend (if any)
    # and then looks at the slope
    corrslopes = (corrnf[1:]-corrnf[:-1])/(t[1:]-t[:-1])
    feats_dict["skewslope"] = skewslope = stats.skew(slopes)  # F20

    # Checks if the flux crosses the 'zero' line.
    zcrossind = np.where(corrnf[:-1]*corrnf[1:] < 0)[0]
    feats_dict["num_zcross"] = num_zcross = len(zcrossind)  # F31

    if len(naivemax) > 2:
        feats_dict["mautocorrcoef"] = mautocorrcoef = (
            np.corrcoef(naivemax[:-1], naivemax[1:])[0][1])  # F35
    else:
        feats_dict["mautocorrcoef"] = mautocorrcoef = 0

    if len(maxdiff) == 0:
        feats_dict["periodicity"] = periodicity = 0
        feats_dict["periodicityr"] = periodicityr = 0
        feats_dict["naiveperiod"] = naiveperiod = 0
    else:
        feats_dict["periodicity"] = periodicity = (
            np.std(maxdiff)/np.mean(maxdiff))  # F37
        feats_dict["periodicityr"] = periodicityr = (
            np.sum(abs(maxdiff-np.mean(maxdiff)))/np.mean(maxdiff)  # F38
        )
        feats_dict["naiveperiod"] = naiveperiod = np.mean(maxdiff)  # F39
    if len(naivemax) == 0:
        feats_dict["maxvars"] = maxvars = 0
        feats_dict["maxvarsr"] = maxvarsr = 0
    else:
        # F40
        feats_dict["maxvars"] = maxvars = np.std(naivemax)/np.mean(naivemax)
        feats_dict["maxvarsr"] = maxvarsr = (
            np.sum(abs(naivemax-np.mean(naivemax)))/np.mean(naivemax)  # F41
        )

    # amp here is actually amp_2 in revantese
    # 2x the amplitude (peak-to-peak really)
    feats_dict["amp"] = amp = np.percentile(nf, 99)-np.percentile(nf, 1)  # F43
    # this should prob go, since flux is norm'd
    feats_dict["normamp"] = normamp = amp / nf_mean  # F44
    feats_dict["autocorrcoef"] = autocorrcoef = (
        np.corrcoef(nf[:-1], nf[1:])[0][1])  # F54

    feats_dict["sautocorrcoef"] = sautocorrcoef = (
        np.corrcoef(slopes[:-1], slopes[1:])[0][1])  # F55
    # ratio of points within one fifth of the amplitude to the median to total
    # number of points
    feats_dict["mbp"] = mbp = (
        len(nf[(nf <= (nf_med+0.1*amp)) & (nf >= (nf_med-0.1*amp))]) / len(nf)
    )  # F45

    f595 = np.percentile(nf, 95)-np.percentile(nf, 5)
    f1090 = np.percentile(nf, 90)-np.percentile(nf, 10)
    f1782 = np.percentile(nf, 82)-np.percentile(nf, 17)
    f2575 = np.percentile(nf, 75)-np.percentile(nf, 25)
    f3267 = np.percentile(nf, 67)-np.percentile(nf, 32)
    f4060 = np.percentile(nf, 60)-np.percentile(nf, 40)
    feats_dict["mid20"] = mid20 = f4060 / f595  # F46
    feats_dict["mid35"] = mid35 = f3267 / f595  # F47
    feats_dict["mid50"] = mid50 = f2575 / f595  # F48
    feats_dict["mid65"] = mid65 = f1782 / f595  # F49
    feats_dict["mid80"] = mid80 = f1090 / f595  # F50
    # F51
    feats_dict["percentamp"] = percentamp = max(np.abs(nf-nf_med)/nf_med)

    feats_dict["magratio"] = magratio = (max(nf)-nf_med) / amp  # F52

    return feats_dict


def feats(t, nf, err):
    """
    Purpose:
        Calculates the features determined to be useful for machine learning
        applications to photometric data.
        Splits the computation into 2 categories, those which can be easily
        decorated with the njit decorator, and those which can be otherwise
        optimized.

    Args:
        t (np.array, dtype np.float64) - time
        nf (np.array, dtype np.float32) - normalized flux
        err (np.array, dtype np.float32) - normalized error

    returns:
        feats (dict, dtype varied) - the calculated features tied to their key
    """
    # features that were jit-able
    feats, naivemax, maxdiff = easy_feats(t, nf, err)

    # features that were not (easily) jit-able
    numpy_feats = fancy_feats(t, nf, err, naivemax, maxdiff)

    feats.update(numpy_feats)
    for key in feats.keys():
        assert key in fts, f"{key} is an unrecognized feature"
    for ft in fts:
        assert ft in feats.keys(), f"{ft} not found in features, check code."
    return feats


def feature_calc(lc):
    """
    Purpose:
        This method is for use when processing multiple lightcurves. It takes
        a single lightcurve of data at a time, processes it,
        dumps the results as an indexed dataframe into a tmp file as a
        failsafe, and returns the file name as well as the features in a
        list to be parsed.

    Args:
        lc (list) - a list containing:
            lc[0] (str) - nfile - the name of the file being processed
            lc[1] (list/np.array) - t - array of times for lightcurve
            lc[2] (list/np.array) - nf - array of normalized fluxes for
                lightcurve
            lc[3] (list/np.array) - err - array of normalized errors for
                lightcurve
    Note: requires a global variable "tmpfile" to be defined, this should be a
        string specifying where data should be temporarily stored.
    Returns:
        nfile - the name of the file sans path
        ndata - the feature data
    """
    nfile = lc.meta["filename"]
    t = lc.time.value
    # lc.flux should default to pdcsap_flux for Kepler files.
    # Same for the tess-goddard light curves.
    nf = lc.flux.value
    err = lc.flux_err.value

    try:
        features = feats(
            t,
            nf,
            err
            )

        # A failsafe, dumps the data to a temp file in case of failure during
        # processing of another lightcurve.
        # Reading in the temp file is a huge pain though.
        df = pd.DataFrame(
            data=[features],
            index=[lc.targetid]
            )
        # with open(tmpfile, 'ab') as f:
        #     pickle.dump(df, f)
        return df

    except TypeError:
        kml_log = 'kml_log'
        os.system(f"""
        echo {nfile} ... TYPE ERROR >> {kml_log}
        """)
        return


def join_data(full_features_array):
    file = []
    data = []

    for i in full_features_array:
        file.append(i[0])
        data.append(i[1])
    df = pd.DataFrame(data, index=file, columns=fts)
    return df


def features_from_filelist(lcs,
                           of,
                           numCpus=cpu_count(),
                           verbose=False,
                           tmp_file='tmp_data.pkl',
                           useCpus=1):
    """
    Purpose:
        This method calculates the features of the given filelist from the
            fits files located in fitsDir.
        All output is saved to a pickle file called tmp_data.p (or
            whatever is specified).
            Run clean_up(filelist, fits/file/directory)
            and save_output('output/file/path') to clean up the filelist
            (makes a completed filelist) and tosave to the desired location.
            Note: save_output() replaces tmp_data.p
    Args:
        fl (string or list) - filelist as a path to a text file containing the
            files to be processed, one per line, or a list of the files
        fitDir (string) - path to the fits directory, if path is specified in
            the filelist, give this a value of ''
        of (string) - output file, the path to the file where the output
            should be saved as a pickle
        fl_as_array (boolean) - if True specifies that fl is given as a list,
            if False specifies that fl is a filepath
        numCpus (integer) - Number of cpus on which to process the files
        verbose (boolean) - if True will output progress statements
        tmp_file (string) - specifies where to save the temporary data. Useful
            if processing multiple sets of data in parallel.

    Returns:
        Pandas dataframe of output for all files in the filelist
    """
    startTime = datetime.now()
    # These are variables needed in other methods that need to be pickled for
    # mutliprocessing, which can only take a single iterable as an argument.
    # There's probably a better way to do this.

    global tmpfile
    tmpfile = tmp_file

    if verbose:
        print("Lightcurve import took {}".format(datetime.now()-startTime))
    df = pd.DataFrame(columns=fts)  # fts is defined globally

    # Splitting up lcs into 10k lc chunks so that progress can be monitored
    # AND to dump out results periodically.
    nlcs = len(lcs)
    nchunks = int(nlcs/10000)
    len_last_chunk = nlcs % 10000
    if verbose:
        print(f"Processing {nlcs} files...")
    featsStartTime = datetime.now()
    for i in range(nchunks):
        chunkStartTime = datetime.now()
        p = Pool(useCpus)
        # Method saves to tmp_data.pkl file after processing each lightcurve as
        # a failsafe.
        full_features = p.map(feature_calc, lcs[i*10000:(i+1)*10000])
        p.close()
        p.join()

        df = df.append(join_data(full_features))
        with open(tmpfile, 'wb') as file:
            # overwrites failsafe with single dataframe containing all
            # processed data, easier to process if system failure later.
            pickle.dump(df, file)
        del(full_features)

        if verbose:
            chunktime = datetime.now()-chunkStartTime
            print(
                f"""
                {(i+1)*10000}/{nlcs} completed. Time for chunk: {chunktime}
                Est. time remaining: {chunktime*(nchunks-i+1)/nchunks}
                """
                )

    # last chunk
    p = Pool(useCpus)
    full_features = p.map(feature_calc, lcs[nlcs-len_last_chunk:])
    p.close()
    p.join()
    df = df.append(join_data(full_features))
    del(full_features)

    if verbose:
        print(
            f"""
            Features have been calculated, total time to calculate features: {
                datetime.now()-featsStartTime
                }
            """
        )
        print(f"Saving output to {of}")
    with open(of, 'wb') as file:
        pickle.dump(df, file)
    # explicitly freeing memory
    # memory has been an issue when running sequentially
    if verbose:
        print("Cleaning up...")
    os.remove(tmpfile)
    del(lcs, p, df)
    if verbose:
        print("Done.")

    if __name__ == "__main__":
        return
    else:
        with open(of, 'rb') as file:
            feats = pickle.load(file)
        return feats


def main(args):
    start = datetime.now()
    fl = args.fl
    global fitsDir
    fitsDir = args.fits_dir
    of = args.of
    numCpus = args.ncpus

    if numCpus == -1:
        useCpus = cpu_count()-1
    else:
        useCpus = min([cpu_count()-1, max(numCpus, 1)])

    lcs = load_filelist(fl, fitsDir, verbose=True, useCpus=useCpus)
    features_from_filelist(fl, fitsDir, of, verbose=True)
    print(datetime.now()-start)
    pass


if __name__ == "__main__":
    """
    If this is run as a script, the following will parse the arguments it's fed
    or prompt the user for input.

    python keplerml.py path/to/filelist
        path/to/fits_file_directory path/to/output_file
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fl",
        type=str,
        default="./filelist.txt",
        help="Document with fits file paths to process."
    )

    parser.add_argument(
        "--fits-dir",
        type=str,
        default="~/data/",
        help="Directory to locally access light curve data from",
    )

    parser.add_argument(
        "--of",
        type=str,
        default="./data/output.pkl",
        help="File to save results to.",
    )

    parser.add_argument(
        "--ncpus",
        type=int,
        default=1,
        help="number of cpus to use"
    )

    args = parser.parse_args()

    main(args)
    print(f"Features calculated successfully, saved to {args.of}.")
