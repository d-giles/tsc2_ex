import lightkurve as lk
from multiprocessing import Pool
import numpy as np
import pickle
import pandas as pd
import os
import data


class LightCurveCollection:
    def __init__(
            self,
            sec: list,
            cam: list,
            files: list,
            factor: int = 10,
    ):
        self.sector = sec
        self.camera = cam
        self.files = files
        self.ref = pd.DataFrame(
            data={"sector": sec, "camera": cam, "file": files}
            )
        self.factor = factor
        self.useCpus = 1
        self.mad_table = data.load_mad()
        self.badtimes = data.load_bad_times()
        self.__load_scmad()
        self.__calc_cutmask()
        self.__calc_log_weight()
        self.loaded = False
        self.__make_summary()
        return

    def __make_summary(self):
        self.summary = {
            "Sector": self.sector,
            "Camera": self.camera,
            "Count": len(self.files),
            "Loaded": self.loaded
        }
        return self.summary

    def __load_scmad(self):
        sc_mad = dict()
        if "mad_table" not in locals():
            mad_table = self.mad_table
        sectors = np.unique(self.sector)
        cams = np.unique(self.camera)
        for sector in sectors:
            for cam in cams:
                subref = self.ref[(self.ref.sector == sector)
                                  & (self.ref.camera == cam)]
                if len(subref) == 0:
                    pass
                else:
                    f0 = subref.file.values[0]
                    lc0 = load_lc(f0)
                    sc_mad[f'{sector}-{cam}'] = mad_table.loc[
                        :len(lc0)-1,
                        f"{sector}-{cam}"
                    ]
        self.sc_mad = sc_mad
        return sc_mad

    def __calc_cutmask(self):
        self.cutmask = dict()

        sectors = np.unique(self.sector)
        cams = np.unique(self.camera)
        for sector in sectors:
            for cam in cams:
                subref = self.ref[(self.ref.sector == sector)
                                  & (self.ref.camera == cam)]
                if len(subref) == 0:
                    pass
                else:
                    sc = f'{sector}-{cam}'
                    cut = (np.mean(self.sc_mad[sc])
                           + (self.factor
                              * np.std(self.sc_mad[sc]))
                           )
                    self.cutmask[sc] = self.sc_mad[sc] < cut
        return self.cutmask

    def __calc_log_weight(self):
        # lookup MAD values for appropriate sector/camera combination
        self.mad_scaler = dict()

        sectors = np.unique(self.sector)
        cams = np.unique(self.camera)
        for sector in sectors:
            for cam in cams:
                subref = self.ref[(self.ref.sector == sector)
                                  & (self.ref.camera == cam)]
                if len(subref) == 0:
                    pass
                else:
                    sc = f'{sector}-{cam}'
                    # create the scaling array, log scaled in this instance,
                    # inverted so large MAD cadences are suppressed
                    sc_mad_loginv = -np.log(self.sc_mad[sc])
                    self.mad_scaler[sc] = np.array(
                        (sc_mad_loginv-sc_mad_loginv.min())
                        / (sc_mad_loginv.max() - sc_mad_loginv.min())
                    )

        return self.mad_scaler

    def load_cut_lc(self, miniref: pd.DataFrame):
        """Load a masked light curve.
        """
        f = miniref.file
        sec = miniref.sector
        cam = miniref.camera
        msg = "File must be in the list this object was instantiated with."
        assert f in self.files, msg
        # ! Not masked initially so that it lcs have consistent length
        # ! with the cutmask from the MAD array
        lc = load_lc(f)
        sc = f"{sec}-{cam}"
        # all the masks
        lc = lc[self.cutmask[sc].values & (lc.quality == 0)].remove_nans()
        if np.nanmedian(lc.flux) < 0:
            # A negative median results from bad background subtraction
            # Normalizing (w/o inverting) makes the median -1, so we translate
            # the light curve up to make the median positive.
            lc.flux += 2*abs(np.nanmedian(lc.flux))

        nfluxes = np.array(lc.flux/abs(np.nanmedian(lc.flux)))
        lc.flux = nfluxes
        

        for i in self.badtimes:
            lc = lc[
                (lc.time.value+2457000 < i[0]) 
                | (lc.time.value+2457000 > i[1])
            ]

        lc_copy = lc[lc.flux >= 0]  # one final mask

        assert len(lc_copy) > 100, f"Check TIC {lc_copy.targetid}"

        return lc_copy

    def load_weighted_lc(self, miniref):
        f = miniref.file
        sec = miniref.sector
        cam = miniref.camera

        # import all light curves for the sector/camera combination
        lc = load_lc(f)
        sc = f"{sec}-{cam}"
        nfluxes = np.array(lc.flux/np.nanmedian(lc.flux))
        nfluxes = (nfluxes-1)*self.mad_scaler[sc]+1  # scaled fluxes
        lc.flux = nfluxes
        mask = (lc.quality == 0) & (lc.flux >= 0)
        lc = lc[mask].remove_nans()
        return lc

    def load_all_lcs(self, method: str = 'cut'):
        """Load all light curves specified in self.files
        """
        if self.useCpus == 1:
            if method == "cut":
                self.lcs = [self.load_cut_lc(self.ref.iloc[i])
                            for i in range(len(self.ref))]
            elif method == "log_w":
                self.lcs = [self.load_weighted_lc(self.ref.iloc[i])
                            for i in range(len(self.ref))]
        else:
            # todo: update for weighted
            with Pool(self.useCpus) as p:
                minirefs = [self.ref.iloc[i] for i in range(len(self.ref))]
                self.lcs = p.map(self.load_cut_lc, minirefs, chunksize=50)
        # with Pool(useCpus) as p:
        #    lcs = p.map(load_weighted_lc, files, chunksize=100)
        self.loaded = True
        return


def load_lc(fp, fluxtype="PDC", mask=False):
    """Load light curve data from pickle file into a lightkurve object
    Args:
        fp (str) - file path to pickle file in standard format
        fluxtype (str) - Type of flux to prioritize,
            choose between "raw", "corr", and "PDC"
        mask (bool) - Mask data points non-zero flags in quality

    returns:
        lc (lightkurve.lightcurve.LightCurve) - a LightCurve object
    """

    with open(fp, 'rb') as file:
        lc_list = pickle.load(file)

    fluxes = {"raw": lc_list[7], "corr": lc_list[8], "PDC": lc_list[9]}

    try:
        flux = fluxes[fluxtype]

    except KeyError:
        print("""
        The flux type must be 'raw', 'corr', or 'PDC'. Defaulting to 'PDC'.""")
        flux = fluxes["PDC"]

    finally:
        time = lc_list[6]
        flux_err = lc_list[10]
        quality = lc_list[11]

        if mask:
            mask = lc_list[11] == 0
            flux = flux[mask]
            time = time[mask]
            flux_err = flux_err[mask]
            quality = quality[mask]  # just 0's if masked

        # for meta information
        fluxes.update(
            {"TESS Magnitude": lc_list[3], "filename": fp.split("/")[-1]})
        lc = lk.lightcurve.TessLightCurve(
            time=time, flux=flux, flux_err=flux_err, targetid=lc_list[0],
            quality=quality, camera=lc_list[4], ccd=lc_list[5],
            ra=lc_list[1], dec=lc_list[2], label=f"TIC {lc_list[0]}",
            meta=fluxes
        )

    return lc


# old name updated to PEP8 class convention
lc_collection = LightCurveCollection


def main():
    print("This module is intended to be used within other scripts.")
    return

if __name__ == "__main__":
    main()
