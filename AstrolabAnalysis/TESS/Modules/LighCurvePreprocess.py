import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from pathlib import Path


class LightCurvePreprocess:
    """
    Class to preprocess the TESS light curve
    """

    def __init__(self, file_lc_name, data_folder, results_folder, yaml_preprocess):
        """
        Args:
            file_lc_name (str): light curve filename
            data_folder (str): folder in which the data are located
            results_folder (str): folder in which to save results
            yaml_preprocess (todo): yaml dictionary (todo) with preprocess data
        """
        self.file_lc_name = file_lc_name
        self.data_folder = data_folder
        self.results_folder = results_folder
        self.yaml_preprocess = yaml_preprocess
        self.sector_number = None
        self.sap_flux = None
        self.sap_flux_error = None
        self.pdcsap_flux = None
        self.pdcsap_flux_error = None
        self.time_array = None
        self.conservative_selection = None

    def read_and_conservative_selection(self):
        """
        Read the lc file and plot the light curve with datapoint selected using a conservative approach (exclude all points that have any flag marked as True).
        """
        # Read the file and sector number
        lchdu = fits.open(str(Path(self.data_folder, self.file_lc_name)))
        self.sector_number = lchdu[0].header['SECTOR']
        # Get the SAP (Simple Aperture Photometry) and
        # PDCSAP (Pre-search Data Conditioning SAP)
        self.sap_flux = lchdu[1].data['SAP_FLUX']
        self.sap_flux_error = lchdu[1].data['SAP_FLUX_ERR']
        self.pdcsap_flux = lchdu[1].data['PDCSAP_FLUX']
        self.pdcsap_flux_error = lchdu[1].data['PDCSAP_FLUX_ERR']
        quality_bitmask = lchdu[1].data['QUALITY']
        # Convert from BTJD (Barycentric TESS Julian Date) to BJD
        self.time_array = lchdu[1].data['TIME'] + lchdu[1].header['BJDREFI'] + lchdu[1].header['BJDREFF']
        # Conservative selection
        finite_selection = np.isfinite(self.pdcsap_flux)
        self.conservative_selection = ~(quality_bitmask > 0) & finite_selection
        # Plot the fluxes
        plt.figure(figsize=(12.8, 7.2))
        plt.scatter(self.time_array[self.conservative_selection],
                    self.sap_flux[self.conservative_selection],
                    s=5, label='SAP - selected data')
        plt.scatter(self.time_array, self.pdcsap_flux, s=5, label='PDCSAP')
        plt.scatter(self.time_array[~self.conservative_selection],
                    self.sap_flux[~self.conservative_selection],
                    s=5, c='r', label='SAP - excluded data')
        plt.errorbar(self.time_array[self.conservative_selection],
                     self.sap_flux[self.conservative_selection],
                     yerr=self.sap_flux_error[self.conservative_selection],
                     fmt=' ', alpha=0.5, ecolor='k', zorder=-1)
        plt.xlabel('BJD_TDB [d]')
        plt.ylabel('e-/s')
        plt.title("TESS Lightcurve for GJ3470 - sector {0:d}".format(self.sector_number))
        plt.legend()
        plt.savefig(str(Path(self.results_folder,
                             "sector{0:d}_sap-pscsap_raw.png".format(self.sector_number))))
        plt.show()

    def final_data_selection(self):
        """Manually exclude some points at the beginning or end of the time series"""
        if self.yaml_preprocess["low_lim"] is not None:
            final_selection = self.conservative_selection & (self.time_array > self.yaml_preprocess["low_lim"])
        elif self.yaml_preprocess["high_lim"] is not None:
            final_selection = self.conservative_selection & (self.time_array < self.yaml_preprocess["high_lim"])
        else:
            final_selection = self.conservative_selection
            # Apply the final selection on the data
            self.time_array = self.time_array[final_selection]
            self.sap_flux = self.sap_flux[final_selection]
            self.sap_flux_error = self.sap_flux_error[final_selection]
            self.pdcsap_flux = self.pdcsap_flux[final_selection]
            self.pdcsap_flux_error = self.pdcsap_flux_error[final_selection]
            return
        # Plot a zoom of the manually excluded data
        plt.figure(figsize=(12.8, 7.2))
        plt.scatter(self.time_array[self.conservative_selection],
                    self.sap_flux[self.conservative_selection],
                    s=9, label='SAP - selected data')
        plt.scatter(self.time_array, self.pdcsap_flux, s=5, label='PDCSAP')
        plt.scatter(self.time_array[~self.conservative_selection],
                    self.sap_flux[~self.conservative_selection],
                    s=9, c='r', label='SAP - excluded data')
        plt.scatter(self.time_array[~final_selection & self.conservative_selection],
                    self.sap_flux[~final_selection & self.conservative_selection],
                    s=30, c='y', marker='x', label='SAP - manually excluded')
        plt.errorbar(self.time_array[self.conservative_selection],
                     self.sap_flux[self.conservative_selection],
                     yerr=self.sap_flux_error[self.conservative_selection],
                     fmt=' ', alpha=0.5, ecolor='k', zorder=-1)
        plt.xlabel('BJD_TDB [d]')
        plt.ylabel('e-/s')
        plt.title("TESS Lightcurve for GJ3470 - sector {0:d}".format(self.sector_number),
                  fontsize=12)
        plt.xlim(self.yaml_preprocess["xlim"])
        plt.ylim(self.yaml_preprocess["ylim"])
        plt.legend()
        plt.savefig(str(Path(self.results_folder,"sector{0:d}_sap-pscsap_final-selection_zoom.png".format(self.sector_number))))
        plt.show()
        plt.close()
        # Apply the final selection on the data
        self.time_array = self.time_array[final_selection]
        self.sap_flux = self.sap_flux[final_selection]
        self.sap_flux_error = self.sap_flux_error[final_selection]
        self.pdcsap_flux = self.pdcsap_flux[final_selection]
        self.pdcsap_flux_error = self.pdcsap_flux_error[final_selection]

    def execute_preprocess(self):
        """
        Execute the preprocess of TESS light curve files
        """
        self.read_and_conservative_selection()
        self.final_data_selection()
        return self.time_array, self.sap_flux, self.sap_flux_error, self.pdcsap_flux, self.pdcsap_flux_error, self.sector_number
