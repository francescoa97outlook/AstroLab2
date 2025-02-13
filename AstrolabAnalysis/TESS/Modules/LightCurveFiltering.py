import numpy as np
import matplotlib.pyplot as plt
from wotan import flatten, transit_mask
from pathlib import Path

class LightCurveFiltering:
    """Class that filters the data to empirically remove instrumental and astrophysical unwanted signals"""
    
    def __init__(self, time, sap_flux, sap_flux_error, pdcsap_flux,
                 pdcsap_flux_error, orbital_params, yaml_sector,
                 sector_number, results_folder):
        """
        Args:
            time: time array
            sap_flux: Simple Aperture Photometry array
            sap_flux_error: errors associated to the sap flux array
            pdcsap_flux: Pre-search Data Conditioning SAP flux array
            pdcsap_flux_error: errors associated to the pdcsap flux array
            orbital_params: yaml dictionary containig the orbital parameters of the analyzed planet
            yaml_sector: yaml dictionary containig the methods and window apertures to do the filtering process
            sector_number: TESS sector number considered
            results_folder (str): string which contain the location of the folder where to save results
        """
        self.time = time
        self.sap_flux = sap_flux
        self.sap_flux_error = sap_flux_error
        self.pdcsap_flux = pdcsap_flux
        self.pdcsap_flux_error = pdcsap_flux_error
        # self.orbital_params = orbital_params
        self.transit_time = orbital_params["transit_time"]
        self.period = orbital_params["period"]
        self.transit_window = orbital_params["transit_duration"] * 2 / 24
        self.phase_folded_time = (self.time - self.transit_time - self.period / 2) % self.period - self.period / 2
        self.mask = transit_mask(
            time=self.time,
            period=self.period,
            duration=self.transit_window,
            T0=self.transit_time
            )
        self.yaml_sector = yaml_sector
        self.filter_list = self.yaml_sector["filter_list"]
        self.break_tolerance = 0.5
        self.sector_number = sector_number
        self.results_folder = results_folder

    def filter_flux(self, time, flux, method, wl, bt, mask=None):
        """
        Function that returns the filtered flux and associated error.

        Args:
            time: time array
            flux: flux array
            method (str): filtering algorithm used
            wl (float): window length
            bt (float): break tolerance
            mask: mask to exclude some points from the filtering
        """
        flatten_flux, flatten_model = flatten(
            time,
            flux,
            method=method,
            window_length=wl,
            break_tolerance=bt,
            return_trend=True,
            mask=mask
        )
        return flatten_flux, flatten_model

    def testing_filter(self):
        '''
        method = 'hspline'
        window_length = self.yaml_sector["testing_filter_window"]
        sap_flatten_flux, sap_flatten_model = self.filter_flux(
            self.time, self.sap_flux, method, window_length, self.break_tolerance)
        sap_masked_flatten_flux, sap_masked_flatten_model = self.filter_flux(
            self.time, self.sap_flux, method, window_length, self.break_tolerance,
            mask=self.mask)
        #
        fig1, ax1 = plt.subplots(2, 1, figsize=(9, 7))  #, layout='constrained')
        ax1[0].scatter(self.time, self.sap_flux, c='C0', s=3)
        ax1[0].errorbar(self.time, self.sap_flux, yerr=self.sap_flux_error, ecolor='k',
                        fmt=' ', alpha=0.25, zorder=-1)
        ax1[0].plot(self.time, sap_flatten_model, c='C1', zorder=10)
        ax1[0].set_xlabel('BJD_TDB')
        ax1[0].set_ylabel('TESS SAP flux [e-/s]')
        ax1[0].set_title('TESS: original lightcurve and flattening model')
        #
        ax1[1].scatter(self.time, sap_flatten_flux, c='C0', s=3)
        ax1[1].errorbar(self.time, sap_flatten_flux, yerr=self.sap_flux_error/sap_flatten_model,
                        ecolor='k', fmt=' ', alpha=0.25, zorder=-1)
        ax1[1].axhline(1.000, c='C1')
        ax1[1].set_xlabel('BJD_TDB')
        ax1[1].set_ylabel('Normalized flux')
        ax1[1].set_title('TESS: normalized SAP light curve')
        fig1.tight_layout()
        plt.savefig(str(Path(self.results_folder, "sector{0:d}_testing_flatten-{1}-wl{2:.1f}.png".format(
            self.sector_number, method, window_length))))
        #plt.show()
        plt.close()
        '''

    def plot_excluded_points(self):
        """
        Function to plot the SAP flux around the transit with pholded time
        """
        plt.figure(figsize=(9, 4))
        plt.scatter(self.phase_folded_time, self.sap_flux, s=2)
        plt.scatter(self.phase_folded_time[self.mask], self.sap_flux[self.mask],
                    s=2, c='r', zorder=1, label='excluded points')
        plt.xlabel('Time from mid-transit [d]')
        plt.ylabel('Normalized SAP flux')
        plt.xlim(-0.4, 0.4)
        plt.legend()
        plt.savefig(str(Path(self.results_folder, "sector{0}_filtering_excluded-points.png".format(
            self.sector_number))))
        plt.show()
        plt.close()

    def test_filter_algorithm(self):
        """
        Function to compute the filtered flux using different methods and window lengths
        """
        self.array_flat_sap_flux = np.empty(shape=(len(self.filter_list), len(self.sap_flux)))
        self.array_flat_sap_model = np.empty(shape=(len(self.filter_list), len(self.sap_flux)))
        self.array_flat_pdcsap_flux = np.empty(shape=(len(self.filter_list), len(self.pdcsap_flux)))
        self.array_flat_pdcsap_model = np.empty(shape=(len(self.filter_list), len(self.pdcsap_flux)))
        self.array_std_sap = np.empty(shape=(len(self.filter_list)))
        self.array_std_pdcsap = np.empty(shape=(len(self.filter_list)))
        for i, filter in enumerate(self.filter_list):
            # print("Flattening with method {0}, window={1:.1f}".format(
            #     self.filter_list[filter]["model"], self.filter_list[filter]["window"]
            # ))
            self.array_flat_sap_flux[i, :], self.array_flat_sap_model[i, :] = self.filter_flux(
                    self.time, self.sap_flux, self.filter_list[filter]["model"],
                    self.filter_list[filter]["window"], self.break_tolerance,
                    self.mask)
            self.array_flat_pdcsap_flux[i, :], self.array_flat_pdcsap_model[i, :] = self.filter_flux(
                    self.time, self.pdcsap_flux, self.filter_list[filter]["model"],
                    self.filter_list[filter]["window"], self.break_tolerance,
                    self.mask)
            # Calculate the standard deviation of the out-transit part of the
            # light curve
            self.array_std_sap[i] = np.std(self.array_flat_sap_flux[i, ~self.mask])
            self.array_std_pdcsap[i] = np.std(self.array_flat_pdcsap_flux[i, ~self.mask])
            # print("STD SAP {0}, window={1:.1f}: {2:.6f}".format(
            #     self.filter_list[filter]["model"], self.filter_list[filter]["window"],
            #     self.array_std_sap[i]
            # ))
            # print("STD PDCSAP {0}, window={1:.1f}: {2:.6f}".format(
            #     self.filter_list[filter]["model"], self.filter_list[filter]["window"],
            #     self.array_std_pdcsap[i]
            # ))
            # print('-----------')

    def best_algorithm_decider(self):
        """
        Function used to select the best filtering algorithm. The selection
        consists in taking the filtered pdcsap with the smallest standard
        deviation. Also plot a comparison of the different methods.
        """
        key_list = list(self.filter_list)
        len_one_filter = int(len(self.filter_list) / 4)
        #
        self.index_sap = np.argmin(self.array_std_sap[:len_one_filter])
        self.index_pdcsap = np.argmin(self.array_std_pdcsap[:len_one_filter])
        key_sap = key_list[self.index_sap]
        key_pdcsap = key_list[self.index_pdcsap]
        print("SAP: Sector {0}, best method={1}, window={2:.1f}, filtered std={3:.7f}".format(
            self.sector_number, self.filter_list[key_sap]["model"],
            self.filter_list[key_sap]["window"], self.array_std_sap[self.index_sap]
        ))
        print("PDCSAP: Sector {0}, best method={1}, window={2:.1f}, filtered std={3:.7f}".format(
            self.sector_number, self.filter_list[key_pdcsap]["model"],
            self.filter_list[key_pdcsap]["window"], self.array_std_pdcsap[self.index_pdcsap]
        ))
        #
        self.index_sap = np.argmin(self.array_std_sap[len_one_filter:]) + len_one_filter
        self.index_pdcsap = np.argmin(self.array_std_pdcsap[len_one_filter:]) + len_one_filter
        key_sap = key_list[self.index_sap]
        key_pdcsap = key_list[self.index_pdcsap]
        print("SAP: Sector {0}, best method={1}, window={2:.1f}, filtered std={3:.7f}".format(
            self.sector_number, self.filter_list[key_sap]["model"],
            self.filter_list[key_sap]["window"], self.array_std_sap[self.index_sap]
        ))
        print("PDCSAP: Sector {0}, best method={1}, window={2:.1f}, filtered std={3:.7f}".format(
            self.sector_number, self.filter_list[key_pdcsap]["model"],
            self.filter_list[key_pdcsap]["window"], self.array_std_pdcsap[self.index_pdcsap]
        ))
        #
        self.index_sap = np.argmin(self.array_std_sap[2 * len_one_filter:]) + len_one_filter
        self.index_pdcsap = np.argmin(self.array_std_pdcsap[2 * len_one_filter:]) + len_one_filter
        key_sap = key_list[self.index_sap]
        key_pdcsap = key_list[self.index_pdcsap]
        print("SAP: Sector {0}, best method={1}, window={2:.1f}, filtered std={3:.7f}".format(
            self.sector_number, self.filter_list[key_sap]["model"],
            self.filter_list[key_sap]["window"], self.array_std_sap[self.index_sap]
        ))
        print("PDCSAP: Sector {0}, best method={1}, window={2:.1f}, filtered std={3:.7f}".format(
            self.sector_number, self.filter_list[key_pdcsap]["model"],
            self.filter_list[key_pdcsap]["window"], self.array_std_pdcsap[self.index_pdcsap]
        ))
        #
        self.index_sap = np.argmin(self.array_std_sap[3 * len_one_filter:]) + len_one_filter
        self.index_pdcsap = np.argmin(self.array_std_pdcsap[3 * len_one_filter:]) + len_one_filter
        key_sap = key_list[self.index_sap]
        key_pdcsap = key_list[self.index_pdcsap]
        print("SAP: Sector {0}, best method={1}, window={2:.1f}, filtered std={3:.7f}".format(
            self.sector_number, self.filter_list[key_sap]["model"],
            self.filter_list[key_sap]["window"], self.array_std_sap[self.index_sap]
        ))
        print("PDCSAP: Sector {0}, best method={1}, window={2:.1f}, filtered std={3:.7f}".format(
            self.sector_number, self.filter_list[key_pdcsap]["model"],
            self.filter_list[key_pdcsap]["window"], self.array_std_pdcsap[self.index_pdcsap]
        ))
        #
        self.index = np.argmin(self.array_std_pdcsap)
        key_min = key_list[self.index]
        # TODO: average normalized error is computed using what?
        average_error = np.average(self.pdcsap_flux_error[~self.mask]/self.array_flat_pdcsap_model[self.index][~self.mask])
        print("Average normalized error wrt best method model: {0:.6f}".format(
            average_error))
        if average_error > self.array_std_pdcsap[self.index]:
            print("Average associated error of the observations higher than the standard deviation of the filtered light curve. Either there is overcorrection of the light curve, or the errors associated by TESS team are overestimated")
        # Comparison of the different fitering algorithm
        plt.figure(figsize=(12.8, 7.2))
        plt.scatter(self.time, self.pdcsap_flux, s=2)
        plt.errorbar(self.time, self.pdcsap_flux, yerr=self.pdcsap_flux_error,
                     ecolor='k', fmt=' ', alpha=0.25, zorder=-1)
        for i, model in enumerate(self.array_flat_pdcsap_model):
            key = key_list[i]
            if i == self.index:
                plt.plot(self.time, model, zorder=11,
                         label="BEST: {0} w:{1:.1f}".format(
                             self.filter_list[key]["model"],
                             self.filter_list[key]["window"]))
                continue
            plt.plot(self.time, model, zorder=10,
                     label="{0} w:{1:.1f}".format(
                         self.filter_list[key]["model"],
                         self.filter_list[key]["window"]))
        plt.xlabel("BJD_TDB")
        plt.ylabel("TESS PDCSAP flux [e-/s]")
        plt.title("Filtering algorithm comparison")
        # Place the legend outside of the plot:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2, fontsize='small')
        # Adjust the subplot parameters so the legend fits within the figure:
        plt.subplots_adjust(right=0.75)
        plt.savefig(str(Path(self.results_folder, "sector{0}_filtering_algorithm-comparison.png".format(
            self.sector_number))), dpi=150)  # you can adjust dpi as needed
        plt.show()
        plt.close()
        # Best filtering algorithm result near the transit
        plt.figure(figsize=(12.8, 7.2))
        plt.scatter(self.phase_folded_time, self.array_flat_pdcsap_flux[self.index],
                    label="BEST: {0} w:{1:.1f}".format(
                        self.filter_list[key_min]["model"], self.filter_list[key_min]["window"]),
                    c='C0', s=3)
        plt.errorbar(self.phase_folded_time, self.array_flat_pdcsap_flux[self.index],
                     yerr=self.pdcsap_flux_error/self.array_flat_pdcsap_model[self.index],
                     ecolor='k', fmt=' ', alpha=0.5, zorder=-1)
        plt.axhline(1.000, c='C1', label='y=1 line', ls="--")
        plt.xlabel('Time from mid-transit [d]')
        plt.ylabel('Normalized flux')
        plt.xlim(-0.4, 0.4)
        plt.legend()
        plt.savefig(str(Path(self.results_folder, "sector{0}_filtering_best-algorithm-zoom.png".format(
            self.sector_number))))
        plt.show()
        plt.close()

    def execute_flatten(self):
        """
        Execute the class functions
        """
        # self.testing_filter()
        self.plot_excluded_points()
        self.test_filter_algorithm()
        self.best_algorithm_decider()
        return self.array_flat_pdcsap_flux[self.index], self.pdcsap_flux_error[self.index]/self.array_flat_pdcsap_model[self.index]
