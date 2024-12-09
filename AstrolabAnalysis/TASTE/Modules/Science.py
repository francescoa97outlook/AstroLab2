import os
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
from astropy import coordinates as coord, units as u
from astropy.time import Time

from AstrolabAnalysis.TASTE.Modules.general_functions import *


class Science:
    """

    """
    def __init__(
        self, result_folder, science_folder, science_list, number_of_science,
        median_pixel_error, median_bias, median_normalized_flat, median_normalized_flat_errors,
        number_test=30, yaml_science=""
    ):
        """

        """
        print("----------------------- SCIENCE -----------------------")
        self.result_folder = result_folder
        self.science_folder = science_folder
        self.science_list = science_list
        self.number_of_science = number_of_science
        self.median_bias = median_bias
        self.median_pixel_error = median_pixel_error
        self.median_normalized_flat = median_normalized_flat
        self.median_normalized_flat_errors = median_normalized_flat_errors
        self.number_test = number_test
        self.yaml_science = yaml_science
        self.stack_science = None
        self.JD_first_science = None
        self.AIRMASS_first_science = None
        self.GAIN_first_science = None
        self.GAIN_comments_first_science = None
        self.rdnoise_stack_first_science = None
        self.rdnoise_stack_comments_first_science = None
        self.ccd_y_axis = None
        self.ccd_x_axis = None
        self.EXPTIME_first_science = None
        self.SITELAT_first_science = None
        self.SITELONG_first_science = None
        self.OBJCTRA_first_science = None
        self.OBJCTDEC_first_science = None
        self.first_science_data = None
        self.skip_overscan = None
        self.bjd_tdb = None
        self.science_corrected_errors_stack = None
        self.array_airmass = None
        self.array_exptime = None
        self.julian_date = None
        self.science_corrected_stack = None
        
    def get_first_science(self):
        """

        """
        # Get first_science_information
        (
            first_science_header,
            first_science_header_comments,
            self.first_science_data
        ) = read_fits(str(Path(self.science_folder, self.science_list[0])))
        # get header info of first science
        (
            self.JD_first_science,
            self.AIRMASS_first_science,
            self.GAIN_first_science,
            self.GAIN_comments_first_science,
            self.rdnoise_stack_first_science,
            self.rdnoise_stack_comments_first_science,
            self.ccd_x_axis,
            self.ccd_y_axis,
            self.EXPTIME_first_science,
            self.SITELAT_first_science,
            self.SITELONG_first_science,
            self.OBJCTRA_first_science,
            self.OBJCTDEC_first_science,
        ) = (
            get_header_info(first_science_header, first_science_header_comments, "science")
        )
        
    def science_stack_function(self):
        if int(self.yaml_science["read_pkl"]):
            pkl_file = str(Path(self.result_folder, "3_science", "stack_corr.pkl"))
            with open(pkl_file, 'rb') as f:
                self.science_corrected_stack = pickle.load(f)
                self.science_corrected_errors_stack = pickle.load(f)
                self.julian_date = pickle.load(f)
                self.array_exptime = pickle.load(f)
                self.array_airmass = pickle.load(f)
        else:
            self.stack_science = np.empty([self.number_test, self.ccd_y_axis, self.ccd_x_axis])
            self.science_corrected_stack = np.empty([self.number_test, self.ccd_y_axis, self.ccd_x_axis])
            self.julian_date = np.empty(self.number_test)
            self.science_corrected_errors_stack = np.empty([self.number_test, self.ccd_y_axis, self.ccd_x_axis])
            self.array_exptime = np.zeros(self.number_test)
            self.array_airmass = np.zeros(self.number_test)
            path_correct = str(Path(self.result_folder, "3_science"))
            os.system("mkdir -p " + path_correct)
            science_test_list = self.science_list[:self.number_test + 1]
            for i, science in enumerate(science_test_list):
                (current_science_header,
                 current_science_header_comments,
                 current_science_data) = read_fits(str(Path(self.science_folder, science)))
                (
                    self.julian_date[i], self.array_airmass[i], gain_current, _, _, _, _, _, self.array_exptime[i], _, _, _, _,
                ) = get_header_info(current_science_header, current_science_header_comments)
                self.stack_science[i, :, :] = current_science_data * gain_current
                science_debiased = self.stack_science[i, :, :] - self.median_bias
                self.science_corrected_stack[i, :, :] = science_debiased / self.median_normalized_flat
                # Error associated to the science corrected frame
                science_debiased_errors = np.sqrt(gain_current ** 2 + self.median_pixel_error ** 2 + science_debiased)
                self.science_corrected_errors_stack[i, :, :] = (
                        self.science_corrected_stack[i, :, :] *
                        np.sqrt(
                            (science_debiased_errors / science_debiased) ** 2 +
                            (self.median_normalized_flat_errors / self.median_normalized_flat) ** 2
                        )
                )
            with open(str(Path(path_correct, "stack_corr.pkl")), "wb") as fo:
                pickle.dump(self.science_corrected_stack, fo)
                pickle.dump(self.science_corrected_errors_stack, fo)
                pickle.dump(self.julian_date, fo)
                pickle.dump(self.array_exptime, fo)
                pickle.dump(self.array_airmass, fo)

    def plot_bjd_tdb(self):
        target = coord.SkyCoord(
            self.OBJCTRA_first_science, self.OBJCTDEC_first_science, unit=(u.hourangle, u.deg), frame='icrs'
        )
        # Let's make a plot of the light travel time correction along one year, for our target:
        # https://docs.astropy.org/en/stable/time/
        # let's compute the light travel time for one year of observations
        jd_plot = np.arange(
            float(self.yaml_science["jd_plot_1"]),
            float(self.yaml_science["jd_plot_2"]),
            float(self.yaml_science["step_jd"])
        )
        tm_plot = Time(jd_plot, format='jd', scale='utc', location=(self.SITELAT_first_science, self.SITELONG_first_science))  # location=('45.8472d', '11.569d'))
        ltt_plot = tm_plot.light_travel_time(target, ephemeris='jpl')
        # Convert to BJD_TDB, and then add the light travel time
        bjd_tdb_plot = tm_plot.tdb + ltt_plot
        print("bjd_tdb_plot", bjd_tdb_plot)
        fig, ax = plt.subplots(1, 1, figsize=(10, 8), dpi=300)
        ax.plot(jd_plot, ltt_plot.to_value(u.min))
        ax.set_xlabel('JD [d]')
        ax.set_ylabel('Light travel time [m]')
        plt.savefig(str(Path(self.result_folder, "3_science", "bjd_corr.png")))
        plt.show()
        plt.close(fig)
        #
        seconds_in_day = 86400
        jd = self.julian_date + self.array_exptime / seconds_in_day / 2.
        #
        tm = Time(jd, format='jd', scale='utc', location=(self.SITELAT_first_science, self.SITELONG_first_science))
        # Asiago - Cima Ekar
        # 45° 50' 50'' N -> 45.8472
        # 11° 34' 08'' E -> 11.569
        ltt_bary = tm.light_travel_time(target)
        self.bjd_tdb = tm.tdb + ltt_bary
        print(
            'Average Light travel time: {0:12.2f} minutes'.format(
                np.average(ltt_bary.to_value(u.min))
            )
        )
        print('Average difference between JD_UTC and BJD_TDB: {0:12.2f} seconds'.format(
            np.average(jd - self.bjd_tdb.to_value('jd')) * seconds_in_day
        ))

    def execute_science(self):
        self.get_first_science()
        self.science_stack_function()
        self.plot_bjd_tdb()
        return self.science_corrected_stack, self.bjd_tdb.to_value('jd'), self.array_airmass
