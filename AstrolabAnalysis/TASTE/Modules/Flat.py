import pickle
import matplotlib.pyplot as plt
from pathlib import Path

from AstrolabAnalysis.TASTE.Modules.general_functions import *


class Flat:
    """

    """
    def __init__(self, result_folder, flat_folder, flat_list, number_of_flat, median_bias, median_pixel_error, yaml_flat):
        """

        """
        print("----------------------- FLAT -----------------------")
        self.result_folder = result_folder
        self.flat_folder = flat_folder
        self.flat_list = flat_list
        self.number_of_flat = number_of_flat
        self.median_bias = median_bias
        self.median_pixel_error = median_pixel_error
        self.yaml_flat = yaml_flat
        self.stack_flat = None
        self.JD_first_flat = None
        self.AIRMASS_first_flat = None
        self.GAIN_first_flat = None
        self.GAIN_comments_first_flat = None
        self.rdnoise_stack_first_flat = None
        self.rdnoise_stack_comments_first_flat = None
        self.ccd_y_axis = None
        self.ccd_x_axis = None
        self.EXPTIME_first_flat = None
        self.SITELAT_first_flat = None
        self.SITELONG_first_flat = None
        self.OBJCTRA_first_flat = None
        self.OBJCTDEC_first_flat = None
        self.first_flat_data = None
        self.skip_overscan = None
        self.normalization_factors = None
        self.stack_normalized_flat = None
        self.median_normalized_flat = None
        self.median_normalized_flat_errors = None
        self.rdnoise_stack = None
        
    def get_first_flat(self):
        """

        """
        # Get first_flat_information
        (
            first_flat_header,
            first_flat_header_comments,
            self.first_flat_data
        ) = read_fits(str(Path(self.flat_folder, self.flat_list[0])))
        # get header info of first flat
        (
            self.JD_first_flat,
            self.AIRMASS_first_flat,
            self.GAIN_first_flat,
            self.GAIN_comments_first_flat,
            self.rdnoise_stack_first_flat,
            self.rdnoise_stack_comments_first_flat,
            self.ccd_x_axis,
            self.ccd_y_axis,
            self.EXPTIME_first_flat,
            self.SITELAT_first_flat,
            self.SITELONG_first_flat,
            self.OBJCTRA_first_flat,
            self.OBJCTDEC_first_flat,
        ) = (
            get_header_info(first_flat_header, first_flat_header_comments, "flat")
        )
        fig, ax = plt.subplots(2, 1, figsize=(10, 8), dpi=300)
        im1 = ax[0].imshow(self.first_flat_data, origin='lower', cmap="inferno")
        median_column = np.average(self.first_flat_data, axis=0)
        _ = ax[1].plot(median_column)
        cbar = fig.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("e")
        ax[0].set_xlabel('X [pixels]')
        ax[0].set_ylabel('Y [pixels]')
        ax[1].set_xlabel('X [pixels]')
        ax[1].set_ylabel('Y [pixels]')
        fig.suptitle("Plot of first flat with overscan included")
        plt.savefig(str(Path(self.result_folder, "2_flat", "first_flat_no_overscan.png")))
        plt.show()
        plt.close(fig)
        
    def define_overscan(self):
        mean_columns = np.mean(self.first_flat_data, axis=0)
        # Define the overscan columns by looking for the columns that have
        index_overscan = mean_columns < np.max(mean_columns) / 2
        self.skip_overscan = int(np.ceil(np.sum(index_overscan) / 2))
        vmin = np.min(self.first_flat_data[:, self.skip_overscan:-self.skip_overscan])
        vmax = np.max(self.first_flat_data[:, self.skip_overscan:-self.skip_overscan])
        print(
            "Vmin and vmax after overscan exclusion: [" +
            str(vmin) + ", " + str(vmax) + "]"
        )
        # Plot again with limits
        fig, ax = plt.subplots(2, 1, figsize=(10, 8), dpi=300)
        im1 = ax[0].imshow(
            self.first_flat_data, origin='lower', cmap="inferno",
            vmin=vmin, vmax=vmax
        )
        median_column = np.average(self.first_flat_data, axis=0)
        _ = ax[1].plot(median_column)
        ax[1].set_ylim(vmin, vmax)
        cbar = fig.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("e")
        ax[0].set_xlabel('X [pixels]')
        ax[0].set_ylabel('Y [pixels]')
        ax[1].set_xlabel('X [pixels]')
        ax[1].set_ylabel('Average counts [e]')
        fig.suptitle("Plot of first flat with overscan excluded")
        plt.savefig(str(Path(self.result_folder, "2_flat", "first_flat_overscan.png")))
        plt.show()
        plt.close(fig)
        
    def stack_flat_func(self):
        # Computing the normalization factors
        self.stack_flat = np.empty([self.number_of_flat, self.ccd_y_axis, self.ccd_x_axis])
        self.rdnoise_stack = np.empty(self.number_of_flat)
        for i, flat in enumerate(self.flat_list):
            current_flat_header, current_flat_header_comments, current_flat_data = read_fits(
                str(Path(self.flat_folder, flat)))
            (
                _, _, current_GAIN, _, self.rdnoise_stack[i], _, _, _, _, _, _, _, _
            ) = get_header_info(current_flat_header, current_flat_header_comments, None)
            self.stack_flat[i, :, :] = current_flat_data * current_GAIN - self.median_bias
        
    def normalization_factor(self, ):
        windows_size = int(self.yaml_flat["windows_size"])
        # x0, x1, y0, y1 represents the coordinates of the four corners
        x0 = np.int16(self.ccd_x_axis / 2 - windows_size / 2)
        x1 = np.int16(self.ccd_x_axis / 2 + windows_size / 2)
        y0 = np.int16(self.ccd_y_axis / 2 - windows_size / 2)
        y1 = np.int16(self.ccd_y_axis / 2 + windows_size / 2)
        print('Coordinates of the box: x0:{0}, x1:{1}, y0:{2}, y1:{3}'.format(x0, x1, y0, y1))
        # 
        self.normalization_factors = np.median(self.stack_flat[:, y0:y1, x0:x1], axis=(1, 2))
        print('Number of normalization factors (must be the same as the number of frames): {0}'.format(
            np.shape(self.normalization_factors)
        ))
        print("Normalization factors: ", self.normalization_factors)
        #
        normalization_factors_std = np.std(self.stack_flat[:, y0:y1, x0:x1], axis=(1, 2)) / np.sqrt(windows_size ** 2)
        print(normalization_factors_std)
        fig, ax = plt.subplots(1, 1, figsize=(10, 8), dpi=300)
        x_frame = np.arange(0, self.number_of_flat, 1)
        ax.scatter(x_frame, self.normalization_factors)
        ax.errorbar(x_frame, self.normalization_factors, normalization_factors_std, fmt='o', ms=2)
        ax.set_xlabel('Frame number')
        ax.set_ylabel('Average counts [e]')
        plt.savefig(str(Path(self.result_folder, "2_flat", "normalization_factors.png")))
        plt.show()
        plt.close(fig)
        
    def stack_flat_nomralized_func(self):
        #
        stack_normalized_iter = np.zeros_like(self.stack_flat)  # initialization of the output array
        for i_flat in range(self.number_of_flat):
            stack_normalized_iter[i_flat, :, :] = self.stack_flat[i_flat, :, :] / self.normalization_factors[i_flat]
        print("Shape of stack array           : ", np.shape(self.stack_flat))
        print("Shape of transposed stack array: ", np.shape(self.stack_flat.T))
        self.stack_normalized_flat = (self.stack_flat.T / self.normalization_factors).T
        print("Shape of normalized stack array: ", np.shape(self.stack_normalized_flat))
        # We can verify if the two methods deliver the same results by checking the maximum deviations between the two arrays in absolute terms.
        print("Maximum absolute difference between the two arrays: {0:2.6e}".format(
            np.max(abs(stack_normalized_iter - self.stack_normalized_flat)))
        )
        
    def median_flat(self):
        # Median flat
        self.median_normalized_flat = np.median(self.stack_normalized_flat, axis=0)
        #
        nmin = np.amin(self.median_normalized_flat[:, self.skip_overscan:-self.skip_overscan])
        nmax = np.amax(self.median_normalized_flat[:, self.skip_overscan:-self.skip_overscan])
        #
        fig, ax = plt.subplots(2, 1, figsize=(10, 8), dpi=300)
        im1 = ax[0].imshow(self.median_normalized_flat, origin='lower', vmin=nmin, vmax=nmax, cmap="inferno")
        median_column = np.average(self.median_normalized_flat, axis=0)
        _ = ax[1].plot(median_column)
        # we set the plot limits
        ax[1].set_ylim(nmin, nmax)
        # add the colorbar using the figure's method,
        # telling which mappable we're talking about and
        # which axes object it should be near
        cbar = fig.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("e")
        ax[0].set_xlabel('X [pixels]')
        ax[0].set_ylabel('Y [pixels]')
        ax[1].set_xlabel('X [pixels]')
        ax[1].set_ylabel('Average value [normalized]')
        plt.savefig(str(Path(self.result_folder, "2_flat", "median_flat.png")))
        plt.show()
        plt.close(fig)
        #
        photon_noise = np.sqrt(abs(self.stack_flat))
        stack_error = np.sqrt(np.mean(self.rdnoise_stack) ** 2 + self.median_pixel_error ** 2 + photon_noise ** 2)
        #
        stack_normalized_error = (stack_error.T / self.normalization_factors).T
        # 
        self.median_normalized_flat_errors = np.sum(stack_normalized_error ** 2, axis=0) / self.number_of_flat
        print("Shape of the Median normalized error array: ", np.shape(self.median_normalized_flat_errors))
        
    def save_pkl(self):
        list_names = ["median_normalized_flat", "stack_normalized", "normalization_factors", "stack_flat",
                      "median_normalized_flat_errors"]
        with open(str(Path(self.result_folder, "2_flat", "flat_info.pkl")), "wb") as fo:
            pickle.dump(list_names, fo)
            pickle.dump(self.median_normalized_flat, fo)
            pickle.dump(self.stack_normalized_flat, fo)
            pickle.dump(self.normalization_factors, fo)
            pickle.dump(self.stack_flat, fo)
            pickle.dump(self.median_normalized_flat_errors, fo)

    def some_statistic_flat(self):
        mean_normalization = np.mean(self.normalization_factors)
        fig, ax = plt.subplots(1, 1, figsize=(10, 8), dpi=300)
        limit_x_1 = int(self.yaml_flat["limit_x_1"])
        limit_x_2 = int(self.yaml_flat["limit_x_2"])
        limit_y_1 = int(self.yaml_flat["limit_y_1"])
        limit_y_2 = int(self.yaml_flat["limit_y_2"])
        bins = int(self.yaml_flat["bins"])
        ax.hist(self.stack_flat[:, limit_x_1:limit_x_2, limit_y_1:limit_y_2].flatten(),
                bins=bins, alpha=0.5, label='Before norm.')
        ax.hist(self.stack_normalized_flat[:, limit_x_1:limit_x_2, limit_y_1:limit_y_2].flatten() * mean_normalization,
                bins=bins, alpha=0.5, label='After norm.')
        ax.set_xlabel('Counts [e]')
        ax.set_ylabel('#')
        ax.legend()
        plt.savefig(str(Path(self.result_folder, "2_flat", "statistic_1.png")))
        plt.show()
        plt.close(fig)
        #
        sigma_mean_normalization = np.sqrt(mean_normalization)
        x = np.arange(np.amin(self.normalization_factors), np.amax(self.normalization_factors), 10)
        y = (1. / (
                sigma_mean_normalization * np.sqrt(2 * np.pi)) *
             np.exp(- (x - mean_normalization) ** 2 / (2 * sigma_mean_normalization ** 2))
             )
        fig, ax = plt.subplots(1, 1, figsize=(10, 8), dpi=300)
        ax.hist(self.normalization_factors, alpha=0.5, density=True, label='Normalization factors')
        ax.plot(x, y)
        ax.set_xlabel('Counts [e]')
        ax.set_ylabel('Probability density')
        ax.legend()
        plt.savefig(str(Path(self.result_folder, "2_flat", "statistic_2.png")))
        plt.show()
        plt.close(fig)

    def execute_flat(self):
        self.get_first_flat()
        self.define_overscan()
        self.stack_flat_func()
        self.normalization_factor()
        self.stack_flat_nomralized_func()
        self.median_flat()
        self.save_pkl()
        self.some_statistic_flat()
        return self.median_normalized_flat, self.median_normalized_flat_errors, self.skip_overscan
