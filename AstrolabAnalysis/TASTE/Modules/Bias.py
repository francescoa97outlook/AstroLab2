from pathlib import Path
import matplotlib.pyplot as plt
import pickle
import numpy as np

from AstrolabAnalysis.TASTE.Modules.general_functions import read_fits, get_header_info


class Bias:
    """
    Class to analyze bias
    """
    def __init__(self, result_folder, bias_folder, bias_list, number_of_bias, yaml_bias):
        """
        Args:
            result_folder (str): Path to the folder where analysis results will be saved.
            bias_folder (str): Path to the folder containing bias frames.
            bias_list (list): List of file paths for bias frames.
            number_of_bias (int): Number of bias frames to process.
            yaml_bias (dict): Configuration parameters related to bias analysis from the YAML file.
        """
        print("----------------------- BIAS -----------------------")
        self.result_folder = result_folder
        self.bias_folder = bias_folder
        self.bias_list = bias_list
        self.number_of_bias = number_of_bias
        self.yaml_bias = yaml_bias
        self.stack_bias = None
        self.JD_first_BIAS = None
        self.AIRMASS_first_BIAS = None
        self.GAIN_first_BIAS = None
        self.GAIN_comments_first_BIAS = None
        self.RDNOISE_first_BIAS = None
        self.RDNOISE_comments_first_BIAS = None
        self.ccd_y_axis = None
        self.ccd_x_axis = None
        self.EXPTIME_first_bias = None
        self.SITELAT_first_bias = None
        self.SITELONG_first_bias = None
        self.OBJCTRA_first_bias = None
        self.OBJCTDEC_first_bias = None
        self.median_bias = None
        self.median_pixel_error = None

    def get_first_bias(self):
        """
        Workflow:
            1. Reads the FITS file of the first bias frame using its path.
            2. Extracts the header, header comments, and data from the FITS file.
            3. Processes the header information to obtain key details related to the first bias frame.
        """
        # Get first_bias_information
        (
            first_bias_header,
            first_bias_header_comments,
            first_bias_data
        ) = read_fits(str(Path(self.bias_folder, self.bias_list[0])))
        # get header info of first bias
        (
            self.JD_first_BIAS,
            self.AIRMASS_first_BIAS,
            self.GAIN_first_BIAS,
            self.GAIN_comments_first_BIAS,
            self.RDNOISE_first_BIAS,
            self.RDNOISE_comments_first_BIAS,
            self.ccd_x_axis,
            self.ccd_y_axis,
            self.EXPTIME_first_bias,
            self.SITELAT_first_bias,
            self.SITELONG_first_bias,
            self.OBJCTRA_first_bias,
            self.OBJCTDEC_first_bias,
        ) = (
            get_header_info(first_bias_header, first_bias_header_comments, "bias")
        )

    def stack_bias_function(self):
        """
        Workflow:
            1. Initializes an empty array to hold stacked bias frames with dimensions based on the number of bias frames and CCD dimensions.
            2. Iterates through the list of bias files:
                a. Reads the FITS file for each bias frame.
                b. Extracts header information, including the gain value.
                c. Multiplies the bias frame data by the gain value to scale it appropriately.
            3. Stores the scaled bias frames in the initialized array.
        """
        self.stack_bias = np.empty([self.number_of_bias, self.ccd_y_axis, self.ccd_x_axis])
        for i, bias in enumerate(self.bias_list):
            current_bias_header, current_bias_header_comments, current_bias_data = read_fits(
                str(Path(self.bias_folder, self.bias_list[i])))
            (
                _, _, gain_current, _, _, _, _, _, _, _, _, _, _
            ) = (
                get_header_info(current_bias_header, current_bias_header_comments, None)
            )
            # BIAS MULTIPLIED BY THE GAIN
            self.stack_bias[i, :, :] = current_bias_data * gain_current

    def median_bias_function(self):
        """
        Workflow:
            1. Computes the median bias frame across all stacked bias frames.
            2. Calculates the average along the columns of the median bias frame.
            3. Visualizes and compares the first bias frame (scaled by gain) with the median bias frame.
            4. Analyzes noise characteristics by comparing expected and measured noise in selected columns.
            5. Computes the pixel-based error and visualizes the distribution of errors.
            6. Saves median bias data, error values, and related metadata to files.
        """
        # Calculate the median along all the biases
        self.median_bias = np.median(self.stack_bias, axis=0)
        # Average along a column
        median_column = np.average(self.median_bias, axis=0)
        # PLOT the comparison between the first bias multiplied by the gain
        # and the median along all the biases multiplied by the gain (made before)
        fig, ax = plt.subplots(3, 1, sharex=True, figsize=(10, 8), dpi=300)
        fig.suptitle("Biases analysis")
        im1 = ax[0].imshow(
            self.stack_bias[0], vmin=np.min(self.median_bias),
            vmax=np.max(self.median_bias), origin='lower', cmap="inferno"
        )
        cbar = fig.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("e")
        #
        _ = ax[1].imshow(
            self.median_bias, vmin=np.min(self.median_bias), vmax=np.max(self.median_bias),
            origin='lower', cmap="inferno"
        )
        #
        ax[2].plot(median_column)
        #
        ax[0].set_ylabel('Y [pixels]')
        ax[0].set_title("First bias multiplied by gain")
        ax[1].set_ylabel('Y [pixels]')
        ax[1].set_title("Median along all the biases multiplied by the gain")
        ax[2].set_ylabel('Average counts [e]')
        ax[2].set_title("Average counts for each colum")
        ax[2].set_xlabel('X [pixels]')
        plt.savefig(str(Path(self.result_folder, "1_bias", "median_bias.png")))
        plt.show()
        plt.close(fig)
        #
        # Knowing that the bias is not constant, we can still compare the readout noise as reported in the
        # header of the fits frame and the standard deviation directly computed of the data if we restrict
        # ourself to a small range in columns.
        starting_column = int(self.yaml_bias["start_column"])
        ending_column = int(self.yaml_bias["end_column"])
        std_single_frame = np.std(self.stack_bias[0, :, starting_column:ending_column])
        print('Readout noise: {0:4.2f} e'.format(self.RDNOISE_first_BIAS))
        print('Std single frame: {0:4.2f} e'.format(std_single_frame))
        #
        expected_noise_medianbias = self.RDNOISE_first_BIAS / np.sqrt(self.number_of_bias)
        print('1) Expected noise of median bias : {0:4.2f} e'.format(expected_noise_medianbias))
        expected_std_medianbias = np.std(
            self.stack_bias[0, :, starting_column:ending_column]
        ) / np.sqrt(self.number_of_bias)
        print('2) Expected std of median bias: {0:4.2f} e'.format(expected_std_medianbias))
        #
        measured_std_medianbias = np.std(self.median_bias[:, starting_column:ending_column])
        print('Measured std of median bias: {0:4.2f} e'.format(measured_std_medianbias))
        median_error = np.std(self.stack_bias, axis=0) / np.sqrt(self.number_of_bias)
        self.median_pixel_error = np.median(median_error)
        print('Median std of each pixel: {0:4.2f} e'.format(self.median_pixel_error))
        #
        STD_pixel = np.std(self.stack_bias, axis=0)
        print("Standard deviation of each pixel bias:", STD_pixel)
        fig, ax = plt.subplots(1, 1, figsize=(10, 8), dpi=300)
        ax.hist(median_error.flatten(), bins=20, range=(0, 2.5), density=True, histtype='step',
                 label='Pixel-based error')
        ax.axvline(expected_noise_medianbias, c='C1', label='Error using readoutnoise')
        ax.axvline(expected_std_medianbias, c='C2', label='Expected error using bias std')
        ax.axvline(measured_std_medianbias, c='C3', label='Measured std of median bias')
        ax.axvline(self.median_pixel_error, c='C4', label='Average Pixel-based error')
        ax.set_xlabel('e')
        ax.set_ylabel('Density')
        ax.legend()
        plt.savefig(str(Path(self.result_folder, "1_bias", "medians_bias_info.png")))
        plt.show()
        plt.close(fig)
        # save files
        # create a dictionary
        with open(str(Path(self.result_folder, "1_bias", "bias_info.pkl")), "wb") as fo:
            pickle.dump([
                "self.median_bias", "self.median_bias_error", "self.median_bias_error_value",
                "stack_bias", "RDNOISE_first_BIAS",
                "measured_std_medianbias", "GAIN_first_BIAS"
            ], fo)
            pickle.dump(self.median_bias, fo)
            pickle.dump(median_error, fo)
            pickle.dump(self.median_pixel_error, fo)
            pickle.dump(self.stack_bias, fo)
            pickle.dump(measured_std_medianbias, fo)
            pickle.dump(self.RDNOISE_first_BIAS, fo)
            pickle.dump(self.GAIN_first_BIAS, fo)
    
    def execute_bias(self):
        """
        Workflow:
            1. Executes the bias analysis pipeline in the following order:
                a. Retrieves and processes the first bias frame.
                b. Stacks all bias frames and scales them by their respective gains.
                c. Computes the median bias frame and performs error analysis.
            2. Returns the median pixel error and the computed median bias frame.
        """
        self.get_first_bias()
        self.stack_bias_function()
        self.median_bias_function()
        return self.median_pixel_error, self.median_bias

