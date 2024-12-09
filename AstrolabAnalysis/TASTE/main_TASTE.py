import os

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

import yaml

from AstrolabAnalysis.TASTE.Modules.AperturePhotometry import AperturePhotometry
from AstrolabAnalysis.TASTE.Modules.Bias import Bias
from AstrolabAnalysis.TASTE.Modules.Centroid import FirstCentroidPosition
from AstrolabAnalysis.TASTE.Modules.Flat import Flat
from AstrolabAnalysis.TASTE.Modules.Science import Science

# ############
if __name__ == '__main__':
    # Path descriptions
    current_file = Path(__file__).parent
    with open(str(Path(current_file, "constants.yaml"))) as in_f:
        yaml_file = yaml.load(in_f, Loader=yaml.FullLoader)
    yaml_file_main = yaml_file["main"]
    #
    target = yaml_file_main["target"]
    base_folder = str(Path(current_file.parent.parent, target))
    result_folder = str(Path(base_folder, "results"))
    os.makedirs(result_folder, exist_ok=True)
    os.makedirs(str(Path(result_folder, "1_bias")), exist_ok=True)
    os.makedirs(str(Path(result_folder, "2_flat")), exist_ok=True)
    os.makedirs(str(Path(result_folder, "3_science")), exist_ok=True)
    os.makedirs(str(Path(result_folder, "4_centroid")), exist_ok=True)
    os.makedirs(str(Path(result_folder, "5_aperture")), exist_ok=True)
    # BIAS ANALYSIS
    bias_folder = str(Path(base_folder, "bias"))
    bias_list_str = np.genfromtxt(str(Path(bias_folder, "list_bias.txt")), dtype=str)
    number_of_BIAS = len(bias_list_str)
    median_pixel_error_bias, median_bias = Bias(
        result_folder, bias_folder, bias_list_str, number_of_BIAS, yaml_file["bias"]
    ).execute_bias()
    # FLAT ANALYSIS
    flat_folder = str(Path(base_folder, "flat"))
    flat_list_str = np.genfromtxt(str(Path(flat_folder, "list_flat.txt")), dtype=str)
    number_of_FLAT = len(flat_list_str)
    median_normalized_flat, median_normalized_flat_errors, skip_overscan = Flat(
        result_folder, flat_folder, flat_list_str, number_of_FLAT, median_bias, median_pixel_error_bias, yaml_file["flat"]
    ).execute_flat()
    # SCIENCE ANALYSIS
    multipler_inner_radius = float(yaml_file_main["multipler_inner_radius"])
    science_folder = str(Path(base_folder, "science"))
    science_list_str = np.genfromtxt(str(Path(science_folder, "list_science.txt")), dtype=str)
    number_of_SCIENCE = len(science_list_str)
    science_corrected_stack, bjd_julian_date, array_airmass = Science(
        result_folder, science_folder, science_list_str, number_of_SCIENCE, median_pixel_error_bias, median_bias,
        median_normalized_flat, median_normalized_flat_errors, number_of_SCIENCE, yaml_file["science"]
    ).execute_science()
    # CENTROID FIRST IMAGE
    tar_star, ref_star1, ref_star2, vmin =  FirstCentroidPosition(
        science_corrected_stack, skip_overscan, multipler_inner_radius,
        str(Path(science_folder, str(science_list_str[0]))), result_folder, yaml_file["centroid0"]
    ).execute_centroid()
    # CHECK BACKGROUND FOR SELECT SKIP IMAGE
    sky_background_medians = np.zeros(number_of_SCIENCE)
    sky_background_average = np.zeros(number_of_SCIENCE)
    for i, science in enumerate(science_corrected_stack):
        sky_background_average[i], sky_background_medians[i], _ = tar_star.sky_background(science)
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(range(number_of_SCIENCE), sky_background_medians, label="Median")
    ax.plot(range(number_of_SCIENCE), sky_background_average, label="Average")
    ax.set_xlabel("Science image")
    ax.set_ylabel("Sky background count")
    ax.legend()
    plt.show()
    # CENTROID first good IMAGE
    avoid_input = int(yaml_file_main["avoid_input"])
    if avoid_input:
        skip_to_image = int(yaml_file["centroidskip"]["img_number"])
    else:
        print('Enter the amount of images to skip')
        skip_to_image = int(input())
    tar_star, ref_star1, ref_star2, vmin = FirstCentroidPosition(
        science_corrected_stack, skip_overscan, multipler_inner_radius,
        str(Path(science_folder, str(science_list_str[skip_to_image]))), result_folder,
        yaml_file["centroidskip"]
    ).execute_centroid()
    # APERTURE ANALYSIS
    AperturePhotometry(
        science_corrected_stack[skip_to_image:, :, :], bjd_julian_date[skip_to_image:], array_airmass[skip_to_image:],
        skip_overscan, tar_star, ref_star1, ref_star2, vmin, avoid_input, result_folder, yaml_file["aperture"]
    ).execute_aperture()