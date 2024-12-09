from pathlib import Path

import skimage
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from matplotlib import colors
import cv2
from matplotlib.colors import ListedColormap

from AstrolabAnalysis.TASTE.Modules.Star import Star
from AstrolabAnalysis.TASTE.Modules.general_functions import make_circle_around_star2


class FirstCentroidPosition:
    
    def __init__(
            self, science_corrected, skip_border, multipler_inner_radius, science_list_str_f,
            result_folder, yaml_centroids
    ):
        print("----------------------- CENTROID -----------------------")
        self.size_arr = len(science_corrected[int(yaml_centroids["img_number"]):, 0, 0])
        science_corrected = science_corrected[int(yaml_centroids["img_number"]), :, :]
        science_corrected[:, :skip_border] = 0
        science_corrected[:, -skip_border:] = 0
        self.science_corrected = science_corrected
        self.skip_border = skip_border
        self.multipler_inner_radius = multipler_inner_radius
        self.science_list_str_f = science_list_str_f
        self.result_folder = result_folder
        self.yaml_centroids = yaml_centroids
        self.vmin = None
        self.y_target_refined = None
        self.x_target_refined = None
        self.ref_star2 = None
        self.ref_star1 = None
        self.tar_star = None
        self.y_axis = None
        self.x_axis = None
        self.y_mesh = None
        self.x_mesh = None
        self.ydim = None
        self.xdim = None
        
    def plot_first_scientific(self):
        fig, ax = plt.subplots(1, 1, figsize=(10, 8), dpi=300)
        im1 = ax.imshow(self.science_corrected, cmap=plt.colormaps['inferno'], origin='lower')
        plt.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xlabel('self.x_mesh [pixels]')
        ax.set_ylabel('self.y_mesh [pixels]')
        ax.set_title("First scientific image without overscan")
        plt.savefig(str(Path(self.result_folder, "4_centroid", "no_overscan_" + str(int(self.yaml_centroids["img_number"])) + ".png")))
        plt.show()
        plt.close(fig)
        self.xdim, self.ydim = np.shape(self.science_corrected)
        print('Shape of our science frame: {0:d} x {1:d}'.format(self.ydim, self.xdim))
        self.x_axis = np.arange(0, self.ydim, 1)
        self.y_axis = np.arange(0, self.xdim, 1)
        self.x_mesh, self.y_mesh = np.meshgrid(self.x_axis, self.y_axis)

    def normalize_and_binarize(self):
        labels = [
            "Star0", "Star1", "Star2", "Star3", "Star4", "Star5", "Star6", "Star7", "Star8",
            "Star9", "Star10", "Star11", "Star12", "Star13", "Star14"
        ]
        cmap = ListedColormap(['black', 'yellow'])
        normalized_image = np.array((self.science_corrected - np.min(self.science_corrected)) / np.max(self.science_corrected) * 255, dtype=np.uint8)
        normalized_image = normalized_image[:, self.skip_border:-self.skip_border]
        limits = np.linspace(0, len(normalized_image[0, :]), int(self.yaml_centroids["division"]), dtype=int)
        binary_image = np.zeros_like(normalized_image)
        centers = list()
        radius = list()
        vmax = list()
        for i in range(1, limits.shape[0]):
            _, binary_image[:, limits[i - 1]:limits[i]] = cv2.threshold(
                normalized_image[:, limits[i - 1]:limits[i]],
                0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            #
            blobs = skimage.measure.label(binary_image[:, limits[i - 1]:limits[i]] > 0)
            stats = skimage.measure.regionprops(blobs)
            for s in stats:
                if self.yaml_centroids["area_min"] < s.area < self.yaml_centroids["area_max"]:
                    centers.append([s.centroid[0], s.centroid[1] + limits[i - 1]])
                    radius.append(np.sqrt(s.area / np.pi))
                    vmax.append(np.max(self.science_corrected[:, limits[i - 1]:limits[i]]))
        # Display the results
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), dpi=300)
        ax1.set_title('Original Matrix')
        ax1.imshow(normalized_image, cmap='inferno', origin="lower")
        ax2.set_title(f'Binarized Matrix')
        ax2.imshow(binary_image, cmap=cmap, origin="lower")
        for lim in limits[1:-1]:
            ax1.axvline(x=lim, color='red', linestyle='--')
            ax2.axvline(x=lim, color='red', linestyle='--')
        colorss = [
            "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10", "C11",
            "C12", "C13", "C14", "C15", "C16"
        ]
        for i in range(len(radius)):
            make_circle_around_star2(
                centers[i][1], centers[i][0], radius[i], thickness=3,
                axf=ax1, label=labels[i], color=colorss[i]
            )
            make_circle_around_star2(
                centers[i][1], centers[i][0], radius[i], thickness=3,
                axf=ax2, label=labels[i], color=colorss[i]
            )
            centers[i][1] += self.skip_border
        ax1.legend()
        ax2.legend()
        plt.savefig(str(Path(self.result_folder, "4_centroid", "centroids_" + str(int(self.yaml_centroids["img_number"])) + ".png")))
        plt.show()
        plt.close(fig)
        #
        if self.yaml_centroids["order"]:
            index_tar = int(self.yaml_centroids["index_tar"])
            index_ref1 = int(self.yaml_centroids["index_ref1"])
            index_ref2 = int(self.yaml_centroids["index_ref2"])
        else:
            print('Enter target star index (0,1,2,3,..)')
            index_tar = int(input())
            print("Enter first reference star index (0,1,2,3,..)")
            index_ref1 = int(input())
            print("Enter second reference star index (0,1,2,3,..)")
            index_ref2 = int(input())
        self.tar_star = Star(
            self.x_mesh, self.y_mesh, self.x_axis, int(centers[index_tar][1]), int(centers[index_tar][0]),
            int(radius[index_tar]), vmax[index_tar], "Target", self.multipler_inner_radius
        )
        self.ref_star1 = Star(
            self.x_mesh, self.y_mesh, self.x_axis, int(centers[index_ref1][1]), int(centers[index_ref1][0]),
            int(radius[index_ref1]), vmax[index_ref1], "Reference1", self.multipler_inner_radius
        )
        self.ref_star2 = Star(
            self.x_mesh, self.y_mesh, self.x_axis, int(centers[index_ref2][1]), int(centers[index_ref2][0]),
            int(radius[index_ref2]), vmax[index_ref2], "Reference2", self.multipler_inner_radius
        )
    
    def plot_3d(self, title, vmin=None):
        vmax = None
        radius_plot = self.yaml_centroids["radius_plot_3d"]
        if vmin is not None:
            vmax = self.tar_star.vmax
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(10, 8), dpi=300)
        # Plot the surface.
        surf = ax.plot_surface(
            self.x_mesh[
                self.tar_star.y_initial - radius_plot:self.tar_star.y_initial + radius_plot,
                self.tar_star.x_initial - radius_plot:self.tar_star.x_initial + radius_plot
            ],
            self.y_mesh[
                self.tar_star.y_initial - radius_plot:self.tar_star.y_initial + radius_plot,
                self.tar_star.x_initial - radius_plot:self.tar_star.x_initial + radius_plot
            ],
            self.science_corrected[
                self.tar_star.y_initial - radius_plot:self.tar_star.y_initial + radius_plot,
                self.tar_star.x_initial - radius_plot:self.tar_star.x_initial + radius_plot
            ], cmap=plt.colormaps['inferno'], norm=colors.LogNorm(vmin=vmin, vmax=vmax),
            linewidth=0, antialiased=False
        )
        # 
        ax.azim = int(self.yaml_centroids["azim_3d"] ) # value in degree
        ax.elev = int(self.yaml_centroids["elev_3d"])  # value in degre
        ax.set_xlabel('X [pixels]')
        ax.set_ylabel('Y [pixels]')
        # Fix the orientation fo the Z label
        ax.zaxis.set_rotate_label(False)  # disable automatic rotation
        ax.set_zlabel('photoelectrons', rotation=90)
        fig.colorbar(surf, shrink=0.5, aspect=15, ticks=[10, 100, 1000, 10000, 100000])
        plt.title(title)
        plt.savefig(str(Path(self.result_folder, "4_centroid", "surf_" + str(vmax) + "_" + str(int(self.yaml_centroids["img_number"])) + ".png")))
        plt.show()
        plt.close(fig)

    def photocenter_star(self):
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(10, 8), dpi=300)
        # Plot the surface.
        surf = ax.plot_surface(
            self.x_mesh, self.y_mesh, self.science_corrected, cmap=plt.colormaps['inferno'],
            norm=colors.LogNorm(),
            linewidth=0, antialiased=False
        )
        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)
        ax.set_title("Plot with overscan and no vmax limitations")
        plt.savefig(str(Path(self.result_folder, "4_centroid", "photocenter_" + str(int(self.yaml_centroids["img_number"])) + ".png")))
        plt.show()
        plt.close(fig)
        #
        self.plot_3d("Plot without vmin and vmax limitations")
        #
        vmin = np.amin(self.science_corrected[:, 100:400])
        self.plot_3d(f"Plot limitated between {vmin} and {self.tar_star.vmax}", vmin=vmin)

    def stars_operations(self):
        self.tar_star.find_first_refined_center(self.science_corrected)
        self.ref_star1.find_first_refined_center(self.science_corrected)
        self.ref_star2.find_first_refined_center(self.science_corrected)
        #
        self.vmin = np.amin(self.science_corrected[:, self.skip_border:-self.skip_border])
        self.tar_star.example_bad_good_inner_radius(self.science_corrected, self.vmin)
        self.ref_star1.example_bad_good_inner_radius(self.science_corrected, self.vmin)
        self.ref_star2.example_bad_good_inner_radius(self.science_corrected, self.vmin)
        #
        maximum_number_of_iterations = int(self.yaml_centroids["maximum_number_of_iterations"])
        self.tar_star.calc_refined_center(maximum_number_of_iterations, self.science_corrected)
        self.ref_star1.calc_refined_center(maximum_number_of_iterations, self.science_corrected)
        self.ref_star2.calc_refined_center(maximum_number_of_iterations, self.science_corrected)

    def plot_target_refined(self):
        print('Refined target coordinates  x: {0:5.2f}   y: {1:5.2f}'.format(
            self.tar_star.x_refined, self.tar_star.y_refined
        ))
        print('vmin:  {0:.1f}    vmax: {1:.1f}'.format(self.vmin, self.tar_star.vmax))
        fig, ax = plt.subplots(1, 1, figsize=(10, 8), dpi=300)
        im1 = ax.imshow(
            self.science_corrected, cmap=plt.colormaps['inferno'],
            norm=colors.LogNorm(vmin=self.vmin, vmax=self.tar_star.vmax),
            origin='lower'
        )
        plt.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)
        # Cut the plot around the target star, with some margin with respect to the inner radius
        ax.set_xlim(self.tar_star.x_refined - self.tar_star.inner_radius * 1.2, self.tar_star.x_refined + self.tar_star.inner_radius * 1.2)
        ax.set_ylim(self.tar_star.y_refined - self.tar_star.inner_radius * 1.2, self.tar_star.y_refined + self.tar_star.inner_radius * 1.2)
        make_circle_around_star2(self.tar_star.x_refined, self.tar_star.y_refined, self.tar_star.inner_radius, label='inner radius', axf=ax)
        ax.scatter(self.tar_star.x_initial, self.tar_star.y_initial, s=20, c='C2', label='Starting point')
        ax.scatter(self.tar_star.x_refined, self.tar_star.y_refined, s=10, c='k', label='Final point')
        ax.set_xlabel('X [pixels]')
        ax.set_ylabel('Y [pixels]')
        ax.legend(loc='upper left')
        plt.savefig(str(Path(self.result_folder, "4_centroid", "target_refined_" + str(int(self.yaml_centroids["img_number"])) + ".png")))
        plt.show()
        plt.close(fig)
        #

    def plot_fwhm_target(self):
        fwhm_x, fwhm_y, cumulative_sum_x, cumulative_sum_y = self.tar_star.determine_FWHM_axis(
            self.science_corrected, int(self.yaml_centroids["img_number"])
        )
        fig, ax = plt.subplots(1, 1, figsize=(10, 8), dpi=300)
        ax.scatter(self.x_axis - self.tar_star.x_refined, cumulative_sum_x, label='NCD along the self.x_mesh axis')
        ax.scatter(self.y_axis - self.tar_star.y_refined, cumulative_sum_y, label='NCD along the self.y_mesh axis')
        ax.axvline(0, c='k')
        ax.set_xlim(-self.tar_star.inner_radius * 1.3, self.tar_star.inner_radius * 1.3)
        ax.axvline(self.tar_star.inner_radius, c='C5', label='Inner radius')
        ax.axvline(-self.tar_star.inner_radius, c='C5')
        ax.set_xlabel('Distance from the photocenter [pixels]')
        ax.set_ylabel('Normalized cumulative distribution [NCD]')
        ax.legend()
        plt.savefig(str(Path(self.result_folder, "4_centroid", "fwhm_" + str(int(self.yaml_centroids["img_number"])) + ".png")))
        plt.show()
        plt.close(fig)
        print('FWHM along the self.x_mesh axis: {0:.2f}'.format(fwhm_x))
        print('FWHM along the self.y_mesh axis: {0:.2f}'.format(fwhm_y))
        # From the fits header of the first image:
        # CCDSCALE=                 0.25 / [arcsec/px] unbinned CCD camera scale
        # BINX    =                    4 / Horizontal Binning factor used
        # BINY    =                    4 / Vertical Binning factor used
        hdul = fits.open(self.science_list_str_f)
        CCDSCALE = hdul[0].header['CCDSCALE']
        BINX = hdul[0].header['BINX']
        BINY = hdul[0].header['BINY']
        hdul.close()
        print('Seeing along the self.x_mesh axis (after defocusing): {0:.2f}'.format(
            fwhm_x * (BINX + BINY) / 2 * CCDSCALE))
        print('Seeing along the self.y_mesh axis (after defocusing): {0:.2f}'.format(
            fwhm_y * (BINX + BINY) / 2 * CCDSCALE))

    def execute_centroid(self):
        self.plot_first_scientific()
        self.normalize_and_binarize()
        self.photocenter_star()
        self.stars_operations()
        self.plot_target_refined()
        self.tar_star.assign_dimension_array(self.size_arr)
        self.ref_star1.assign_dimension_array(self.size_arr)
        self.ref_star2.assign_dimension_array(self.size_arr)
        self.plot_fwhm_target()
        return self.tar_star, self.ref_star1, self.ref_star2, self.vmin
