import numpy as np
from matplotlib import pyplot as plt, colors

from AstrolabAnalysis.TASTE.Modules.general_functions import make_circle_around_star2


class Star:
    def __init__(self, X_mesh, Y_mesh, ref_axis, x, y, radius, vmax, name, multipler_inner_radius):
        self.percent_variance_y = None
        self.percent_variance_x = None
        self.name = name
        self.ref_axis = ref_axis
        self.X_mesh = X_mesh
        self.Y_mesh = Y_mesh
        self.x_initial = x
        self.y_initial = y
        self.x_refined = x
        self.y_refined = y
        self.radius = radius
        self.inner_radius = int(self.radius * multipler_inner_radius)
        self.outer_radius = int(self.inner_radius * 1.5)
        self.vmax = vmax
        self.fwhm_x = None
        self.fwhm_y = None
        self.aperture_star = None
        self.x_arr = None
        self.y_arr = None
        self.target_distance = None
        self.sky_background_arr = None

    def assign_dimension_array(self, size_arr):
        self.x_arr = np.zeros(size_arr)
        self.y_arr = np.zeros(size_arr)
        self.sky_background_arr = np.zeros(size_arr)
        self.aperture_star = np.zeros(size_arr)
        self.fwhm_x = np.zeros(size_arr)
        self.fwhm_y = np.zeros(size_arr)

    def example_bad_good_inner_radius(self, science_corrected, vmin):
        vmax = self.vmax
        print('vmin:  {0:.1f}    vmax: {1:.1f}'.format(vmin, vmax))
        fig, ax = plt.subplots(1, 1, figsize=(10, 8), dpi=300)
        im1 = plt.imshow(
            science_corrected, cmap=plt.colormaps['inferno'],
            norm=colors.LogNorm(vmin=vmin, vmax=vmax),
            origin='lower'
        )
        plt.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)
        # Cut the plot around the target star, with some margin with respect to the inner radius
        ax.set_xlim(
            self.x_refined - self.inner_radius * 1.2,
            self.x_refined + self.inner_radius * 1.2
        )
        ax.set_ylim(
            self.y_refined - self.inner_radius * 1.2,
            self.y_refined + self.inner_radius * 1.2
        )
        make_circle_around_star2(
            self.x_refined, self.y_refined, self.inner_radius,
            label='Good inner radius', axf=ax)
        make_circle_around_star2(
            self.x_refined, self.y_refined, self.radius, color='y',
            label='Bad inner radius', axf=ax)
        ax.set_xlabel('X [pixels]')
        ax.set_ylabel('Y [pixels]')
        ax.set_title(self.name)
        ax.legend(loc='upper left')
        plt.show()
        plt.close(fig)

    def find_first_refined_center(self, science_corrected):
        temp_x = self.x_refined
        temp_y = self.y_refined
        # 2D array with the distance of each pixel from the target star
        self.target_distance = np.sqrt((self.X_mesh - self.x_refined) ** 2 + (self.Y_mesh - self.y_refined) ** 2)
        # Selection of the pixels within the inner radius
        annulus_sel = (self.target_distance < self.inner_radius)
        # Weighted sum of coordinates
        weighted_X = np.sum(
            science_corrected[annulus_sel] * self.X_mesh[annulus_sel]
        )
        weighted_Y = np.sum(
            science_corrected[annulus_sel] * self.Y_mesh[annulus_sel]
        )
        # Sum of the weights
        total_flux = np.sum(science_corrected[annulus_sel])
        # Refined determination of coordinates
        self.x_refined = weighted_X / total_flux
        self.y_refined = weighted_Y / total_flux
        self.percent_variance_x = (self.x_refined - temp_x) / temp_x * 100.
        self.percent_variance_y = (self.x_refined - temp_y) / temp_y * 100.
    
    def calc_refined_center(self, maximum_number_of_iterations, science_corrected):
        for i_iter in range(0, maximum_number_of_iterations):
            self.find_first_refined_center(science_corrected)
            # exit condition: both percent variance are smaller than 0.1%
            if np.abs(self.percent_variance_x) < 0.1 and np.abs(self.percent_variance_y) < 0.1:
                break

    def determine_FWHM_axis(self, science_corrected, index=-1):
        annulus_sel = (self.target_distance < self.inner_radius)
        # We compute the sum of the total flux within the inner radius.
        total_flux = np.nansum(science_corrected * annulus_sel)
        # We compute the sum of the flux along each axis, within the inner radius.
        flux_x = np.nansum(science_corrected * annulus_sel, axis=0)
        flux_y = np.nansum(science_corrected * annulus_sel, axis=1)
        # We use the nansum function to avoid problems with the overscan region
        # we compute the cumulative sum along each axis, normalized to the total flux
        cumulative_sum_x = np.cumsum(flux_x) / total_flux
        cumulative_sum_y = np.cumsum(flux_y) / total_flux
        FWHM = []
        for normalized_cumulative_distribution in [cumulative_sum_x, cumulative_sum_y]:
            # Find the closest point to NCD= 0.15865 (-1 sigma)
            NCD_index_left = np.argmin(np.abs(normalized_cumulative_distribution - 0.15865))
            # Find the closest point to NCD= 0.84135 (+1 sigma)
            NCD_index_right = np.argmin(np.abs(normalized_cumulative_distribution - 0.84135))
            # We model the NCD around the -1sgima value with a polynomial curve.
            # The independet variable is actually the normalized cumulative distribution,
            # the depedent variable is the pixel position
            p_fitted = np.polynomial.Polynomial.fit(
                normalized_cumulative_distribution[NCD_index_left - 1: NCD_index_left + 2],
                self.ref_axis[NCD_index_left - 1: NCD_index_left + 2], deg=2
            )
            # We get a more precise estimate of the pixel value corresponding to the -1sigma position
            pixel_left = p_fitted(0.15865)
            # We repeat the step for the 1sigma value
            p_fitted = np.polynomial.Polynomial.fit(
                normalized_cumulative_distribution[NCD_index_right - 1: NCD_index_right + 2],
                self.ref_axis[NCD_index_right - 1: NCD_index_right + 2],
                deg=2)
            pixel_right = p_fitted(0.84135)
            FWHM_factor = 2 * np.sqrt(2 * np.log(2))  # = 2.35482
            FWHM.append((pixel_right - pixel_left) / 2. * FWHM_factor)
        if index != -1:
            self.fwhm_x[index] = FWHM[0]
            self.fwhm_y[index] = FWHM[1]
        return FWHM[0], FWHM[1], cumulative_sum_x, cumulative_sum_y

    def sky_background(self, science_corrected):
        annulus_selection = (
                (self.target_distance > self.inner_radius) &
                (self.target_distance <= self.outer_radius)
        )
        sky_flux_average = np.sum(science_corrected[annulus_selection]) / np.sum(annulus_selection)
        sky_flux_median = np.median(science_corrected[annulus_selection])
        #
        return sky_flux_average, sky_flux_median, annulus_selection

    def aperture_photometry(self, science_corrected, aperture_radius, index=0):
        #
        sky_flux_average, sky_flux_median, annulus_selection = self.sky_background(science_corrected)
        #
        science_sky_corrected = science_corrected - sky_flux_median  # sky_flux_average
        # Ricalcolo il centroide solo per avere la posizione precisa dopo sky background remove
        self.calc_refined_center(
            30, science_sky_corrected,
        )
        self.determine_FWHM_axis(science_corrected, index)
        aperture_selection = (self.target_distance < aperture_radius)
        self.aperture_star[index] = np.sum(science_sky_corrected[aperture_selection])
        self.x_arr[index] = self.x_refined
        self.y_arr[index] = self.y_refined
        self.sky_background_arr[index] = sky_flux_median
        return (np.sum(science_sky_corrected[aperture_selection]),
                science_sky_corrected, annulus_selection, sky_flux_average, sky_flux_median)
