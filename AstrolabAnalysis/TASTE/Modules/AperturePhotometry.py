import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.stats import multivariate_normal
from numpy.polynomial import Polynomial

from AstrolabAnalysis.TASTE.Modules.general_functions import make_circle_around_star2, make_annulus_around_star


class AperturePhotometry:
    
    def __init__(
        self, full_science_stack, julian_date, airmass, skip_border, tar_star, ref_star1, ref_star2,
        vmin, avoid_input, result_folder, yaml_aperture
    ):
        print("----------------------- APERTURE PHOTOMETRY -----------------------")
        self.full_science_stack = full_science_stack
        self.julian_date = julian_date
        self.airmass = airmass
        self.skip_border = skip_border
        self.tar_star = tar_star
        self.ref_star1 = ref_star1
        self.ref_star2 = ref_star2
        self.vmin = vmin
        self.avoid_input = avoid_input
        self.result_folder = result_folder
        self.yaml_aperture = yaml_aperture
        self.time_offset = None
        self.aperture_ref2 = None
        self.aperture_ref1 = None
        self.aperture_tar = None
        self.aperture = None

    def assignment_plus_first_analysis(self):
        self.full_science_stack[:, :, :self.skip_border] = 0
        self.full_science_stack[:, :, -self.skip_border:] = 0
        # Show for first
        (_, science_sky_corrected, annulus_selection, sky_flux_average,
         sky_flux_median) = self.tar_star.aperture_photometry(
            self.full_science_stack[0, :, :], self.tar_star.radius
        )
        inner_selection = (self.tar_star.target_distance < self.tar_star.inner_radius)
        total_flux = np.sum(science_sky_corrected[inner_selection])
        radius_array = np.arange(0, self.tar_star.inner_radius + 1., 0.5)
        flux_vs_radius = np.zeros_like(radius_array)
        for ii, aperture_radius in enumerate(radius_array):
            aperture_selection = (self.tar_star.target_distance < aperture_radius)
            flux_vs_radius[ii] = np.sum(science_sky_corrected[aperture_selection]) / total_flux
        #
        print('vmin:  {0:.1f}    vmax: {1:.1f}'.format(self.vmin, self.tar_star.vmax))
        fig, ax = plt.subplots(1, 1, figsize=(10, 8), dpi=300)
        im1 = ax.imshow(self.full_science_stack[0, :, :], cmap=plt.colormaps['inferno'],
                        norm=colors.LogNorm(vmin=self.vmin, vmax=self.tar_star.vmax), origin='lower')
        plt.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)
        # Cut the plot around the target star, with some margin with respect to the inner radius
        ax.set_xlim(self.tar_star.x_refined - self.tar_star.outer_radius * 1.2,
                    self.tar_star.x_refined + self.tar_star.outer_radius * 1.2)
        ax.set_ylim(self.tar_star.y_refined - self.tar_star.outer_radius * 1.2,
                    self.tar_star.y_refined + self.tar_star.outer_radius * 1.2)
        make_circle_around_star2(self.tar_star.x_refined, self.tar_star.y_refined, self.tar_star.inner_radius,
                                 label='inner radius', axf=ax)
        make_circle_around_star2(self.tar_star.x_refined, self.tar_star.y_refined, self.tar_star.outer_radius,
                                 color='y', label='outer radius',
                                 axf=ax)
        ax.set_xlabel(' X [pixels]')
        ax.set_ylabel(' Y [pixels]')
        ax.legend(loc='upper left')
        plt.savefig(str(Path(self.result_folder, "5_aperture", "first_image_inner_outer.png")))
        plt.show()
        plt.close(fig)
        print(self.tar_star.name + ' number of pixels included in the annulus: {0:7.0f}'.format(
            np.sum(annulus_selection)))
        print(self.tar_star.name + ' average Sky flux: {0:7.1f} photons/pixel'.format(sky_flux_average))
        print(self.tar_star.name + ' median Sky flux: {0:7.1f} photons/pixel'.format(sky_flux_median))
        #
        fig, ax = plt.subplots(1, 1, figsize=(10, 8), dpi=300)
        ax.scatter(radius_array, flux_vs_radius, c='C1')
        ax.axhline(0.80)
        ax.axhline(0.85)
        ax.axhline(0.90)
        ax.axhline(0.95)
        ax.axhline(0.99)
        ax.set_xlabel('Aperture [pixels]')
        ax.set_ylabel('Fractional flux within the aperture')
        plt.savefig(str(Path(self.result_folder, "5_aperture", "first_image_flux.png")))
        plt.show()
        plt.close(fig)
        #
        xy_range = np.arange(-self.tar_star.inner_radius * 1.2, self.tar_star.inner_radius * 1.2, 0.1)
        X_gauss, Y_gauss = np.meshgrid(xy_range, xy_range)
        pos = np.dstack((X_gauss, Y_gauss))
        gauss_distance = np.sqrt(X_gauss ** 2 + Y_gauss ** 2)
        plot_range = np.arange(0, self.tar_star.inner_radius, 0.1)
        # Let's compare two different values for the covarince:
        mv_normal_cov05 = multivariate_normal(mean=[0., 0.], cov=5., allow_singular=False)
        mv_normal_cov05_pdf = mv_normal_cov05.pdf(pos)
        plot_cov05_flux = np.zeros_like(plot_range)
        mv_normal_cov10 = multivariate_normal(mean=[0., 0.], cov=10., allow_singular=False)
        mv_normal_cov10_pdf = mv_normal_cov10.pdf(pos)
        plot_cov10_flux = np.zeros_like(plot_range)
        mv_normal_cov20 = multivariate_normal(mean=[0., 0.], cov=20., allow_singular=False)
        mv_normal_cov20_pdf = mv_normal_cov20.pdf(pos)
        plot_cov20_flux = np.zeros_like(plot_range)
        for ii, aperture_radius in enumerate(plot_range):
            pdf_selection = (gauss_distance < aperture_radius)
            plot_cov05_flux[ii] = np.sum(mv_normal_cov05_pdf[pdf_selection])
            plot_cov10_flux[ii] = np.sum(mv_normal_cov10_pdf[pdf_selection])
            plot_cov20_flux[ii] = np.sum(mv_normal_cov20_pdf[pdf_selection])
        fig, ax = plt.subplots(1, 1, figsize=(10, 8), dpi=300)
        ax.scatter(radius_array, flux_vs_radius, c='C1', label='Measurmeents')
        ax.plot(plot_range, plot_cov05_flux / plot_cov05_flux[-1], c='C2', label='Covariance= 5.')
        ax.plot(plot_range, plot_cov10_flux / plot_cov10_flux[-1], c='C3', label='Covariance=10.')
        ax.plot(plot_range, plot_cov20_flux / plot_cov20_flux[-1], c='C4', label='Covariance=20.')
        ax.set_xlabel('Aperture [pixels]')
        ax.set_ylabel('Fractional flux within the aperture')
        ax.legend()
        plt.savefig(str(Path(self.result_folder, "5_aperture", "first_image_flux_in_ap.png")))
        plt.show()
        plt.close(fig)

    def aperture_photometry(self):
        if self.avoid_input:
            self.aperture = self.yaml_aperture["aperture"]
        else:
            print('Enter the chosen aperture')
            self.aperture = int(input())
        #
        for aperture_rad in [self.aperture - 1, self.aperture + 1, self.aperture]:
            self.aperture_ref2 = np.empty(len(self.full_science_stack[:, 0, 0]))
            self.aperture_ref1 = np.empty(len(self.full_science_stack[:, 0, 0]))
            self.aperture_tar = np.empty(len(self.full_science_stack[:, 0, 0]))
            for ii_science, science_img in enumerate(self.full_science_stack[:, 0, 0]):
                self.aperture_tar[ii_science], _, _, _, _ = self.tar_star.aperture_photometry(
                    self.full_science_stack[ii_science, :, :], aperture_rad, ii_science
                )
                #
                self.aperture_ref1[ii_science], _, _, _, _ = self.ref_star1.aperture_photometry(
                    self.full_science_stack[ii_science, :, :], aperture_rad, ii_science
                )
                #
                self.aperture_ref2[ii_science], _, _, _, _ = self.ref_star2.aperture_photometry(
                    self.full_science_stack[ii_science, :, :], aperture_rad, ii_science
                )
            #
            fig, ax = plt.subplots(1, 1, figsize=(10, 8), dpi=300)
            ax.scatter(self.julian_date, self.aperture_tar / self.aperture_ref1, s=2)
            ax.set_ylim(float(self.yaml_aperture["ylim1_aperture_ref1"]), float(self.yaml_aperture["ylim2_aperture_ref1"]))
            ax.set_title(f"Aperture " + str(aperture_rad))
            plt.savefig(str(Path(self.result_folder, "5_aperture", "aperture" + str(aperture_rad) + ".png")))
            plt.show()
            plt.close(fig)
            #
            fig, ax = plt.subplots(1, 1, figsize=(10, 8), dpi=300)
            ax.scatter(self.julian_date, self.aperture_tar / self.aperture_ref1, s=2)
            ax.set_ylim(float(self.yaml_aperture["ylim1_aperture_ref1"]), float(self.yaml_aperture["ylim2_aperture_ref1"]))
            ax.set_xlim(float(self.yaml_aperture["xlim1_aperture_zoom"]), float(self.yaml_aperture["xlim2_aperture_zoom"]))
            ax.set_title(f"Aperture " + str(aperture_rad))
            plt.savefig(str(Path(self.result_folder, "5_aperture", "aperture" + str(aperture_rad) + "_restricted.png")))
            plt.show()
            plt.close(fig)
            
    def weather_info(self):
        fig, ax = plt.subplots(1, 1, figsize=(10, 8), dpi=300)
        im1 = ax.imshow(
            self.full_science_stack[0, :, :], cmap=plt.colormaps['inferno'],
            norm=colors.LogNorm(vmin=self.vmin, vmax=self.tar_star.vmax), origin='lower'
        )
        plt.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)
        make_circle_around_star2(self.tar_star.x_refined, self.tar_star.y_refined, self.aperture, color="blue",
                                 label='Aperture', axf=ax)
        make_annulus_around_star(
            self.tar_star.x_refined, self.tar_star.y_refined, self.tar_star.inner_radius, self.tar_star.outer_radius,
            label='Target', ax=ax
        )
        make_annulus_around_star(
            self.ref_star1.x_refined, self.ref_star1.y_refined, self.ref_star1.inner_radius,
            self.ref_star1.outer_radius,
            label='Reference #1', color='g', ax=ax
        )
        make_annulus_around_star(
            self.ref_star2.x_refined, self.ref_star2.y_refined, self.ref_star2.inner_radius,
            self.ref_star2.outer_radius, label='Reference #2',
            color='r', ax=ax
        )
        ax.set_xlabel(' X [pixels]')
        ax.set_ylabel(' Y [pixels]')
        ax.legend()
        plt.savefig(str(Path(self.result_folder, "5_aperture", "stars.png")))
        plt.show()
        plt.close(fig)
        #
        normalization_index = int(self.yaml_aperture["normalization_index"])
        self.time_offset = float(self.yaml_aperture["time_offset"])
        fig, axs = plt.subplots(5, 1, figsize=(8, 10), dpi=300)
        # Reduce vertical space between axes
        fig.subplots_adjust(hspace=0.05)
        axs[0].scatter(self.julian_date - self.time_offset, self.aperture_tar / self.aperture_tar[normalization_index],
                       s=2, zorder=3, c='C0', label='Target')
        axs[0].scatter(self.julian_date - self.time_offset,
                       self.aperture_ref1 / self.aperture_ref1[normalization_index], s=2, zorder=2, c='C1',
                       label='Ref #1')
        axs[0].scatter(self.julian_date - self.time_offset,
                       self.aperture_ref2 / self.aperture_ref2[normalization_index], s=2, zorder=1, c='C2',
                       label='Ref #2')
        #
        for i, science in enumerate(self.full_science_stack):
            self.tar_star.sky_background(science)
            self.tar_star.sky_background(science)
            self.tar_star.sky_background(science)
        # axs[0].set_yticks(np.arange(0.90, 1.1, 0.025))
        # axs[0].set_ylim(0.88, 1.052)
        axs[0].set_ylabel('Normalized flux')
        axs[0].legend()
        axs[1].scatter(self.julian_date - self.time_offset, self.airmass, s=2, c='C0', label='Airmass')
        axs[1].set_ylabel('Airmass')
        axs[2].scatter(self.julian_date - self.time_offset, self.tar_star.sky_background_arr, s=2, zorder=3, c='C0',
                       label='Target')
        axs[2].scatter(self.julian_date - self.time_offset, self.ref_star1.sky_background_arr, s=2, zorder=2, c='C1',
                       label='Ref #1')
        axs[2].scatter(self.julian_date - self.time_offset, self.ref_star2.sky_background_arr, s=2, zorder=1, c='C2',
                       label='Ref #2')
        axs[2].set_ylabel('Sky background [ph]')
        # axs[2].set_yscale('log')
        axs[2].legend()
        axs[3].scatter(self.julian_date - self.time_offset, self.tar_star.x_arr - self.tar_star.x_arr[0], s=2, zorder=3,
                       c='C0', label='X direction')
        axs[3].scatter(self.julian_date - self.time_offset, self.tar_star.y_arr - self.tar_star.y_arr[0], s=2, zorder=2,
                       c='C1', label='Y direction')
        axs[3].set_ylabel('Telescope drift [pixels]')
        axs[3].legend()
        axs[4].scatter(self.julian_date - self.time_offset, self.tar_star.fwhm_x, s=2, zorder=3, c='C0', label='X direction')
        axs[4].scatter(self.julian_date - self.time_offset, self.tar_star.fwhm_y, s=2, zorder=2, c='C1', label='Y direction')
        axs[4].set_ylabel('Target fWHM [pixels]')
        axs[4].legend()
        axs[4].set_xlabel('BJD-TDB - {0:.1f} [days]'.format(self.time_offset))
        plt.savefig(str(Path(self.result_folder, "5_aperture", "weather_instrument.png")))
        plt.show()
        plt.close(fig)
        
    def differential_photometry(self):
        fig, ax = plt.subplots(1, 1, figsize=(10, 8), dpi=300)
        ax.scatter(self.julian_date - self.time_offset, self.aperture_tar / self.aperture_ref1, s=2, c='C0', label='Ref #1')
        ax.set_xlabel('BJD-TDB - {0:.1f} [days]'.format(self.time_offset))
        ax.set_ylabel('Differential photometry')
        ax.set_ylim(float(self.yaml_aperture["ylim1_aperture_ref1"]), float(self.yaml_aperture["ylim2_aperture_ref1"]))
        ax.legend()
        ax.set_title("Reference 1")
        plt.savefig(str(Path(self.result_folder, "5_aperture", "ref1_star_ap.png")))
        plt.show()
        plt.close(fig)
        #
        fig, ax = plt.subplots(1, 1, figsize=(10, 8), dpi=300)
        ax.scatter(self.julian_date - self.time_offset, self.aperture_tar / self.aperture_ref2, s=2, c='C1', label='Ref #2')
        ax.set_xlabel('BJD-TDB - {0:.1f} [days]'.format(self.time_offset))
        ax.set_ylabel('Differential photometry')
        ax.set_ylim(float(self.yaml_aperture["ylim1_aperture_ref2"]), float(self.yaml_aperture["ylim2_aperture_ref2"]))
        ax.legend()
        ax.set_title("Reference 2")
        plt.savefig(str(Path(self.result_folder, "5_aperture", "ref2_star_ap.png")))
        plt.show()
        plt.close(fig)
        #
        fig, ax = plt.subplots(1, 1, figsize=(10, 8), dpi=300)
        ax.scatter(self.julian_date - self.time_offset, self.aperture_tar / (self.aperture_ref1 + self.aperture_ref2), s=2,
                    c='C3', label='Sum of refs')
        ax.set_xlabel('BJD-TDB - {0:.1f} [days]'.format(self.time_offset))
        ax.set_ylabel('Differential photometry')
        ax.set_ylim(float(self.yaml_aperture["ylim1_aperture_sum"]), float(self.yaml_aperture["ylim2_aperture_sum"]))
        ax.legend()
        ax.set_title("Reference 1 + 2")
        plt.savefig(str(Path(self.result_folder, "5_aperture", "sum_star_ap.png")))
        plt.show()
        plt.close(fig)
        #
        differential_ref01 = self.aperture_tar / self.aperture_ref1
        differential_ref02 = self.aperture_tar / self.aperture_ref2
        differential_allref = self.aperture_tar / (self.aperture_ref1 + self.aperture_ref2)
        #
        target_aperture_error = np.sqrt(self.aperture_tar)
        reference01_aperture_error = np.sqrt(self.aperture_ref1)
        reference02_aperture_error = np.sqrt(self.aperture_ref2)
        # Error propagation
        differential_ref01_error = differential_ref01 * np.sqrt(
            (target_aperture_error / self.aperture_tar) ** 2 +
            (reference01_aperture_error / self.aperture_ref1) ** 2
        )
        differential_ref02_error = differential_ref02 * np.sqrt(
            (target_aperture_error / self.aperture_tar) ** 2 +
            (reference02_aperture_error / self.aperture_ref2) ** 2
        )
        differential_allref_error = differential_allref * np.sqrt(
            (target_aperture_error / self.aperture_tar) ** 2 +
            (reference01_aperture_error / (self.aperture_ref1 + self.aperture_ref2)) ** 2 +
            (reference02_aperture_error / (self.aperture_ref1 + self.aperture_ref2)) ** 2
        )
        #
        limit_transit_inf = self.yaml_aperture["xlim1_aperture_zoom"]
        limit_transit_sup = self.yaml_aperture["xlim2_aperture_zoom"]
        fig, ax = plt.subplots(1, 1, figsize=(10, 8), dpi=300)
        ax.scatter(self.julian_date - self.time_offset, differential_allref, s=2, label='All refs')
        ax.errorbar(self.julian_date - self.time_offset, differential_allref,
                     fmt=' ', c='k', alpha=0.25, zorder=-1, yerr=differential_allref_error)
        ax.legend()
        ax.axvline(limit_transit_inf - self.time_offset, c='C3')
        ax.axvline(limit_transit_sup - self.time_offset, c='C3')
        ax.set_xlabel('BJD-TDB - {0:.1f} [days]'.format(self.time_offset))
        ax.set_ylabel('Differential photometry')
        # ax.ylim(0.240, 0.250)
        plt.savefig(str(Path(self.result_folder, "5_aperture", "differential_photometry.png")))
        plt.show()
        plt.close(fig)
        #
        out_transit_selection = (
            ((self.julian_date < limit_transit_inf) | (self.julian_date > limit_transit_sup))
        )
        bjd_median = np.median(self.julian_date)
        poly_ref01_deg01_pfit = Polynomial.fit(
            self.julian_date[out_transit_selection] - bjd_median,
            differential_ref01[out_transit_selection], deg=1
        )
        poly_ref02_deg01_pfit = Polynomial.fit(
            self.julian_date[out_transit_selection] - bjd_median,
            differential_ref02[out_transit_selection], deg=1
        )
        poly_allref_deg01_pfit = Polynomial.fit(
            self.julian_date[out_transit_selection] - bjd_median,
            differential_allref[out_transit_selection], deg=1
        )
        #
        differential_ref01_normalized = differential_ref01 / poly_ref01_deg01_pfit(self.julian_date - bjd_median)
        differential_ref02_normalized = differential_ref02 / poly_ref02_deg01_pfit(self.julian_date - bjd_median)
        differential_allref_normalized = differential_allref / poly_allref_deg01_pfit(self.julian_date - bjd_median)
        # Compute the standard deviation (normalized error)
        differential_ref01_normalized_error = np.std(differential_ref01_normalized[out_transit_selection])
        differential_ref02_normalized_error = np.std(differential_ref02_normalized[out_transit_selection])
        differential_allref_normalized_error = np.std(differential_allref_normalized[out_transit_selection])
        fig, ax = plt.subplots(1, 1, figsize=(10, 8), dpi=300)
        ax.scatter(self.julian_date - self.time_offset, differential_ref01_normalized, s=2)
        ax.scatter(self.julian_date - self.time_offset, differential_ref02_normalized, s=2)
        ax.scatter(self.julian_date - self.time_offset, differential_allref_normalized, s=2)
        ax.axvline(limit_transit_inf - self.time_offset, c='C3')
        ax.axvline(limit_transit_sup - self.time_offset, c='C3')
        # ax.ylim(0.975, 1.025)
        ax.set_xlabel('BJD-TDB - {0:.1f} [days]'.format(self.time_offset))
        ax.set_ylabel('Normalized differential photometry')
        plt.savefig(str(Path(self.result_folder, "5_aperture", "normalized_differential_photometry.png")))
        plt.show()
        plt.close(fig)
        print('Standard deviation aperture 08 reference #1:    {0:.7f}'.format(
            np.std(differential_ref01_normalized[out_transit_selection])))
        print('Standard deviation aperture 08 reference #2:    {0:.7f}'.format(
            np.std(differential_ref02_normalized[out_transit_selection])))
        print('Standard deviation aperture 08 all references : {0:.7f}'.format(
            np.std(differential_allref_normalized[out_transit_selection])))
        #
        with open(str(Path(self.result_folder, "5_aperture", "aperture_final.pkl")), "wb") as f:
            pickle.dump(self.julian_date, f)
            pickle.dump(differential_ref01_normalized, f)
            pickle.dump(differential_ref01_normalized_error, f)
            pickle.dump(differential_ref02_normalized, f)
            pickle.dump(differential_ref02_normalized_error, f)
            pickle.dump(differential_allref_normalized, f)
            pickle.dump(differential_allref_normalized_error, f)
            pickle.dump(differential_ref01, f)
            pickle.dump(differential_ref01_error, f)
            pickle.dump(differential_ref02, f)
            pickle.dump(differential_ref02_error, f)
            pickle.dump(differential_allref, f)
            pickle.dump(differential_allref_error, f)

    def execute_aperture(self):
        self.assignment_plus_first_analysis()
        self.aperture_photometry()
        self.weather_info()
        self.differential_photometry()
