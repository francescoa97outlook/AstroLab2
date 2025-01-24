import batman
import pickle
import emcee
import corner
import pygtc
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from multiprocessing import Pool

from AstrolabAnalysis.ANALYSIS.Modules.utility import TASTE_reader, TESS_sector


class TransitFit(TASTE_reader):
    def __init__(self, params, current_file, taste_dir, list_yaml_sectors, ld_dictionary, results_dir):
        self.params = params
        self.current_file = current_file
        self.taste_dir = taste_dir
        TASTE_reader.__init__(self, taste_dir)
        self.list_yaml_sectors = list_yaml_sectors
        self.ld_dictionary = ld_dictionary
        self.results_dir = results_dir

    def get_taste_tess_results(self):
        # Read the results of the TASTE analysis
        ind_list = [0, 11, 12]
        self.taste_bjd_tdb, self.differential_allref, self.differential_allref_error = TASTE_reader.read_taste_results(self, ind_list)
        # Read the results of the TESS analysis
        self.sector_list = []
        for yaml_sector_name in self.list_yaml_sectors["yaml_list"]:
            self.sector_list.append(TESS_sector(self.current_file, yaml_sector_name))

    def populate_theta(self):
        theta = np.empty(14)
        # Planet specific parameters
        planet_params = self.params["planet_parameters"]
        theta[0] = planet_params["epoch"]
        theta[1] = planet_params["period"]
        theta[2] = planet_params["rp"]
        theta[3] = planet_params["a"]
        theta[4] = planet_params["inc"]
        # Limb darkening parameters
        theta[5] = self.ld_dictionary["u1_tess"][0]
        theta[6] = self.ld_dictionary["u2_tess"][0]
        theta[7] = self.ld_dictionary["u1_sloan_g"][0]
        theta[8] = self.ld_dictionary["u2_sloan_g"][0]
        # Polynomial coefficients
        # TODO: where is this value coming from?
        theta[9] = 0.155
        theta[10] = 0.0
        theta[11] = 0.0
        # Jitter parameters
        # TODO: change these values
        theta[12] = 0.01
        theta[13] = 0.01
        return theta

    def populate_boundaries(self, theta):
        boundaries = np.empty([2, len(theta)])
        boundaries[:, 0] = [theta[0] - 0.5, theta[0] + 0.5]
        boundaries[:, 1] = [theta[1] - 0.5, theta[1] + 0.5]
        boundaries[:, 2] = [0.0, 0.5]
        boundaries[:, 3] = [0.0, 20.]
        boundaries[:, 4] = [0.00, 90.0]
        boundaries[:, 5] = [0.00, 1.0]
        boundaries[:, 6] = [0.00, 1.0]
        boundaries[:, 7] = [0.00, 1.0]
        boundaries[:, 8] = [0.00, 1.0]
        boundaries[:, 9] = [0.00, 1.0]
        boundaries[:, 10] = [-1.0, 1.0]
        boundaries[:, 11] = [-1.0, 1.0]
        boundaries[:, 12] = [0.0, 0.05]
        boundaries[:, 13] = [0.0, 0.05]
        return boundaries

    def log_likelihood(self, theta):
        params = batman.TransitParams()
        params.t0 = theta[0]
        params.per = theta[1]
        params.rp = theta[2]
        params.a = theta[3]
        params.inc = theta[4]
        params.ecc = 0.
        params.w = 90.
        #
        # TASTE model
        params.u = [theta[7], theta[8]]
        params.limb_dark = "quadratic"
        median_bjd = np.median(self.taste_bjd_tdb)
        polynomial_trend = theta[9]+theta[10]*(self.taste_bjd_tdb-median_bjd) + theta[11]*(self.taste_bjd_tdb-median_bjd)**2
        # Initialize model
        m_taste = batman.TransitModel(params, self.taste_bjd_tdb)
        # Calculate light_curve
        taste_flux = m_taste.light_curve(params) * polynomial_trend
        # Likelihood computation
        taste_errors_with_jitter = self.differential_allref_error**2 + theta[13]**2
        chi2_taste = np.sum((self.differential_allref-taste_flux)**2 / taste_errors_with_jitter)
        sum_ln_sigma_taste = np.sum(np.log(taste_errors_with_jitter))
        N = len(taste_errors_with_jitter)
        #
        # TESS model
        params.u = [theta[5], theta[6]]
        chi2_tess = 0.0
        sum_ln_sigma_tess = 0.0
        for sector in self.sector_list:
            # Initialize model
            m_tess = batman.TransitModel(params, sector.tess_bjd_tdb)
            # Calculate light curve
            tess_flux = m_tess.light_curve(params)
            tess_errors_with_jitter = (sector.tess_normalized_flux_error ** 2) + (theta[12] ** 2)
            chi2_tess += np.sum((sector.tess_normalized_flux - tess_flux)**2 / tess_errors_with_jitter)
            sum_ln_sigma_tess += np.sum(np.log(tess_errors_with_jitter))
            N += len(tess_errors_with_jitter)
        #
        log_likelihood = -0.5 * (N * np.log(2*np.pi) + chi2_tess + chi2_taste + sum_ln_sigma_tess + sum_ln_sigma_taste)
        return log_likelihood

    def log_prior(self, theta):
        prior = 0.0
        prior += np.log(stats.norm.pdf(theta[5], loc=self.tess_u1, scale=0.10))
        prior += np.log(stats.norm.pdf(theta[6], loc=self.tess_u2, scale=0.10))
        prior += np.log(stats.norm.pdf(theta[7], loc=self.taste_u1, scale=0.10))
        prior += np.log(stats.norm.pdf(theta[8], loc=self.taste_u2, scale=0.10))
        # TODO: Should we specify a uniform prior on the other parameters?
        #prior += np.log(stats.uniform.pdf(theta[0], loc=self.epoch-0.05, scale=0.1))
        return prior

    def plot_parameter_chain(self, samples):
        labels = ["Tc", "P", "Rp/Rs", "a/Rs", "inc", "u1_tess", "u2_tess",
                  "u1_taste", "u2_taste", "0th poly", "1st poly", "2nd poly",
                  "jitter_tess", "jitter_taste"]
        # Plot first seven parameters
        fig1, axes1 = plt.subplots(7, figsize=(10, 7), sharex=True)
        for i in range(7):
            ax1 = axes1[i]
            ax1.plot(samples[:, :, i], "k", alpha=0.3)
            ax1.set_xlim(0, len(samples))
            ax1.set_ylabel(labels[i])
            ax1.yaxis.set_label_coords(-0.1, 0.5)
        axes1[-1].set_xlabel("step number")
        fig1.savefig(str(Path(self.results_dir, "parameter_chain_1.png")))
        #plt.show()
        plt.close()
        # Plot remaining parameters
        fig2, axes2 = plt.subplots(7, figsize=(10, 7), sharex=True)
        for i in range(7, 14):
            ax2 = axes2[i-7]
            ax2.plot(samples[:, :, i], "k", alpha=0.3)
            ax2.set_xlim(0, len(samples))
            ax2.set_ylabel(labels[i])
            ax2.yaxis.set_label_coords(-0.1, 0.5)
        axes2[-1].set_xlabel("step number")
        fig2.savefig(str(Path(self.results_dir, "parameter_chain_2.png")))
        #plt.show()
        plt.close()

    def plot_corner_better(self, flat_samples, names, estimates):
        # List of Gaussian curves to plot to represent priors
        priors = (None,
                  None,
                  None,
                  None,
                  None,
                  (self.tess_u1, 0.1),
                  (self.tess_u2, 0.1),
                  (self.taste_u1, 0.1),
                  (self.taste_u2, 0.1),
                  None,
                  None,
                  None,
                  None,
                  None,)
        # List of parameter ranges to show,
        paramRanges = (None,
                       None,
                       None,
                       None,
                       None,
                       None,
                       None,
                       None,
                       None,
                       None,
                       None,
                       None,
                       None,
                       None)
        #truths = (list(estimates[1, :]),
                  #list(estimates[0, :]),
                  #list(estimates[2, :])
        #)
        truthLineStyles = ('--', ':', ':')
        GTC = pygtc.plotGTC(chains=flat_samples, paramNames=names,
                            priors=priors, figureSize=8.5,
                            paramRanges=paramRanges,
                            smoothingKernel=0,
                            #truths=truths,
                            #truthLineStyles=truthLineStyles,
                            plotName=str(Path(self.results_dir, "corner_plot.pdf")))
        plt.savefig(str(Path(self.results_dir, "corner_plot.png")))
        #plt.show()

    def plot_model_on_data(self, flat_sample, theta):
        params = batman.TransitParams()
        params.t0 = theta[0]
        params.per = theta[1]
        params.rp = theta[2]
        params.a = theta[3]
        params.inc = theta[4]
        params.ecc = 0.
        params.w = 90.
        #
        # TASTE model
        params.u = [theta[7], theta[8]]
        params.limb_dark = "quadratic"
        median_bjd = np.median(self.taste_bjd_tdb)
        polynomial_trend = theta[9]+theta[10]*(self.taste_bjd_tdb-median_bjd) + theta[11]*(self.taste_bjd_tdb-median_bjd)**2
        # Initialize model
        m_taste = batman.TransitModel(params, self.taste_bjd_tdb)
        # Calculate light_curve
        taste_flux = m_taste.light_curve(params) * polynomial_trend
        #
        # Plot TASTE
        plt.figure(figsize=(6, 4))
        plt.scatter(self.taste_bjd_tdb, self.differential_allref, s=2, label="TASTE data")
        plt.plot(self.taste_bjd_tdb, taste_flux, lw=2, c='C1', label="final model")
        plt.xlabel("BJD TDB")
        plt.ylabel("Relative flux")
        plt.legend()
        plt.tight_layout()
        plt.savefig(str(Path(self.results_dir, "taste_final_model_on_data.png")))
        #plt.show()
        plt.close()
        #
        # TESS model
        params.u = [theta[5], theta[6]]
        for sector in self.sector_list:
            # Initialize model
            m_tess = batman.TransitModel(params, sector.tess_bjd_tdb)
            # Calculate light curve
            tess_flux = m_tess.light_curve(params)
            plt.figure(figsize=(6, 4))
            plt.scatter(sector.tess_bjd_tdb, sector.tess_normalized_flux, s=2, label="TESS data, sector {0}".format(sector.sector_number))
            plt.plot(sector.tess_bjd_tdb, tess_flux, lw=2, c='C1', label="final model")
            plt.xlabel("BJD TDB")
            plt.ylabel("Relative flux")
            plt.legend()
            plt.tight_layout()
            plt.savefig(str(Path(self.results_dir, "tess_final_model_on_data_sector{0}.png".format(sector.sector_number))))
            #plt.show()
            plt.close()

            folded_tess_time = (sector.tess_bjd_tdb - params.t0 - params.per / 2. ) % params.per - params.per / 2. 
            folded_range = np.arange(- params.per/2.,  params.per/2., 0.001)
            temp = params.t0
            params.t0 = 0.                     #time of inferior conjunction
            m_folded_tess = batman.TransitModel(params, folded_range)    #initializes model
            tess_folded_flux =m_folded_tess.light_curve(params)          #calculates light curv
            plt.figure(figsize=(6, 4))
            plt.scatter(folded_tess_time, sector.tess_normalized_flux, s=2, label="TESS data, sector {0}".format(sector.sector_number))
            plt.plot(folded_range, tess_folded_flux, lw=2, c='C1', label="final model")
            plt.xlim(-0.2, 0.2)
            plt.xlabel("Time from mid-transit [days]")
            plt.ylabel("Relative flux")
            plt.legend()
            plt.tight_layout()
            plt.savefig(str(Path(self.results_dir, "tess_final_model_on_folded_data_sector{0}.png".format(sector.sector_number))))
            #plt.show()
            plt.close()
            params.t0 = temp

    def execute(self):
        self.get_taste_tess_results()
        theta = self.populate_theta()
        # Parameters used in the prior
        self.epoch = theta[0]
        self.tess_u1 = theta[5]
        self.tess_u2 = theta[6]
        self.taste_u1 = theta[7]
        self.taste_u2 = theta[8]
        # Boundaries
        boundaries = self.populate_boundaries(theta)
        #
        global log_probability
        def log_probability(theta):
            sel = (theta < boundaries[0, :]) | (theta > boundaries[1, :])
            if np.sum(sel) > 0:
                return -np.inf
            log_prob = self.log_prior(theta)
            log_prob += self.log_likelihood(theta)
            return log_prob
        #
        # MCMC sample
        try:
            with open(str(Path(self.current_file, "AstrolabAnalysis", "ANALYSIS", "emcee_sampler.p")), 'rb') as f:
                print("Using previously computed sampler file...")
                sampler = pickle.load(f)
        except FileNotFoundError:
            print("Sampler file not found. Proceeding with new computation...")
            # increasing the walker will make each iteration slower
            # increasing the steps will make the total time longer
            # nwalkers = 50
            # nsteps = 20000
            nwalkers = 50
            nsteps = 10000
            ndim = len(theta)
            # We initialize the walkers in a tiny Gaussian ball around our approximate result
            starting_point = theta + np.abs(1e-5 * np.random.randn(nwalkers, ndim))
            with Pool() as pool:
                sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, pool=pool)
                sampler.run_mcmc(starting_point, nsteps, progress=True)
            pickle.dump(sampler, open(str(Path(self.current_file, "AstrolabAnalysis", "ANALYSIS", "emcee_sampler.p")),'wb'))
        #
        # Analyze the results
        samples = sampler.get_chain()
        flat_samples = sampler.get_chain(discard=2500, thin=100, flat=True)
        # List of parameter names
        names = ['Tc',
                 'P',
                 'Rp/Rs',
                 'a/Rs',
                 'inc',
                 'u1 ts',
                 'u2 ts',
                 'u1 tt',
                 'u2 tt',
                 '$0^{th}$ pol',
                 '$1^{st}$ pol',
                 '$2^{nd}$ pol',
                 'jit. ts',
                 'jit. tt']
        self.plot_parameter_chain(samples)
        self.plot_model_on_data(flat_samples, theta)
        # Print the final results
        ndim = len(theta)
        final_results = ""
        estimates = np.empty(shape=(3, ndim))
        for i in range(ndim):
            mcmc = np.percentile(flat_samples[:, i], [15.865, 50, 84.135])
            estimates[:, i] = mcmc
            q = np.diff(mcmc)
            line = "{0}\t= {1:.7f} (+{2:.7f} -{3:.7f})".format(names[i], mcmc[1], q[0], q[1])
            final_results += (line + "\n")
            print(line)
        with open(str(Path(self.results_dir, "final_results.txt")), "w") as f:
            print(final_results, file=f)
        # Print corner plot
        self.plot_corner_better(flat_samples, names, estimates)
