import numpy as np
import matplotlib.pyplot as plt
import pickle
import batman
from pathlib import Path
from ldtk import SVOFilter, LDPSetCreator
SVOFilter.shortcuts

from AstrolabAnalysis.ANALYSIS.Modules.utility import TASTE_reader

class LimbDarkening(TASTE_reader):
    """
    todo
    """

    def __init__(self, params, current_file, taste_dir):
        self.sloan_g = SVOFilter('SLOAN/SDSS.g')
        self.tess_fr = SVOFilter('TESS')
        self.params = params
        self.current_file = current_file
        self.taste_dir = taste_dir
        TASTE_reader.__init__(self, taste_dir)

    def compute_limbd_coeff(self):
        stellar_params = self.params["stellar_parameters"]
        filters = [self.sloan_g, self.tess_fr]
        sc = LDPSetCreator(
            teff=(stellar_params["teff"][0], stellar_params["teff"][1]),
            logg=(stellar_params["logg"][0], stellar_params["logg"][1]),
            z=(stellar_params["z"][0], stellar_params["z"][1]),
            filters=filters
        )
        #
        ps = sc.create_profiles(nsamples=2000)
        ps.resample_linear_z(100)
        qm, qe = ps.coeffs_qd(do_mc=True, n_mc_samples=10000)
        ps.set_uncertainty_multiplier(10)
        chains = np.array(ps._samples['qd'])
        #
        u1_sloan_g_chains = chains[0, :, 0]
        u2_sloan_g_chains = chains[0, :, 1]
        u1_tess_chains = chains[1, :, 0]
        u2_tess_chains = chains[1, :, 1]
        # Compute the final coefficients
        self.u1_sloan_g = np.mean(u1_sloan_g_chains)
        self.u1_sloan_g_error = np.std(u1_sloan_g_chains)
        self.u2_sloan_g = np.mean(u2_sloan_g_chains)
        self.u2_sloan_g_error = np.std(u2_sloan_g_chains)
        self.u1_tess = np.mean(u1_tess_chains)
        self.u1_tess_error = np.std(u1_tess_chains)
        self.u2_tess = np.mean(u2_tess_chains)
        self.u2_tess_error = np.std(u2_tess_chains)
        print('Sloan g LD coefficients: u1 = {0:4.2f} \pm {1:4.2f}\tu2 = {2:4.2f} \pm {3:4.2f}'.format(
            self.u1_sloan_g, self.u1_sloan_g_error, self.u2_sloan_g, self.u2_sloan_g_error
        ))
        print('TESS LD coefficients: u1 = {0:4.2f} \pm {1:4.2f}\tu2 = {2:4.2f} \pm {3:4.2f}'.format(
            self.u1_tess, self.u1_tess_error, self.u2_tess, self.u2_tess_error
        ))
        # Plot TESS LD coefficients chain
        plt.figure(figsize=(4, 4))
        plt.title('TESS LD Coefficients')
        plt.xlabel('u1')
        plt.ylabel('u2')
        plt.scatter(u1_tess_chains, u2_tess_chains)
        plt.show()

    def save_coefficients(self):
        self.ld_dictionary = {
            'u1_sloan_g' : [self.u1_sloan_g, self.u1_sloan_g_error],
            'u2_sloan_g' : [self.u2_sloan_g, self.u2_sloan_g_error],
            'u1_tess' : [self.u1_tess, self.u1_tess_error],
            'u2_tess' : [self.u2_tess, self.u2_tess_error],
        }
        pickle.dump(self.ld_dictionary, open(str(Path(
            self.current_file, "AstrolabAnalysis", "ANALYSIS", "ld_coefficients.p")), "wb"))

    def check_coefficients(self):
        # Read TASTE data
        index_list = [0, 5, 6]
        taste_bjd_tdb, differential_allref_normalized, _ = TASTE_reader.read_taste_results(self, index_list)
        # Planetary parameters
        par = batman.TransitParams()
        par.t0 = self.params["planet_parameters"]["epoch"]
        par.per = self.params["planet_parameters"]["period"]
        par.rp = self.params["planet_parameters"]["rp"]
        par.a = self.params["planet_parameters"]["a"]
        par.inc = self.params["planet_parameters"]["inc"]
        par.ecc = self.params["planet_parameters"]["ecc"]
        par.w = self.params["planet_parameters"]["w"]
        # TASTE LD
        par.u = [self.u1_sloan_g, self.u2_sloan_g]
        par.limb_dark = "quadratic"
        # Plot TASTE light curve
        m_taste = batman.TransitModel(par, taste_bjd_tdb)
        taste_model_flux = m_taste.light_curve(par)
        plt.figure(figsize=(6, 4))
        plt.scatter(taste_bjd_tdb, differential_allref_normalized, s=2, label="TASTE measures")
        plt.plot(taste_bjd_tdb, taste_model_flux, lw=2, c='C1', label="batman model")
        plt.xlabel("BJD TDB")
        plt.ylabel("Relative flux")
        plt.title("TASTE lightcurve")
        plt.legend()
        plt.show()
        plt.close()
        #
        # TESS LD
        # TODO: implement the visualization also for TESS
        # Different sectors, so read a sector list and take the first one,
        # but wait to see how the part 2 of analysis works

    def execute_limbdark(self):
        """
        Execute the class LimbDarkening
        """
        self.compute_limbd_coeff()
        self.save_coefficients()
        self.check_coefficients()
        return self.ld_dictionary
