import yaml
import os
import pickle
from pathlib import Path

from AstrolabAnalysis.ANALYSIS.Modules.LimbDarkening import LimbDarkening
from AstrolabAnalysis.ANALYSIS.Modules.TransitFit import TransitFit

current_file = Path(__file__).parent
os.chdir(current_file)

if __name__ == '__main__':
    # Planetary system parameters
    with open(str(Path(current_file, "AstrolabAnalysis", "ANALYSIS", "parameters.yaml"))) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    # List of TESS sector considered
    with open(str(Path(current_file, "AstrolabAnalysis", "TESS", "list_yaml_sectors.yaml"))) as in_f:
        list_yaml_sectors = yaml.load(in_f, Loader=yaml.FullLoader)
    # Define TASTE directory with the results from previous analysis
    taste_dir = str(Path(current_file, "GJ3470_20220328", "results", "5_aperture"))
    # Analysis results directory
    results_dir = str(Path(current_file, "Results", "Final_Analysis"))
    #
    # Try to open the limbdark coefficient file
    # If it does no exist, create it
    try:
        with open(str(Path(current_file, "AstrolabAnalysis", "ANALYSIS", "ld_coefficients.p")), "rb") as f:
            print("Using previously computed limb darkening coefficient file...")
            ld_dictionary = pickle.load(f)
    except FileNotFoundError:
        print("Limb darkening coefficients file not found. Proceeding with new computation...")
        ld_dictionary = LimbDarkening(params, current_file, taste_dir).execute_limbdark()
    #
    TransitFit(params, current_file, taste_dir, list_yaml_sectors, ld_dictionary, results_dir).execute()


