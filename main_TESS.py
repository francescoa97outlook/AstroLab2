import yaml
import os
import pickle
from pathlib import Path

from AstrolabAnalysis.TESS.Modules.LighCurvePreprocess import LightCurvePreprocess
from AstrolabAnalysis.TESS.Modules.LightCurveFiltering import LightCurveFiltering

current_file = Path(__file__).parent
os.chdir(current_file)

if __name__ == '__main__':
    """
    Args:
        todo
    Workflow:
        todo
    Return:
    """
    # Read the sector list
    with open(str(Path(current_file, "AstrolabAnalysis", "TESS", "list_yaml_sectors.yaml"))) as in_f:
        list_yaml_sectors = yaml.load(in_f, Loader=yaml.FullLoader)

    with open(str(Path(current_file, "AstrolabAnalysis", "TESS", "orbital_params.yaml"))) as in_f:
        orbital_params = yaml.load(in_f, Loader=yaml.FullLoader)

    data_folder = str(Path(current_file, "Data"))
    results_folder = str(Path(current_file, "Results"))
    os.makedirs(results_folder, exist_ok=True)

    for yaml_sector_name in list_yaml_sectors["yaml_list"]:
        with open(str(Path(current_file, "AstrolabAnalysis", "TESS", yaml_sector_name))) as in_f:
            yaml_sector = yaml.load(in_f, Loader=yaml.FullLoader)
        time_array, sap_flux, sap_flux_error, pdcsap_flux, pdcsap_flux_error, sector_number = LightCurvePreprocess(
                yaml_sector["lc_name"], data_folder, results_folder,
                yaml_sector["preprocess"]).execute_preprocess()
        print("======================")
        print("Sector", sector_number)
        selected_flux, selected_flux_error = LightCurveFiltering(
                time_array, sap_flux, sap_flux_error, pdcsap_flux, pdcsap_flux_error,
                orbital_params, yaml_sector, sector_number, results_folder
                ).execute_flatten()
        sector_dictionary = {
            'time' : time_array,
            'selected_flux' : selected_flux,
            'selected_flux_error' : selected_flux_error,
        }
        pickle.dump(sector_dictionary, open(str(Path(
            results_folder, "GJ3470_TESS_sector{0:d}_selected.p".format(sector_number))), "wb"))
