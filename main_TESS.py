
from pathlib import Path
import yaml
import os

from AstrolabAnalysis.TESS.Modules.LighCurvePreprocess import LightCurvePreprocess

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

    data_folder = str(Path(current_file, "Data"))
    results_folder = str(Path(current_file, "Results"))
    os.makedirs(results_folder, exist_ok=True)

    for yaml_sector_name in list_yaml_sectors["yaml_list"]:
        with open(str(Path(current_file, "AstrolabAnalysis", "TESS", yaml_sector_name))) as in_f:
            yaml_sector = yaml.load(in_f, Loader=yaml.FullLoader)
        time_array, sap_flux, sap_flux_error, pdcsap_flux, pdcsap_flux_error = LightCurvePreprocess(
                yaml_sector["lc_name"], data_folder, results_folder,
                yaml_sector["preprocess"]).execute_preprocess()
