import pickle
import yaml
from astropy.io import fits
from pathlib import Path


class TESS_sector:
    """
    Class to read the previous TESS analysis results and get the sector number, time, flux and flux errors
    """
    def __init__(self, current_file, yaml_sector_name):
        self.current_file = current_file
        self.yaml_sector_name = yaml_sector_name
        self.get_lc_name()
        self.get_sector_number()
        self.read_selected_results()

    def get_lc_name(self):
        with open(str(Path(self.current_file, "AstrolabAnalysis", "TESS", self.yaml_sector_name))) as in_f:
            yaml_sector = yaml.load(in_f, Loader=yaml.FullLoader)
            self.file_lc_name = yaml_sector["lc_name"]

    def get_sector_number(self):
        lchdu = fits.open(str(Path(self.current_file, "Data", self.file_lc_name)))
        self.sector_number = lchdu[0].header['SECTOR']

    def read_selected_results(self):
        name_selected = "GJ3470_TESS_sector" + str(self.sector_number) + "_selected.p"
        tess_sector_dict = pickle.load(open(str(Path(self.current_file, "Results", name_selected)), "rb"))
        self.tess_bjd_tdb = tess_sector_dict["time"]
        self.tess_normalized_flux = tess_sector_dict["selected_flux"]
        self.tess_normalized_flux_error = tess_sector_dict["selected_flux_error"]


class TASTE_reader:
    """
    Class to read the previous TASTE results
    """
    def __init__(self, taste_dir):
        self.taste_dir = taste_dir

    def read_taste_results(self, ind_list):
        with open(str(Path(self.taste_dir, "aperture_final.pkl")), "rb") as f:
            julian_date = pickle.load(f)
            differential_ref01_normalized = pickle.load(f)
            differential_ref01_normalized_error = pickle.load(f)
            differential_ref02_normalized = pickle.load(f)
            differential_ref02_normalized_error = pickle.load(f)
            differential_allref_normalized = pickle.load(f)
            differential_allref_normalized_error = pickle.load(f)
            differential_ref01 = pickle.load(f)
            differential_ref01_error = pickle.load(f)
            differential_ref02 = pickle.load(f)
            differential_ref02_error = pickle.load(f)
            differential_allref = pickle.load(f)
            differential_allref_error = pickle.load(f)
        return julian_date, differential_allref, differential_allref_error
