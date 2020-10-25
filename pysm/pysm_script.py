import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import pspy
import pysm
import pysm.units as u
print(" healpy :", hp.__version__)
print("  numpy :", np.__version__)
print("   pysm :", pysm.__version__)
print("   pspy :", pspy.__version__)

nside = 4096

print("Loading masks for nside =", nside)
galactic_mask = np.load("./masks/mask_galactic_1024.npz")["mask"]
survey_mask = np.load("./masks/mask_survey_1024.npz")["mask"]
mask = galactic_mask * survey_mask

mask = hp.ud_grade(mask, nside_out=nside)
assert np.all((mask == 1) | (mask == 0)), "Mask with values others than 0 or 1"

from pspy import so_map
survey = so_map.healpix_template(ncomp=1, nside=nside)
survey.data = mask

from pspy import so_window
survey = so_window.create_apodization(survey,
                                      apo_type="C1",
                                      apo_radius_degree=5)

print("Generating binning file")
import os
output_dir = "/tmp/pysm"
os.makedirs(output_dir, exist_ok=True)
binning_file = os.path.join(output_dir, "binning.dat")
from pspy import pspy_utils
pspy_utils.create_binning_file(bin_size=40,
                               n_bins=1000,
                               file_name=binning_file)

print("Computing MCM")
window = (survey, survey)
niter = 3
from pspy import so_mcm
mbb_inv, bbl = so_mcm.mcm_and_bbl_spin0and2(window,
                                            binning_file,
                                            lmax=lmax,
                                            type="Dl",
                                            niter=niter)

dust = ["d0", "d1", "d2", "d4", "d6"]  # "d5"
synchrotron = ["s1", "s2", "s3"]
ame = ["a1", "a2"]
free_free = ["f1"]

presets = dust + synchrotron + ame + free_free
frequencies = [93, 145, 225]

store_map = False
store_alms = False
store_spectra = True

models = {k: {} for k in presets}

from itertools import product
for preset, freq in product(presets, frequencies):
    print("Computing {} model @ {} GHz".format(preset, freq))

    # Get emission map
    sky = pysm.Sky(nside=nside, preset_strings=[preset], output_unit=u.uK_CMB)
    emission = sky.get_emission(freq * u.GHz)

    # Compute alm
    from pspy import sph_tools
    tmpl = so_map.healpix_template(ncomp=3, nside=nside)
    tmpl.data = emission.value
    alms = sph_tools.get_alms(tmpl, window, niter=niter, lmax=lmax)

    # Compute spectra
    from pspy import so_spectra
    spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
    db = so_spectra.bin_spectra(*so_spectra.get_spectra(alms, spectra=spectra),
                                binning_file,
                                lmax=lmax,
                                type="Dl",
                                mbb_inv=mbb_inv,
                                spectra=spectra)
    models[preset][freq] = {"spectra": db}

    if store_map:
        models[preset][freq].update({"map": emission})
    if store_alms:
        models[preset][freq].update({"alms": alms})

import pickle
pickle.dump(models, open("./models_{}.pkl".format(nside), "wb"))
