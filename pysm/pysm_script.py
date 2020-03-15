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

nside = 1024
galactic_mask = np.load("./masks/mask_galactic_1024.npz")["mask"]
survey_mask = np.load("./masks/mask_survey_1024.npz")["mask"]
mask = galactic_mask * survey_mask
plt.figure(figsize=(12, 4))
hp.mollview(galactic_mask, title="Galactic", sub=(1, 2, 1))
hp.mollview(survey_mask, title="Survey", sub=(1, 2, 2))

from pspy import so_map
survey = so_map.healpix_template(ncomp=1, nside=nside)
survey.data = mask

from pspy import so_window
survey = so_window.create_apodization(survey, apo_type="C1", apo_radius_degree=5)
hp.mollview(survey.data, title=None)

sky = pysm.Sky(nside=nside, preset_strings=["d1"], output_unit=u.uK_CMB)
map_100GHz = sky.get_emission(100 * u.GHz)
plt.figure(figsize=(18, 4))
hp.mollview(survey.data * map_100GHz[0], min=0, max=100, title="I map", sub=(1, 3, 1))
hp.mollview(survey.data * map_100GHz[1], min=0, max=10, title="Q map", sub=(1, 3, 2))
hp.mollview(survey.data * map_100GHz[2], min=0, max=10, title="U map", sub=(1, 3, 3))

import os
output_dir = "/tmp/pysm"
os.makedirs(output_dir, exist_ok=True)
binning_file = os.path.join(output_dir, "binning.dat")
from pspy import pspy_utils
pspy_utils.create_binning_file(bin_size=40, n_bins=100, file_name=binning_file)

window = (survey, survey)
lmax = 3 * nside - 1
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
frequencies = [93, 145, 225] * u.GHz

store_map = False
store_alms = False
store_spectra = True

models = {k: {} for k in presets}

from itertools import product
for preset, freq in product(presets, frequencies):
  print("Computing {} model @ {}".format(preset, freq))

  # Get emission map
  sky = pysm.Sky(nside=nside, preset_strings=[preset], output_unit=u.uK_CMB)
  emission = sky.get_emission(freq)

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
                                   mbb_inv=mcm,
                                   spectra=spectra)
  models[preset][freq] = {"spectra": db}

  if store_map:
    models[preset][freq].update({"map": emission})
  if store_alms:
    models[preset][freq].update({"alms": alms})

print(models["d0"])

spec = "TT"
nfreq = len(frequencies)
fig, axes = plt.subplots(4, nfreq, sharex=True, sharey="row", figsize=(15, 12))
if spec in ["TT", "EE", "BB"]:
  [ax.set_yscale("log") for ax in axes.flatten()]

def plot_spectra(submodels, row_number):
  for i, (model, freq) in enumerate(product(submodels, frequencies)):
    lb, db = models[model][freq].get("spectra")
    axes[row_number, i%3].plot(lb, db[spec], label=model)

# Show CMB
for ax in axes.flatten():
  ax.plot(ell_camb, dl_dict[spec.lower()], "gray", label="CMB")
  ax.set_xlim(0, lmax)

names = {"dust": dust, "synchrotron": synchrotron, "AME": ame, "free-free": free_free}
for i, submodel in enumerate(names.values()):
  plot_spectra(submodel, i)

for ax, name in zip(axes[:, -1], names.keys()):
    leg = ax.legend(title="{} - {}".format(spec, name), bbox_to_anchor=(1, 1), loc="upper left")
    leg._legend_box.align = "left"

for ax in axes[:, 0]:
  ax.set_ylabel(r"$D_\ell$")
for ax in axes[-1]:
  ax.set_xlabel(r"$\ell$")

for ax, freq in zip(axes[0], frequencies):
  ax.set_title("{}".format(freq))

frequency = 93
I, Q, U = 0, 1, 2
select = I

ncols = len(frequencies)
nrows = len(presets)

fig = plt.figure(figsize=(18, 5*nrows), num=0)
# fig.suptitle("coucou")
i = 1
for model, preset in models.items():
    for freq in frequencies:
        hp.mollview(preset[freq].get("map")[select], min=0, max=100, sub=(nrows, ncols, i), title=None, cbar=False)
        i += 1
