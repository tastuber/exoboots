# Exoboots

A package to model the signal of hot exozodiacal dust in optical
interferometric data using bootstrapping.

## Short description

### Highlights:
* Model hot exozodiacal dust in interferometric data.
* Analyze visibility amplitudes or squared visibilities loaded from an OIFITS
file.
* Account for correlation among spectral channels using Bootstrapping.
* Explore different analytic models and combinations thereof: Uniform disk,
Limb-darkend disk, Uniform emission over the field-of-view, 2d-Gaussian, ring,
off-axis point source.
* Automatic visualization and saving of results.

With this package, you can model the interferometric observables visibility
amplitude or squared visibility that are loaded from an OIFITS file
([Duvert et al.2017](https://ui.adsabs.harvard.edu/abs/2017A%26A...597A...8D/abstract)),
the standard in optical interferometry.

Interferometric data is typically strongly correlated, especially among
spectral channels. To account for this, `Exoboots` can explore different types
of correlations, for instance by assuming data belonging to one interferometric
baseline (one pair of telescopes) to be fully correlated. Exoboots capabilities
are inspired by an IDL package that was developed to analyze observations with
CHARA/FLUOR
([Absil et al. 2006](https://ui.adsabs.harvard.edu/abs/2006A%26A...452..237A/abstract)).

# Installation instructions

I recommend installing `Exoboots` into a virtual environment, for instance
using `conda`, `mamba`, or `python venv`. `Exoboots` requires Python in version
3.12.8 or later.

### Install directly from GitHub
```
pip install git+https://github.com/tastuber/exoboots
```

### Clone repository, then install

```
git clone https://github.com/tastuber/exoboots
cd exoboots
pip install .
```

### Status

The code is in early development, but its capabilities have already been used
in [Ollmann et al. 2025](https://ui.adsabs.harvard.edu/abs/2025A%26A...699A.144O/abstract).\
If you would like to get involved or obtain more information, please contact me
either via GitHub or [email](mailto:tstuber@arizona.edu).
