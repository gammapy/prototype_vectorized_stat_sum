import numpy as np
import astropy.units as u

from gammapy.datasets import Datasets
from gammapy.estimators import FluxPointsEstimator

datasets = Datasets.read("data/datasets.yaml", filename_models="data/models.yaml")

energy_bounds = (0.2, 20) * u.TeV
energy_edges = np.geomspace(*energy_bounds, 10)

for idx, (ds, mod) in enumerate(zip(datasets, datasets.models)):
    ds.models = mod
    fpe = FluxPointsEstimator(energy_edges=energy_edges, source=mod.name)
    flux_points = fpe.run(datasets=[ds])
    flux_points.write(f'data/fp_dataset_{idx}.ecsv', sed_type='dnde', overwrite=True)