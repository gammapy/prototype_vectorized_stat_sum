from gammapy.datasets import Datasets
from gammapy.modeling.models import (
    Models, GaussianPrior
)

from astropy.table import Table

from gammapy.datasets.flux_points import FluxPointsDataset
from gammapy.estimators import FluxPoints

from gammapy.modeling.sampler import Sampler

models = Models.read('data/models.yaml')

datasets = []
for idx, model in enumerate(models):
    data = FluxPoints.from_table(Table.read(f"data/fp_dataset_{idx}.ecsv"))
    dataset = FluxPointsDataset(data=data, models=model, name=model.datasets_names[0])
    datasets.append(dataset)

datasets = Datasets(datasets)

ref_par = models.parameters["alpha_norm"]
ref_par.frozen = False
for mod in models[1:]:
    mod.spectral_model.model2.alpha_norm = ref_par
    mod.spectral_model.model1.index2.frozen = True

    mod.spectral_model.model1.index1.frozen = True
    mod.spectral_model.model1.ebreak.frozen = True
    mod.spectral_model.model1.amplitude.frozen = True

free_parameters = models.parameters.free_unique_parameters
for par in free_parameters:
    par.prior = GaussianPrior(mu=par.value, sigma=0.2*par.value)

sampler_opts = {
    "live_points": 100,
    "frac_remain": 0.1,
    "log_dir": None,
    "vectorized": False,
}

sampler = Sampler(backend="ultranest", sampler_opts=sampler_opts)
result_joint = sampler.run(datasets)
