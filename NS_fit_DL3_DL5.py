from astropy.table import Table

from gammapy.modeling.models import (
    Models, GaussianPrior
)
from gammapy.estimators import FluxPoints
from gammapy.datasets import DATASET_REGISTRY

from src.datasets import Datasets, VecSpectrumDatasetOnOff, VecFluxPointsDataset
from src.sampler import VecSampler


DATASET_REGISTRY.append(VecSpectrumDatasetOnOff)
datasets_DL3 = Datasets.read("data/datasets_vec.yaml", filename_models="data/models.yaml")

models = Models.read('data/models.yaml')
datasets_DL5 = []
for idx, model in enumerate(models):
    data = FluxPoints.from_table(Table.read(f"data/fp_dataset_{idx}.ecsv"))
    ds = VecFluxPointsDataset(data=data, models=model, name=model.datasets_names[0])
    datasets_DL5.append(ds)

datasets = Datasets(datasets_DL3[:3] + datasets_DL5[3:])

ref_par = datasets[0].models.parameters["alpha_norm"]
ref_par.frozen = False
for ds in datasets:
    ds.models[0].spectral_model.model2.alpha_norm = ref_par
    ds.models[0].spectral_model.model1.index2.frozen = True
    ds.models[0].spectral_model.model1.ebreak.frozen = True
    ds.models[0].spectral_model.model1.amplitude.frozen = True

free_parameters = datasets.models.parameters.free_unique_parameters
for par in free_parameters:
    par.prior = GaussianPrior(mu=par.value, sigma=0.2 * par.value)

sampler_vec_opts = {
    "live_points": 100,
    "frac_remain": 0.1,
    "log_dir": None,
    "vectorized": True,
}

sampler_vec = VecSampler(backend="ultranest", sampler_opts=sampler_vec_opts)
result_vec = sampler_vec.run(datasets)
