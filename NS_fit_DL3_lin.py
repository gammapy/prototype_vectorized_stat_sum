from gammapy.datasets import Datasets
from gammapy.modeling.models import GaussianPrior

from gammapy.modeling.sampler import Sampler

datasets = Datasets.read("data/datasets.yaml", filename_models="data/models.yaml")

ref_par = datasets[0].models.parameters["alpha_norm"]
ref_par.frozen = False
for ds in datasets[1:]:
    ds.models[0].spectral_model.model2.alpha_norm = ref_par
    ds.models[0].spectral_model.model1.index2.frozen = True

    ds.models[0].spectral_model.model1.index1.frozen = True
    ds.models[0].spectral_model.model1.ebreak.frozen = True
    ds.models[0].spectral_model.model1.amplitude.frozen = True



free_parameters = datasets.models.parameters.free_unique_parameters
for par in free_parameters:
    par.prior = GaussianPrior(mu=par.value, sigma=0.2*par.value)

sampler_opts = {
    "live_points": 1000,
    "frac_remain": 0.5,
    "log_dir": None,
}

sampler = Sampler(backend="ultranest", sampler_opts=sampler_opts)
result_joint = sampler.run(datasets)

