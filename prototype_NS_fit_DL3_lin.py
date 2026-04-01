import numpy as np
import astropy.units as u
from gammapy.datasets import Datasets
from gammapy.modeling.models import GaussianPrior

import ultranest

datasets = Datasets.read("data/datasets.yaml", filename_models="data/models.yaml")

ref_par = datasets[0].models.parameters["alpha_norm"]
ref_par.frozen = False
for ds in datasets[1:]:
    ds.models[0].spectral_model.model2.alpha_norm = ref_par
    ds.models[0].spectral_model.model1.index2.frozen = True

    # ds.models[0].spectral_model.model1.index1.frozen = True
    # ds.models[0].spectral_model.model1.ebreak.frozen = True
    # ds.models[0].spectral_model.model1.amplitude.frozen = True



free_parameters = datasets.models.parameters.free_unique_parameters
parameters = datasets.models.parameters.unique_parameters
frozen_parameters = set(parameters) - set(free_parameters)

for par in free_parameters:
    par.prior = GaussianPrior(mu=par.value, sigma=0.2*par.value)

def prior_transform(args):
    cube = [par.prior._inverse_cdf(arg) for par, arg in zip(free_parameters, args.T)]
    return u.Quantity(cube).T

def broadcast_parameters(args):
    n_values = len(args[0])
    index_arguments = {}
    for par, arg in zip(free_parameters, args):
        index_arguments[par] = arg * par.unit
    for par in frozen_parameters:
        index_arguments[par] = np.ones(shape=(n_values))*par.quantity
    return index_arguments

def likelihood(args):
    datasets.models.parameters.free_unique_parameters.value = args
    ll = 0
    for ds in datasets:
        ll += ds._stat_sum_likelihood()
    return -0.5 * ll



sampler = ultranest.ReactiveNestedSampler(
    free_parameters.names,
    likelihood,
    transform=prior_transform,
    vectorized=False,
)

result = sampler.run(
    min_num_live_points=1000,
    frac_remain=0.5,
)
sampler.print_results()
