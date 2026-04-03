import pytest
import numpy as np
import astropy.units as u
from astropy.table import Table
from gammapy.modeling import Sampler
from gammapy.modeling.models import Models, GaussianPrior
from numpy.testing import assert_allclose

from sampler import VecSampler
from src.datasets import Datasets, VecSpectrumDatasetOnOff, VecFluxPointsDataset
from gammapy.modeling.tests.test_fit import MyDataset
from gammapy.datasets import DATASET_REGISTRY, FluxPointsDataset
from gammapy.estimators import FluxPoints

from pathlib import Path


DATA_PATH = Path("../data")


@pytest.fixture(scope="session")
def datasets():
    return Datasets([MyDataset(name="test-1"), MyDataset(name="test-2")])


def test_broadcast_parameters(datasets):
    datasets.models[0].x.frozen = True
    datasets.models[1].y.frozen = True
    datasets.models[0].z = datasets.models[1].z
    args = np.array([
        [0, 0.5],
        [1, 1.5],
        [2, 2.5],
    ])

    index_pars = datasets.broadcast_parameters(args)

    assert len(index_pars) == len(datasets.parameters.unique_parameters)

    for p in datasets.parameters:
        assert p in index_pars
        if p.frozen:
            assert np.all(index_pars[p] == index_pars[p][0])


def test_VecSpectrumDatasetOnOff():
    DATASET_REGISTRY.append(VecSpectrumDatasetOnOff)
    datasets_lin = Datasets.read(DATA_PATH / "datasets.yaml", filename_models=DATA_PATH / "models.yaml")
    datasets_vec = Datasets.read(DATA_PATH / "datasets_vec.yaml", filename_models=DATA_PATH / "models.yaml")

    par_ref = datasets_lin.parameters.free_unique_parameters.value
    args = np.random.normal(par_ref, par_ref*0.1, (3, len(par_ref)))

    ll_lin = []
    for arg in args:
        datasets_lin.parameters.free_unique_parameters.value = arg
        ll_lin.append(datasets_lin._stat_sum_likelihood())

    index_arguments = datasets_vec.broadcast_parameters(args.T)
    ll_vec = np.zeros(len(args))
    for ds in datasets_vec:
        eval_args = [index_arguments[par] for par in ds.models.parameters]
        ll_vec += ds._stat_sum_likelihood(eval_args)

    assert_allclose(ll_lin, ll_vec)


    ds_lin, ds_vec = datasets_lin[0], datasets_vec[0]
    npred_lin = []
    for arg in args:
        datasets_lin.parameters.free_unique_parameters.value = arg
        npred = ds_lin.npred_signal()
        npred_lin.append(npred)
    npred_lin = np.array(npred_lin)

    eval_args = [index_arguments[par] for par in ds_vec.models.parameters]
    npred_vec = ds_vec.npred_signal(eval_args)

    assert_allclose(npred_lin, npred_vec)


def test_VecFluxPointsDataset():
    models = Models.read(DATA_PATH / 'models.yaml')

    datasets_lin = []
    datasets_vec = []
    for idx, model in enumerate(models):
        data = FluxPoints.from_table(Table.read(DATA_PATH / f"fp_dataset_{idx}.ecsv"))
        ds_lin = FluxPointsDataset(data=data, models=model, name=model.datasets_names[0])
        ds_vec = VecFluxPointsDataset(data=data, models=model, name=model.datasets_names[0])
        datasets_lin.append(ds_lin)
        datasets_vec.append(ds_vec)

    datasets_lin = Datasets(datasets_lin)
    datasets_vec = Datasets(datasets_vec)

    par_ref = datasets_lin.parameters.free_unique_parameters.value
    args = np.random.normal(par_ref, par_ref*0.1, (3, len(par_ref)))

    ll_lin = []
    for arg in args:
        datasets_lin.parameters.free_unique_parameters.value = arg
        ll_lin.append(datasets_lin._stat_sum_likelihood())

    index_arguments = datasets_vec.broadcast_parameters(args.T)
    ll_vec = np.zeros(len(args))
    for ds in datasets_vec:
        eval_args = [index_arguments[par] for par in ds.models.parameters]
        ll_vec += ds._stat_sum_likelihood(eval_args)

    assert_allclose(ll_lin, ll_vec)


    ds_lin, ds_vec = datasets_lin[0], datasets_vec[0]
    fpred_lin = []
    for arg in args:
        datasets_lin.parameters.free_unique_parameters.value = arg
        fpred = ds_lin.flux_pred()
        fpred_lin.append(fpred)
    fpred_lin = u.Quantity(fpred_lin).T

    eval_args = [index_arguments[par] for par in ds_vec.models.parameters]
    fpred_vec = ds_vec.flux_pred(eval_args)

    assert_allclose(fpred_lin, fpred_vec)