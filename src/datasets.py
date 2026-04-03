from gammapy.datasets import SpectrumDatasetOnOff, SpectrumDataset, Datasets, FluxPointsDataset
import numpy as np
from gammapy.modeling.models import DatasetModels, FoVBackgroundModel

from gammapy.datasets.flux_points import _get_reference_model

from fit_statistics import Chi2VecFitStatistic
from .fit_statistics import WStatVecFitStatistic
from .evaluator import NPredVecEvaluator


def broadcast_parameters(self, args):
    free_parameters = self.models.parameters.free_unique_parameters
    parameters = self.models.parameters.unique_parameters
    frozen_parameters = set(parameters) - set(free_parameters)
    n_values = len(args[0])
    index_arguments = {}
    for par, arg in zip(free_parameters, args):
        index_arguments[par] = arg * par.unit
    for par in frozen_parameters:
        index_arguments[par] = np.ones(shape=(n_values))*par.quantity
    return index_arguments

Datasets.broadcast_parameters = broadcast_parameters

class VecNPredMixin:
    """
    Mixin class to replace stat_sum evaluation with vectorization.
   
    It must be placed left of the base Dataset class in MRO so its
    models setter and npred_signal/stat_sum take priority.
    """

    @property
    def models(self):
        """Models set on the dataset (`~gammapy.modeling.models.Models`)."""
        return self._models
    
    @models.setter
    def models(self, models):
        """Models setter."""
        self._evaluators = {}
        if models is not None:
            models = DatasetModels(models)
            models = models.select(datasets_names=self.name)
            for model in models:
                if not isinstance(model, FoVBackgroundModel):
                    evaluator = NPredVecEvaluator(
                        model=model,
                        dataset=self
                    )
                    self._evaluators[model.name] = evaluator

        self._models = models

    @property
    def evaluators(self):
        """Model evaluators."""
        return self._evaluators

    def npred_signal(self, args):
        """Model predicted signal counts.

        If a list of model name is passed, predicted counts from these components are returned.
        If stack is set to True, a map of the sum of all the predicted counts is returned.
        If stack is set to False, a map with an additional axis representing the models is returned.

        
        Returns
        -------
        npred_sig : `gammapy.maps.Map`
            Map of the predicted signal counts.
        """
        npred_total = 0

        evaluators = self.evaluators

        for evaluator_name, evaluator in evaluators.items():
            npred = evaluator.compute_npred(args)
            npred_total += npred
            
        return npred_total.T

    def _stat_sum_likelihood(self, args):
        return WStatVecFitStatistic.stat_sum_dataset(self, args)


class VecFluxPointsMixin:
    """
    Mixin class to replace stat_sum evaluation with vectorization.

    It must be placed left of the base Dataset class in MRO so its
    models setter and npred_signal/stat_sum take priority.
    """

    def flux_pred(self, args):
        """Compute predicted flux."""
        flux = 0.0
        for model in self.models:
            reference_model = _get_reference_model(model, self._energy_bounds)
            flux += reference_model.evaluate(self.data.energy_ref[:, None], *args)
        return flux[None, None, :, :]

    def _stat_sum_likelihood(self, args):
        """Total statistic at arbitrary parameters without the priors."""
        return Chi2VecFitStatistic.stat_sum_dataset(self, args)

class VecSpectrumDatasetOnOff(VecNPredMixin, SpectrumDatasetOnOff):
    tag = "VecSpectrumDatasetOnOff"

class VecFluxPointsDataset(VecFluxPointsMixin, FluxPointsDataset):
    tag = "VecFluxPointsDataset"