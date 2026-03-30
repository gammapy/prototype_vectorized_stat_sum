from gammapy.datasets import SpectrumDatasetOnOff, SpectrumDataset
from gammapy.maps import Map
from gammapy.modeling.models import DatasetModels, FoVBackgroundModel
from .evaluator import NPredEvaluator

class VectorizedMixin:
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
                    evaluator = NPredEvaluator(
                        model=model,
                        dataset=self
                    )
                    self._evaluators[model.name] = evaluator

        self._models = models

    @property
    def evaluators(self):
        """Model evaluators."""
        return self._evaluators

    def npred_signal(self):
        """Model predicted signal counts.

        If a list of model name is passed, predicted counts from these components are returned.
        If stack is set to True, a map of the sum of all the predicted counts is returned.
        If stack is set to False, a map with an additional axis representing the models is returned.

        
        Returns
        -------
        npred_sig : `gammapy.maps.Map`
            Map of the predicted signal counts.
        """
        npred_total = Map.from_geom(self._geom, dtype=float)

        evaluators = self.evaluators

        for evaluator_name, evaluator in evaluators.items():
            npred = evaluator.compute_npred()
            npred_total.stack(npred)
            
        return npred_total


class VecSpectrumDataset(VectorizedMixin, SpectrumDataset):
    tag = "VecSpectrumDataset"

class VecSpectrumDatasetOnOff(VectorizedMixin, SpectrumDatasetOnOff):
    tag = "VecSpectrumDatasetOnOff"