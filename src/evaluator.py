import numpy as np
from gammapy.modeling.models import integrate_spectrum

class NPredEvaluator:
    def __init__(self, model, dataset):
        self.energy_true = dataset.exposure.geom.axes["energy_true"]
        self.energy = dataset.counts.geom.axes["energy"] 
        self.matrix = dataset.edisp.get_edisp_kernel().pdf_matrix
        self.exposure = dataset.exposure
        self.model = model.spectral_model

    @staticmethod
    def _apply_edisp(values, matrix):
        vals = np.matmul(np.moveaxis(values, 2,-1), matrix)
        return np.rollaxis(vals, -1, 2)
    

    def compute_integrated_spectrum(self, args):
        """Integrate spectrum over parameters arrays."""
        energy_true = self.energy_true.edges

        values = integrate_spectrum(
                self.model,
                energy_true[:-1],
                energy_true[1:],
                parameter_samples=args,
                energy_flux=False,
            )
        return values

        
    def evaluate(self, args):
        values = self.compute_integrated_spectrum(args)
        values = self.exposure.quantity.T[...,np.newaxis]*values
        return self._apply_edisp(values, self.matrix)

    
    def compute_npred(self, args):
        # kwargs = {par.name: par.quantity for par in self.model.parameters}
        # kwargs = self.model._convert_evaluate_unit(kwargs, self.energy_true)
        # args = list(kwargs.values())

        return self.evaluate(args)
     
    
    