import numpy as np

class NPredEvaluator:
    def __init__(self, model, dataset):
        self.energy_true = dataset.exposure.geom.axes["energy_true"].center.to_value("TeV")
        self.energy = dataset.counts.geom.axes["energy"] 
        self.matrix = dataset.edisp.get_edisp_kernel().pdf_matrix
        self.exposure = dataset.exposure
        self.model = model.spectral_model

    @staticmethod
    def _apply_edisp(values, matrix):
        vals = np.matmul(np.rollaxis(values, 2,4), matrix)
        return np.rollaxis(vals, -1, 2)

    def evaluate(self, *args):
        values = self.model.evaluate(self.energy_true[:, None], *args)
        values = self.exposure.data.T[...,np.newaxis]*values
        return self._apply_edisp(values, self.matrix)
    
    