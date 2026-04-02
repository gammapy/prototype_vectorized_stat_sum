from gammapy.stats.fit_statistics import FitStatistic, wstat
import numpy as np


class VecFitStatisticMixin:
    @classmethod
    def stat_sum_dataset(cls, dataset, args):
        """Statistic function value per bin for some specific parameters."""
        stat_array = cls.stat_array_dataset(dataset, args)
        if dataset.mask is not None:
            stat_array = stat_array[:, dataset.mask.data]
        return np.sum(stat_array, axis=1)


class WStatVecFitStatistic(VecFitStatisticMixin, FitStatistic):
    """Vectorized WStat fit statistic class for ON-OFF Poisson measurements."""

    @classmethod
    def stat_array_dataset(cls, dataset, *args):
        """Statistic function value per bin given for some specific parameters."""
        counts, counts_off, alpha = (
            dataset.counts.data,
            dataset.counts_off.data,
            dataset.alpha.data,
        )
        npred_signal = dataset.npred_signal(*args)
        on_stat_ = wstat(
            n_on=counts,
            n_off=counts_off,
            alpha=alpha,
            mu_sig=npred_signal.T,
        )
        return np.nan_to_num(on_stat_)


class Chi2VecFitStatistic(VecFitStatisticMixin, FitStatistic):
    """Chi2 fit statistic class for measurements with gaussian symmetric errors."""

    @classmethod
    def stat_array_dataset(cls, dataset, args):
        """Statistic function value per bin for specific model parameters."""
        model = dataset.flux_pred(args)
        data = dataset.data.dnde.quantity
        try:
            sigma = dataset.data.dnde_err.quantity
        except AttributeError:
            sigma = (dataset.data.dnde_errn + dataset.data.dnde_errp).quantity / 2
        stat_array = ((data[:, :, :, None] - model) / sigma[:, :, :, None]).to_value("") ** 2
        return np.moveaxis(stat_array, -1, 0)