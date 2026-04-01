from gammapy.stats.fit_statistics import FitStatistic, wstat
import numpy as np

class WStatVecFitStatistic(FitStatistic):
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

    @classmethod
    def stat_sum_dataset(cls, dataset, args):
        """Statistic function value per bin for some specific parameters."""
        if dataset.counts_off is None and not np.any(dataset.mask_safe.data):
            return 0
        else:
            stat_array = cls.stat_array_dataset(dataset, args)
            if dataset.mask is not None:
                stat_array = stat_array[:, dataset.mask.data]
            return np.sum(stat_array, axis=1)