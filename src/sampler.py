from gammapy.modeling.sampler import Sampler, SamplerResult
from gammapy.modeling.utils import _parse_datasets

import astropy.units as u
import numpy as np

class VecSamplerMixin:
    def sampler_ultranest(self, parameters, like):
        """
        Defines the Ultranest sampler and options.

        Returns the result in the SamplerResult that contains the updated models, samples, posterior distribution and
        other information.

        Parameters
        ----------
        parameters : `~gammapy.modeling.Parameters`
            The models parameters to sample.
        like : `~gammapy.modeling.sampler.SamplerLikelihood`
            The likelihood function.

        Returns
        -------
        result : `~gammapy.modeling.sampler.SamplerResult`
            The sampler results.
        """
        import ultranest

        def _prior_inverse_cdf(values):
            """Returns a list of model parameters for a given list of values (that are bound in [0,1])."""
            if None in parameters.prior:
                raise ValueError(
                    "Some parameters have no prior set. You need priors on all parameters."
                )
            cube = [par.prior._inverse_cdf(arg) for par, arg in zip(parameters, values.T)]
            return u.Quantity(cube).T

        self._sampler = ultranest.ReactiveNestedSampler(
            parameters.names,
            like,
            transform=_prior_inverse_cdf,
            log_dir=self.sampler_opts["log_dir"],
            resume=self.sampler_opts["resume"],
            vectorized=True,
        )

        if self.sampler_opts["step_sampler"]:
            from ultranest.stepsampler import (
                SliceSampler,
                generate_mixture_random_direction,
            )

            self._sampler.stepsampler = SliceSampler(
                nsteps=self.sampler_opts["nsteps"],
                generate_direction=generate_mixture_random_direction,
            )

        result = self._sampler.run(
            min_num_live_points=self.sampler_opts["live_points"],
            frac_remain=self.sampler_opts["frac_remain"],
            **self.run_opts,
        )

        return result

    def run(self, datasets):
        """
        Run the sampler on the provided datasets.

        Parameters
        ----------
        datasets : `~gammapy.datasets.Datasets`
            Datasets to fit.

        Returns
        -------
        result : `~gammapy.modeling.sampler.SamplerResult`
            The sampler results. See the class description to get the exact content.
        """
        datasets, parameters = _parse_datasets(datasets=datasets)
        parameters = parameters.free_unique_parameters

        if self.backend == "ultranest":

            def like(args):
                index_arguments = datasets.broadcast_parameters(args.T)
                ll = np.zeros(len(args))
                for ds in datasets:
                    eval_args = [index_arguments[par] for par in ds.models.parameters]
                    ll += ds._stat_sum_likelihood(eval_args)
                return -0.5 * ll

            result_dict = self.sampler_ultranest(parameters, like)
            self._sampler.print_results()

            models_copy = datasets.models.copy()
            self._update_models_from_posterior(models_copy, result_dict)

            result = SamplerResult.from_ultranest(result_dict)
            result.models = models_copy

            return result
        else:
            raise ValueError(f"Sampler {self.backend} is not supported.")


class VecSampler(VecSamplerMixin, Sampler):
    tag = "VecSampler"