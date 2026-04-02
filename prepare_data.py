
"""
PKS 2155-304 Joint Spectral Fit with EBL Absorption
=====================================================
Based on the Gammapy flare analysis tutorial.

Each observation yields one SpectrumDataset. A BrokenPowerLaw × EBL
absorption model is assigned to every dataset. The EBL alpha (opacity
scale) parameter is *shared* across all models before a joint fit.

Requirements
------------
    pip install gammapy click  # gammapy >= 1.2

Data
----
    export GAMMAPY_DATA=/path/to/gammapy-data
    The script uses the HESS DL3 DR1 public dataset (PKS 2155-304 observations).

Usage
-----
    python pks2155_ebl_joint_fit.py --help
    python pks2155_ebl_joint_fit.py                          # all defaults
    python pks2155_ebl_joint_fit.py --config my_config.yaml
    python pks2155_ebl_joint_fit.py --outdir results/ --ebl-reference finke -v
"""

import logging
from pathlib import Path

import astropy.units as u
import click

from gammapy.analysis import Analysis, AnalysisConfig
from gammapy.modeling import Fit
from gammapy.modeling.models import (
    BrokenPowerLawSpectralModel,
    EBLAbsorptionNormSpectralModel,
    SkyModel,
    Models,
)

log = logging.getLogger(__name__)


def build_datasets(config_path: str) -> Analysis:
    """Run the standard Gammapy high-level pipeline up to dataset creation."""
    config = AnalysisConfig.read(config_path)

    analysis = Analysis(config)
    analysis.get_observations()
    analysis.get_datasets()

    log.info("Created %d dataset(s):", len(analysis.datasets))
    for ds in analysis.datasets:
        log.info("  %s  livetime=%.1f min", ds.name,
                 ds.gti.time_sum.to("min").value)

    return analysis


def build_models(analysis: Analysis,
                 ebl_reference: str,
                 redshift: float) -> Models:
    """
    Assign a BrokenPowerLaw * EBL SkyModel to every dataset.
    """
    models = []
    ebl_ref = EBLAbsorptionNormSpectralModel.read_builtin(ebl_reference, redshift=redshift)
    for ds in analysis.datasets:
        bpl = BrokenPowerLawSpectralModel(
            amplitude=1e-11 * u.Unit("cm-2 s-1 TeV-1"),
            ebreak=1.5 * u.TeV,
            index1=3.0,
            index2=4.5,
        )
        bpl.index2.frozen=True
 
        model = SkyModel(
            spectral_model=bpl * ebl_ref, 
            name=f"pks2155_{ds.name}",
            datasets_names=[ds.name],
        )
        models.append(model)

    return Models(models)


def run_fit(analysis: Analysis, models: Models) -> Fit:
    """Assign models to datasets and run the joint fit."""
    analysis.datasets.models = models
   
    fit = Fit()
    log.info(f"Starting individual fit over {len(analysis.datasets)} dataset(s)" )
    for ds in analysis.datasets:
        result = fit.run(ds)
        print(f"Fit of obs_id {ds.meta_table['OBS_ID'][0]} is {result.success}.")

    log.info(f"  Shared EBL alpha_norm parameter across {len(models)} models.")
    alpha_norm_ref = models[0].parameters["alpha_norm"]
    alpha_norm_ref.min = 0.5
    alpha_norm_ref.max = 2.
    alpha_norm_ref.frozen = False
    
    log.info("Starting joint fit over %d dataset(s) …", len(analysis.datasets))
 
    result = fit.run(analysis.datasets)

    if result.success:
        log.info("Fit converged  (total stat = %.2f)", result.total_stat)
    else:
        log.warning("Fit did NOT converge – inspect result carefully.")

    return result


def export_results(analysis: Analysis, outdir: Path) -> None:
    """Write datasets (with models) and a standalone models YAML file."""
    outdir.mkdir(parents=True, exist_ok=True)

    datasets_path = outdir / "datasets.yaml"
    models_path   = outdir / "models.yaml"

    # Write datasets (includes the link to models)
    analysis.datasets.write(datasets_path, overwrite=True)

    # Write models separately (standalone, portable)
    analysis.datasets.models.write(models_path, overwrite=True)

    log.info("Datasets written → %s", datasets_path)
    log.info("Models   written → %s", models_path)



EBL_CHOICES = click.Choice(["dominguez", "franceschini", "finke"], case_sensitive=False)


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--config", "-c",
    default="config/config.yaml",
    show_default=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Path to the YAML analysis config.",
)
@click.option(
    "--ebl-reference", "-e",
    default="dominguez",
    show_default=True,
    type=EBL_CHOICES,
    help="EBL absorption table to use.",
)
@click.option(
    "--source-redshift", "-z",
    default=0.116,
    show_default=True,
    type=click.FloatRange(min=0.0),
    help="Source redshift (default is PKS 2155-304).",
)
def main(
    config: Path,
    ebl_reference: str,
    source_redshift: float,
):
    """PKS 2155-304 joint spectral fit with EBL absorption.

    Builds one SpectrumDataset per observation (from the HESS DL3 DR1
    public dataset), assigns a BrokenPowerLaw × EBL SkyModel to each,
    shares the EBL alpha_norm parameter across all models, runs a joint
    fit, and writes datasets.yaml + models.yaml to the output directory.
    """
    if not config.exists():
        raise click.BadParameter(
            f"Config file not found: {config}",
            param_hint="'--config'",
        )

    # 1. Build datasets ---------------------------------------------------
    log.info("Step 1 – Building spectral datasets from config …")
    config = AnalysisConfig.read(config)
    analysis = Analysis(config)
    analysis.get_observations()
    analysis.get_datasets()  

    effective_outdir = Path(analysis.config.general.outdir)

    log.info("Step 2 – Defining spectral models …")
    models = build_models(analysis, ebl_reference=ebl_reference, redshift=source_redshift)

    # 3. Joint fit --------------------------------------------------------
    log.info("Step 3 – Running joint fit …")
    results = run_fit(analysis, models)

    print(results)
    print(results.models.to_parameters_table())

    # 4. Export -----------------------------------------------------------
    log.info("Step 4 – Exporting results to %s …", effective_outdir)
    export_results(analysis, effective_outdir)


if __name__ == "__main__":
    main()