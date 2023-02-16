#!/usr/bin/env python
"""
    mips_misa_gb_bistatic.py

    $Id$

    This is a set of simulations for mapping the bi-static radar performance of
    Millstone MISA to the NRAO Greenbank 140 foot.

"""

from pathlib import Path
from mips.isr_mapper import map_radar_array
from mips.isr_plotting import isr_map_plot
from copy import copy

# from wff_radars import *


def main():

    try:
        from dask.distributed import Client

        client = Client()
        print("Dask avalible client info: " + str(client))
    except:
        print("Dask not avalible, default to single core.")
        client = None

    # set up directories
    savedir = Path("misa_gb_bistatic").absolute()
    savedir.mkdir(exist_ok=True)
    figdir = savedir.joinpath("figs")
    figdir.mkdir(exist_ok=True)
    datadir = savedir.joinpath("data")
    datadir.mkdir(exist_ok=True)

    # These are all consistant parameters
    map_parameters = ["speed", "dNe", "dTi", "dTe", "dV", "gamma"]
    dval_max = [3600.0, 0.1, 20.0, 20.0, 30.0, 100.0]

    # radar concepts
    tx_names = ["Millstone UHF"]
    tx_system_list = ["millstone_misa"]
    rx_names = ["Greenbank 140ft"]
    rx_system_list = ["greenbank_43m"]

    tx_site_list = ["millstone"]
    rx_site_list = ["greenbank"]

    plot_extent = {
        "center_lat": 37.93433,
        "center_lon": -75.47057,
        "delta_lat": 30.0,
        "delta_lon": 35.0,
    }

    ionosphere_eregion = {
        "name": "E-region",
        "use_iri": False,
        "iri_type": "local",
        "iri_time": "fixed parameters",
        "alt_m": 100e3,
        "N_e": 1e11,
        "T_e": 300.0,
        "T_i": 300.0,
    }
    # mode_eregion = dict(n_bauds=16, pulse_length_s=480E-6, ipp=8190E-6)
    mode_eregion = dict(n_bauds=16, tx_pulse_length=480000, ipp=8190000)
    ionosphere_fregion = {
        "name": "F-region",
        "use_iri": False,
        "iri_type": "local",
        "iri_time": "fixed parameters",
        "alt_m": 300e3,
        "N_e": 5e11,
        "T_e": 2000.0,
        "T_i": 1200.0,
    }
    # mode_fregion = dict(n_bauds=1, pulse_length_s=480E-6, ipp=8910E-6)
    mode_fregion = dict(n_bauds=1, tx_pulse_length=480000, ipp=8910000)
    ionosphere_topside = {
        "name": "topside",
        "use_iri": False,
        "iri_type": "local",
        "iri_time": "fixed parameters",
        "alt_m": 800e3,
        "N_e": 5e10,
        "T_e": 2700.0,
        "T_i": 2000.0,
    }
    # mode_topside = dict(n_bauds=1, pulse_length_s=2000E-3, ipp=34600E-6)
    mode_topside = dict(n_bauds=1, tx_pulse_length=2000000, ipp=34600000)

    sim_default = dict(
        tx_sites=tx_site_list,
        tx_radars=tx_system_list,
        rx_sites=rx_site_list,
        rx_radars=rx_system_list,
        pair_list="cross",
        plasma_parameter_errors=True,
        ngrid=[100, 150],
        extent=plot_extent,
        mtime_estimate_method="std",
        mpclient=client,
        pfunc=print,
    )

    ionosphere_list = [ionosphere_eregion, ionosphere_fregion, ionosphere_topside]
    mode_list = [mode_eregion, mode_fregion, mode_topside]

    for iidx, (iono, imode) in enumerate(zip(ionosphere_list, mode_list)):
        isim = copy(sim_default)
        isim["tname"] = "Millstone-Greenbank UHF bistatic " + "(" + iono["name"] + ")"
        isim["ionosphere"] = iono
        isim.update(imode)
        print("mapping " + iono["name"])

        sfname = "millstone_greenbank_bistatic" "_" + iono["name"]
        # Execute mapping, record data, and output plot as png
        ds_WFF = map_radar_array(**isim)

        ds_WFF.to_netcdf(
            datadir.joinpath(sfname + ".nc"), engine="h5netcdf", invalid_netcdf=True
        )

        isr_map_plot(
            ds_WFF,
            map_parameters=map_parameters,
            dval_max=dval_max,
            map_zoom=0.4,
            range_contours=[0],
            map_fname=figdir.joinpath(sfname + ".png"),
            map_type="normal",
            annotate=True,
            legend=True,
            vmin=0.1,
            vmax=-1,
            extent=plot_extent,
        )


if __name__ == "__main__":
    main()
