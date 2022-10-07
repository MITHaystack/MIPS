#!/usr/bin/env python
"""
    mips_misa_mapping.py

    This is a set of simulations for mapping the spatial performance of
    with the MIPS model, targeted to the Millstone site and radar.

"""

from pathlib import Path
from mips import map_radar_array
from mips.isr_plotting import isr_map_plot
from copy import copy


def main():


    try:
        from dask.distributed import Client

        client = Client()
        print("Dask avalible client info: " + str(client))
    except:
        print("Dask not avalible, default to single core.")
        client = None
    # set up directories
    savedir = Path("mapping").absolute()
    savedir.mkdir(exist_ok=True)
    figdir = savedir.joinpath("figs")
    figdir.mkdir(exist_ok=True)
    datadir = savedir.joinpath("data")
    datadir.mkdir(exist_ok=True)

    # These are all consistant parameters
    map_parameters = ["speed", "dNe", "dTi", "dTe", "dV"]
    dval_max = [10000.0, 0.1, 20.0, 20.0, 30.0]

    # radar concepts
    names = ["Millstone UHF"]
    system_list = ["millstone_misa"]
    site_list = ["millstone"]

    plot_extent = {
        "center_lat": 42.6195,
        "center_lon": -71.49173,
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
    mode_eregion = dict(n_bauds=13,
                        tx_baud_length= 30e-6,
                        ipp=.0065)

    ionosphere_fregion = {
        "name": "F-region",
        "pulse_length": 480e-6,
        "use_iri": False,
        "iri_type": "local",
        "iri_time": "fixed parameters",
        "alt_m": 300e3,
        "N_e": 5e11,
        "T_e": 2000.0,
        "T_i": 1200.0,
    }
    mode_fregion = dict(n_bauds=1,
                        tx_baud_length= 480e-6,
                        ipp=.008)

    ionosphere_topside = {
        "name": "topside",
        "pulse_length": 1000e-6,
        "use_iri": False,
        "iri_type": "local",
        "iri_time": "fixed parameters",
        "alt_m": 800e3,
        "N_e": 5e10,
        "T_e": 2700.0,
        "T_i": 2000.0,
    }
    mode_topside = dict(n_bauds=1,
                        tx_baud_length= 1000e-6,
                        ipp=.017)


    sim_default = dict(
        tx_sites=site_list,
        tx_radars=system_list,
        rx_sites=site_list,
        rx_radars=system_list,
        pair_list='self',
        plasma_parameter_errors=True,
        ngrid=100,
        extent=plot_extent,
        mpclient=client,
        pfunc=print,
    )

    ionosphere_list = [ionosphere_eregion, ionosphere_fregion, ionosphere_topside]
    mode_list = [mode_eregion,mode_fregion,mode_topside]
    for iidx, (iono,imode) in enumerate(zip(ionosphere_list,mode_list)):
        isim = copy(sim_default)
        isim['tname'] = "Millstone " + "(" + iono["name"] + ")"
        isim['ionosphere'] = iono
        isim.update(imode)
        print("mapping " + iono["name"])

        sfname = "millstone" + "_" + iono["name"]
        # Execute mapping, record data, and output plot as png
        ds_Millstone = map_radar_array( **isim)

        ds_Millstone.to_netcdf(
            datadir.joinpath(sfname + ".nc"), engine="h5netcdf", invalid_netcdf=True
        )
        isr_map_plot(
            ds_Millstone,
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
