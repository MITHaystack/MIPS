#!/usr/bin/env python
"""
    mips_misa_mapping.py

    This is a set of simulations for mapping the spatial performance of
    with the MIPS model, targeted to the Millstone site and radar.

"""

from pathlib import Path
from mips.isr_mapper import map_radar_array
from mips.isr_plotting import isr_map_plot

from mips.isr_sites import *

def main():
    # set up directories
    savedir = Path('mapping').absolute()
    savedir.mkdir(exist_ok = True)
    figdir = savedir.joinpath('figs')
    figdir.mkdir(exist_ok=True)
    datadir = savedir.joinpath('data')
    datadir.mkdir(exist_ok=True)

    # These are all consistant parameters
    map_parameters=['speed','dNe','dTi','dTe','dV']
    dval_max=[10000.0,0.1,20.0,20.0,30.0]

    # radar concepts
    names = ['Millstone UHF']
    system_list = ['millstone_misa']
    site_list = ['millstone']

    plot_extent = {
        "center_lat" : 42.6195,
        "center_lon" : -71.49173,
        "delta_lat" : 30.0,
        "delta_lon" : 35.0,
    }

    ionosphere_eregion = {
        "name" : "E-region",
        "pulse_length" : 30E-6,
        "use_iri": False,
        "iri_type": "local",
        "iri_time": "fixed parameters",
        "alt_m": 100e3,
        "N_e": 1e11,
        "T_e": 300.0,
        "T_i": 300.0,
    }

    ionosphere_fregion = {
        "name" : "F-region",
        "pulse_length" : 480E-6,
        "use_iri": False,
        "iri_type": "local",
        "iri_time": "fixed parameters",
        "alt_m": 300e3,
        "N_e": 5e11,
        "T_e": 2000.0,
        "T_i": 1200.0,
    }

    ionosphere_topside = {
        "name" : "topside",
        "pulse_length" : 1000E-6,
        "use_iri": False,
        "iri_type": "local",
        "iri_time": "fixed parameters",
        "alt_m": 800e3,
        "N_e": 5e10,
        "T_e": 2700.0,
        "T_i": 2000.0,
    }

    ionosphere_list = [ionosphere_eregion, ionosphere_fregion, ionosphere_topside]

    for iidx, iono in enumerate(ionosphere_list):

        print("mapping " + iono['name'])

        sfname = 'millstone' + '_' + iono['name']
        # Execute mapping, record data, and output plot as png
        ds_Millstone = map_radar_array('Millstone ' + '(' + iono['name'] +')',
                        site_list,system_list,
                        site_list,system_list,
                        pair_list='self',ionosphere=iono,plasma_parameter_errors=True,mpclient=None,extent=plot_extent,ngrid=100)
        ds_Millstone.to_netcdf(datadir.joinpath(sfname + '.nc'),engine='h5netcdf',invalid_netcdf=True)
        isr_map_plot(ds_Millstone,
            map_parameters=map_parameters,
            dval_max=dval_max,
            map_zoom=.4,
            range_contours=[0],
            map_fname=figdir.joinpath(sfname + '.png'),
            map_type="normal",
            annotate=True,
            legend=True,
            vmin=0.1,
            vmax=-1,
            extent=plot_extent)



if __name__ == '__main__':
    main()
