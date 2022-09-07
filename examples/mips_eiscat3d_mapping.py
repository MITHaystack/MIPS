#!/usr/bin/env python
"""
    mips_eiscat3d_mapping.py

    This is a set of simulations for mapping the spatial performance of
    radars and radar networks with the MIPS model, targeted to the EISCAT 3D
    radar system.

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
    tx_names = ['E3D Core (5MW)']
    tx_system_list = ['eiscat3d_tx']
    tx_site_list = ['skibotn']

    rx_names = ['E3D Core (5MW)','E3D RX Kaiseniemi','E3D RX Karesuvanto']
    rx_system_list = ['eiscat3d_tx','eiscat3d_rx','eiscat3d_rx']
    rx_site_list = ['skibotn','kaiseniemi','karesuvanto']


    plot_extent = {
        "center_lat" : 68.82,
        "center_lon" : 20.41,
        "delta_lat" : 20.0,
        "delta_lon" : 30.0,
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

        #print("mapping " + str(names) + " (" + str(site_list) + ", " + str(system_list) + ")" + " for " + iono['name'])

        sfname = 'e3d_multistatic' + '_' + iono['name']
        # Execute mapping, record data, and output plot as png
        ds_E3D = map_radar_array('E3D Multistatic' + '(' + iono['name'] +')',
                        tx_site_list,tx_system_list,
                        rx_site_list,rx_system_list,
                        pair_list='mimo',ionosphere=iono,plasma_parameter_errors=True,mpclient=None,extent=plot_extent,ngrid=120)
        ds_E3D.to_netcdf(datadir.joinpath(sfname + '.nc'),engine='h5netcdf',invalid_netcdf=True)
        isr_map_plot(ds_E3D,
                map_parameters=map_parameters,
                dval_max=dval_max,
                map_zoom=.2,
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
