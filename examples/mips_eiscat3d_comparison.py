#!/usr/bin/env python
"""
    mips_eiscat3d_comparison.py

    This is a set of simulations for mapping the spatial performance of
    radars and radar networks with the MIPS model, targeted to the EISCAT 3D
    radar system.
    J. Stamm, J. Vierinen, J. M. Urco, B. Gustavsson, and J. L. Chau, “Radar imaging with EISCAT 3D,” Annales Geophysicae, vol. 39, no. 1, pp. 119–134, Feb. 2021, doi: 10.5194/angeo-39-119-2021.

"""

from pathlib import Path
from mips import simulate_data
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as sc

def make_mips_data():
    """Create the dataset.

    Returns
    -------
    ds1 : xarray.Dataset
        Data set created from MIPS.

    """
    f_c = 230e6
    ne1 = np.logspace(10.5, 12,20)
    range_res = np.array([100,500,1000,1500,2000])
    b_lengths = 2*range_res/sc.c
    n_bauds = np.ceil(5e-4/b_lengths)
    duty = 5e-4/2e-3
    bulk_dop = (sc.c/(f_c*4))*(1/b_lengths)

    paramvalues = dict(
        frequency_Hz=230e6,
        peak_power_W=5e6,
        maximum_range_m=400e3,
        duty_cycle=duty,
        efficiency_tx=1.0,
        efficiency_rx=1.0,
        gain_tx_dB=43.,
        gain_rx_dB=22.,
        bandwidth_factor=1.0,
        tx_to_target_range_m=150e3,
        target_to_rx_range_m=150e3,
        Te=400.0,
        Ti=300.0,
        tsys_type="fixed_zero",
        excess_rx_noise_K=0,
        estimation_error_stdev=0.05,
        maximum_bulk_doppler=300,
        monostatic=True,
        tx_target_rx_angle=0,
        bistatic_volume_factor=1.0,
        quick_bandwidth_estimate=True,
        calculate_plasma_parameter_errors=False,
    )
    paramvalues["NO+"] = .5
    paramvalues["O2+"] = 0.5


    coorddict = {
        "Ne": ne1,
        "baud_length_s": ("range_res",b_lengths),
        "maximum_bulk_doppler":("range_res",bulk_dop),
        "n_bauds":('range_res',n_bauds),
        "range_res":range_res,
    }

    data_dims = {
        "Ne": len(ne1),
        "range_res": len(range_res),
    }
    dataset = simulate_data(data_dims, coorddict, paramvalues)
    return dataset


def make_plot(ds1,figdir):
    """Create the plot and save a figure.

    Parameters
    ----------
    ds1 : xarray.Dataset
        Data set created from MIPS.
    figdir : Path
        Directory holding the figure.
    """



    damtime = ds1['measurement_time']
    damtime.attrs = ds1.measurement_time.attrs
    damtime.plot.line(x='Ne',xscale='log',yscale='log')
    plt.grid(True)
    plt.xlabel(r'Electron density [m$^{-3}$]')
    plt.ylabel('integration time [s]')
    plt.savefig(figdir.joinpath('figure2compare.png'))

if __name__ == '__main__':
    savedir = Path('comparison').absolute()
    savedir.mkdir(exist_ok = True)
    figdir = savedir.joinpath('figs')
    figdir.mkdir(exist_ok=True)
    datadir = savedir.joinpath('data')
    datadir.mkdir(exist_ok=True)

    d1 = make_mips_data()
    d1.to_netcdf(datadir.joinpath('comparison.nc'),engine='h5netcdf',invalid_netcdf=True)

    make_plot(d1,figdir)
