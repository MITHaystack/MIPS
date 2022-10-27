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
    ne1 = np.logspace(10.5, 12, 30)
    range_res = np.array([100, 500, 1000, 1500, 2000])
    b_lengths = 2 * range_res / sc.c

    plen_ns = 500000
    plen = float(plen_ns) * 1e-9
    n_bauds = np.ceil(plen / b_lengths)
    duty = plen / 2e-3
    bulk_dop = (sc.c / (f_c * 4)) * (1 / b_lengths)

    paramvalues = dict(
        frequency_Hz=230e6,
        peak_power_W=5e6,
        maximum_range_m=400e3,
        pulse_length_ns=plen_ns,
        duty_cycle=duty,
        efficiency_tx=1.0,
        efficiency_rx=1.0,
        gain_tx_dB=43.0,
        gain_rx_dB=22.0,
        bandwidth_factor=1.0,
        tx_to_target_range_m=150e3,
        target_to_rx_range_m=150e3,
        Te=400.0,
        Ti=300.0,
        tsys_type="fixed_zero",
        excess_rx_noise_K=0,
        estimation_error_stdev=0.05,
        monostatic=True,
        tx_target_rx_angle=0,
        bistatic_volume_factor=1.0,
        quick_bandwidth_estimate=True,
        calculate_plasma_parameter_errors=False,
        mtime_estimate_method="std",
    )
    paramvalues["NO+"] = 0.5
    paramvalues["O2+"] = 0.5

    coorddict = {
        "Ne": ne1,
        "maximum_bulk_doppler": ("range_res", bulk_dop),
        "n_bauds": ("range_res", n_bauds),
        "range_res": range_res,
    }

    data_dims = {
        "Ne": len(ne1),
        "range_res": len(range_res),
    }
    dataset = simulate_data(data_dims, coorddict, paramvalues)
    return dataset


def make_paper_data():
    """Run the calculation found in the Eiscat paper.

    Returns
    -------
    range_res : array_like
        Range resolution from paper calculation.
    ne1 : array_like
        Electron density from paper calculation.
    t_mat : array_like
        Measurement time estimation from paper.
    """
    f_c = 230e6
    Ptx = 5e6
    Gtx = np.power(10.0, 43.0 / 10)
    Grx = np.power(10, 22.0 / 10)
    theta = np.deg2rad(1)
    lam = sc.c / f_c
    Tsys = 100

    t_p = 5e-4
    ipp = 2e-3
    Fm = 1.0 / ipp

    Rtx = 150e3
    Rrx = 150e3
    eps = 0.05
    Te = 400.0
    Ti = 300.0

    ne1 = np.logspace(10.5, 12, 30)
    range_res = np.array([100, 500, 1000, 1500, 2000])
    nemat, resmat = np.meshgrid(ne1, range_res)

    b_lengths = 2 * resmat / sc.c
    n_bauds = np.ceil(t_p / b_lengths)

    Fc = Fm * n_bauds * (n_bauds - 1) / 2
    V = (
        (2 * np.pi * resmat * (1 - np.cos(theta / 2)))
        * (3 * Rtx**2 + 3 * Rtx * resmat + resmat**2)
        / 3
    )

    r_e = sc.physical_constants["classical electron radius"][0]
    sige = 4 * np.pi * r_e**2
    sigp = sige * np.power((1 + Te / Ti), -1)
    sig = V * sigp * nemat
    Ps = (Ptx * Gtx * Grx * lam**2 * sig) / ((4 * np.pi) ** 3 * Rtx**2 * Rrx**2)
    Pn = sc.k * Tsys / b_lengths
    snr = Ps / Pn
    t_mat = ((Ps + Pn) / (eps * Ps)) ** 2 / Fc
    t_mats = ((snr + 1) / (eps * snr)) ** 2 / Fc

    return range_res, ne1, t_mat


def make_plot(ds1, figdir, range_res, ne1, t_mat):
    """Create the plot and save a figure.

    Parameters
    ----------
    ds1 : xarray.Dataset
        Data set created from MIPS.
    figdir : Path
        Directory holding the figure.
    range_res : array_like
        Range resolution from paper calculation.
    ne1 : array_like
        Electron density from paper calculation.
    t_mat : array_like
        Measurement time estimation from paper.
    """

    fig, (ax1, ax2) = plt.subplots(2, figsize=(6.5, 10))
    damtime = ds1["measurement_time"]
    damtime.attrs = ds1.measurement_time.attrs
    damtime.plot.line(x="Ne", xscale="log", yscale="log", ax=ax1)
    ax1.grid(True)
    ax1.set_xlabel(r"Electron density [m$^{-3}$]")
    ax1.set_ylabel("integration time [s]")
    ax1.set_title("MIPS")
    ax1.set_ylim([pow(10, -1), pow(10, 5.5)])

    rr_str = ["{0:d} m".format(i) for i in range_res]
    ax2.plot(ne1, t_mat.T)
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.legend(rr_str, title="Range Resolution [m]")
    ax2.grid(True)
    ax2.set_xlabel(r"Electron density [m$^{-3}$]")
    ax2.set_ylabel("integration time [s]")
    ax2.set_title("Original Paper")
    ax2.set_ylim([pow(10, -1), pow(10, 5.5)])

    plt.tight_layout()
    plt.savefig(figdir.joinpath("figure2compare.png"))
    plt.close(fig)


if __name__ == "__main__":

    savedir = Path("comparison").absolute()
    savedir.mkdir(exist_ok=True)
    figdir = savedir.joinpath("figs")
    figdir.mkdir(exist_ok=True)
    datadir = savedir.joinpath("data")
    datadir.mkdir(exist_ok=True)

    d1 = make_mips_data()
    d1.to_netcdf(
        datadir.joinpath("comparison.nc"), engine="h5netcdf", invalid_netcdf=True
    )
    range_res, ne1, t_mat = make_paper_data()
    make_plot(d1, figdir, range_res, ne1, t_mat)
