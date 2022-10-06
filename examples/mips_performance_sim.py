"""
    isr_performance_sim.py

    These are basic IS radar performance evaluations and examples for the MIPS
    model.

"""

import scipy.constants as sc
import math
import numpy as np
import pylab
import urllib.request, urllib.parse, urllib.error
import datetime
import dateutil.parser
import astropy.io.ascii
import madrigalWeb.madrigalWeb
from pathlib import Path
from mips import (
    simulate_data,
    simple_array,
    rx_temperature_model,
    sky_temperature_model,
    iri2016py,
)


version_str = "V1.0 $Id: isr_performance_sim.py 16966 2021-10-27 14:07:26Z brideout $"


def model_run_1():
    """Model run 1: sweep frequency for fixed effective area and power
    in the case of a mono-static radar
    (parabolic antenna radar chart)
    """
    print("model 1 : frequency versus power aperture")

    frequencies = np.arange(50e6, 1300e6, 5e6)
    Aeff = 1000.0
    pwr = 1e6
    ant_eff = 0.6
    gn = 10 * np.log10(4 * math.pi * ant_eff * Aeff / (sc.c / frequencies) ** 2.0)

    paramvalues = dict(
        peak_power_W=pwr,
        maximum_range_m=800e3,
        baud_length_s=1000e-6,
        duty_cycle=0.1,
        efficiency_tx=1.0,
        efficiency_rx=1.0,
        bandwidth_factor=1.0,
        tx_to_target_range_m=600e3,
        target_to_rx_range_m=600e3,
        Ne=1e11,
        Te=1000.0,
        Ti=1000.0,
        tsys_type="fixed_zero",
        estimation_error_stdev=0.05,
        maximum_bulk_doppler=1500.0,
        monostatic=True,
        tx_target_rx_angle=0,
        bistatic_volume_factor=1.0,
        quick_bandwidth_estimate=True,
        calculate_plasma_parameter_errors=False,
        Aeff=Aeff,
    )
    paramvalues["O+"] = 1.0

    excess_tsys = [0.0, 50.0, 100.0, 150.0]

    coorddict = {
        "gain_tx_dB": ("frequency_Hz", gn),
        "gain_rx_dB": ("frequency_Hz", gn),
        "frequency_Hz": frequencies,
        "excess_rx_noise_K": excess_tsys,
    }

    data_dims = {
        "frequency_Hz": len(frequencies),
        "excess_rx_noise_K": len(excess_tsys),
    }
    dataset = simulate_data(data_dims, coorddict, paramvalues)
    # save data to disk
    datapath = Path("model_runs")
    datapath.mkdir(exist_ok=True)
    dataset.to_netcdf(
        datapath.joinpath("model_run_1.nc"), engine="h5netcdf", invalid_netcdf=True
    )
    # make the plots
    attrs = dataset.attrs
    lstr = (
        "%.0f km alt\npl=%.0f ms\nne=%.0e m$^{-3}$\n%.0fMW peak @ d=%.0f%%\nTe=%.0f,Ti=%.0f\n$A_{\mathrm{eff}}=%1.0f$ m$^{2}$"
        % (
            attrs["target_to_rx_range_m"] / 1e3,
            attrs["baud_length_s"] * 1e3,
            attrs["Ne"],
            attrs["peak_power_W"] / 1e6,
            attrs["duty_cycle"] * 100,
            attrs["Te"],
            attrs["Ti"],
            attrs["Aeff"],
        )
    )

    f, ax = pylab.subplots(2, 1, sharex=True)
    excess_tsys = dataset["excess_rx_noise_K"].values
    for i in range(len(excess_tsys)):
        ax[0].plot(
            dataset["frequency_Hz"].values / 1e6,
            dataset["snr"].values[:, i],
            label="$T_{\mathrm{rx}}=%1.0f$ K" % (excess_tsys[i]),
        )
    ax[0].legend(fontsize=8)
    ax[0].text(1050, 1.0, lstr, fontsize=8, backgroundcolor=(1, 1, 1, 0.5))
    ax[0].set_title("IS Radar Performance: Fixed Antenna Area")
    ax[0].set_ylabel("SNR")
    ax[0].set_ylim(0, 6.0)
    ax[0].set_xlim(0, 1300)
    ax[0].grid(True)

    lstr = "%.0f m^2 Aeff" % Aeff
    for i in range(len(excess_tsys)):
        ax[1].plot(
            dataset["frequency_Hz"].values / 1e6,
            dataset["measurement_time"].values[:, i],
            label="$T_{\mathrm{rx}}=%1.0f$ K" % (excess_tsys[i]),
        )

    ax[1].set_xlabel("Freq (MHz)")
    ax[1].set_ylabel("Meas time (sec)")
    ax[1].set_ylim(0, 100)
    ax[1].legend(fontsize=8)
    ax[1].grid(True)
    figpath = Path("figures")
    figpath.mkdir(exist_ok=True)
    f.savefig(figpath.joinpath("is_sim_fixed_ant_area.png"))


def model_run_2():
    """
    Model run 2: sweep frequency for fixed phased array element count
    Assume locally bistatic for RX noise.
    """
    print("model 2 : frequency for fixed array element count")

    frequencies = np.arange(50e6, 1300e6, 5e6)
    element_pwr = 1e3
    n_elements = 1000

    (pwr, gn) = simple_array(n_elements, element_pwr)

    rng = 600e3
    tpulse = rng * 2 / sc.c

    paramvalues = dict(
        peak_power_W=pwr,
        maximum_range_m=800e3,
        baud_length_s=rng * 2 / sc.c,
        excess_rx_noise_K=0.0,
        gain_tx_dB=gn,
        gain_rx_dB=gn,
        duty_cycle=0.1,
        efficiency_tx=1.0,
        efficiency_rx=1.0,
        bandwidth_factor=1.0,
        tx_to_target_range_m=rng,
        target_to_rx_range_m=rng,
        Ne=1e11,
        Te=1000.0,
        Ti=1000.0,
        tsys_type="fixed_medium",
        estimation_error_stdev=0.05,
        maximum_bulk_doppler=1500.0,
        monostatic=True,
        tx_target_rx_angle=0,
        bistatic_volume_factor=1.0,
        quick_bandwidth_estimate=True,
        calculate_plasma_parameter_errors=False,
    )
    paramvalues["O+"] = 1.0

    xtra_vals = dict(n_elements=n_elements, element_pwr=element_pwr)
    paramvalues.update(xtra_vals)

    coorddict = {"frequency_Hz": frequencies}

    data_dims = {"frequency_Hz": len(frequencies)}
    dataset = simulate_data(data_dims, coorddict, paramvalues)

    datapath = Path("model_runs")
    datapath.mkdir(exist_ok=True)
    dataset.to_netcdf(
        datapath.joinpath("model_run_2.nc"), engine="h5netcdf", invalid_netcdf=True
    )

    attrs = dataset.attrs
    lstr = (
        "%i elements\n%.0f W/element\n%.0f km alt\npl=%.0f ms\nne=%.0e m^-3\n%.0fMW peak @ d=%.0f%%\nTe=%.0f,Ti=%.0f"
        % (
            attrs["n_elements"],
            attrs["element_pwr"],
            rng / 1e3,
            attrs["baud_length_s"] * 1e3,
            attrs["Ne"],
            attrs["peak_power_W"] / 1e6,
            attrs["duty_cycle"] * 100,
            attrs["Te"],
            attrs["Ti"],
        )
    )

    f, ax = pylab.subplots(2, 1, sharex=True)
    ax[0].plot(dataset["frequency_Hz"].values / 1e6, dataset["snr"].values)
    ax[0].set_title("IS Radar Performance: Fixed Array Element Count")
    ax[0].set_ylabel("SNR")
    ax[0].set_ylim(0, 8)
    ax[0].set_xlim(0, 1300)
    ax[0].grid(True)

    ax[1].plot(dataset["frequency_Hz"].values / 1e6, dataset["measurement_time"].values)
    ax[1].text(1000.0, 100.0, lstr, fontsize=8, backgroundcolor=(1, 1, 1, 0.5))
    ax[1].set_xlabel("Freq (MHz)")
    ax[1].set_ylabel("Meas time (sec)")
    ax[1].set_ylim(0, 200)
    ax[1].grid(True)

    figpath = Path("figures")
    figpath.mkdir(exist_ok=True)
    f.savefig(figpath.joinpath("is_sim_fixed_element_count.png"))


def model_run_3():
    """
    Model run 3: sweep duty cycle for fixed number of elements and power
    """

    print("model 3 : sweep duty cycle for fixed power aperture")

    duty_swp = np.arange(0.01, 1.0, 0.01)
    frequencies = np.arange(50e6, 1300e6, 5e6)
    element_pwr = 1e3
    n_elements = 1000

    (pwr, gn) = simple_array(n_elements, element_pwr)

    rng = 600e3
    tpulse = rng * 2 / sc.c

    paramvalues = dict(
        peak_power_W=pwr,
        maximum_range_m=800e3,
        baud_length_s=rng * 2 / sc.c,
        excess_rx_noise_K=0.0,
        gain_tx_dB=gn,
        gain_rx_dB=gn,
        frequency_Hz=440e6,
        efficiency_tx=1.0,
        efficiency_rx=1.0,
        bandwidth_factor=1.0,
        tx_to_target_range_m=rng,
        target_to_rx_range_m=rng,
        Ne=1e11,
        Te=1000.0,
        Ti=1000.0,
        tsys_type="fixed_medium",
        estimation_error_stdev=0.05,
        maximum_bulk_doppler=1500.0,
        monostatic=True,
        tx_target_rx_angle=0,
        bistatic_volume_factor=1.0,
        quick_bandwidth_estimate=True,
        calculate_plasma_parameter_errors=False,
    )
    paramvalues["O+"] = 1.0

    xtra_vals = dict(n_elements=n_elements, element_pwr=element_pwr)
    paramvalues.update(xtra_vals)

    coorddict = {"duty_cycle": duty_swp}

    data_dims = {"duty_cycle": len(duty_swp)}
    dataset = simulate_data(data_dims, coorddict, paramvalues)

    datapath = Path("model_runs")
    datapath.mkdir(exist_ok=True)
    dataset.to_netcdf(
        datapath.joinpath("model_run_3.nc"), engine="h5netcdf", invalid_netcdf=True
    )

    duty_max_for_monostatic = 0.2

    f = pylab.figure()
    ax = [pylab.gca()]

    ax[0].semilogy(
        dataset["duty_cycle"].values * 100, dataset["measurement_time"].values
    )
    #    ax[0].set_xlim(0,100)
    #    ax[0].set_ylim(0,100)
    yl = ax[0].get_ylim()
    ax[0].vlines(duty_max_for_monostatic * 100, yl[0], yl[1], "g")
    ax[0].text(25, 275, "bistatic\n(can be clutter limited)\n")
    ax[0].text(2, 450, "mono-static")

    attrs = dataset.attrs
    lstr = (
        "%i elements\n%.0f W/element\n%.0f km alt\npl=%.0f ms\nne=%.0e m^-3\n%.0fMW peak\nTe=%.0f,Ti=%.0f"
        % (
            attrs["n_elements"],
            attrs["element_pwr"],
            rng / 1e3,
            attrs["baud_length_s"] * 1e3,
            attrs["Ne"],
            attrs["peak_power_W"] / 1e6,
            attrs["Te"],
            attrs["Ti"],
        )
    )

    ax[0].text(82, 200.0, lstr, fontsize=8, backgroundcolor=(1, 1, 1, 0.5))

    ax[0].set_title("IS Radar Performance: Duty Cycle at Constant Peak Power")
    ax[0].set_xlabel("Duty cycle, percent")
    ax[0].set_ylabel("Time to 5% error (sec)")
    ax[0].grid(True)

    f.savefig("figures/is_sim_constant_peak_pwr.png")


def model_run_4():
    """
    Model run 4: sweep duty cycle for constant avg element power
    """

    print("model 4 : sweep duty cycle for constant avg element power")

    avg_pwr = 200.0
    n_elements = 1000
    tavg_pwr = avg_pwr * n_elements
    duty_swp = np.arange(0.01, 1.0, 0.01)

    rng = 300e3

    x_tsys_bistat = 0.0
    x_tsys_mono = 80.0

    (pwr_swp, gn) = simple_array(n_elements, avg_pwr / duty_swp)

    paramvalues = dict(
        maximum_range_m=800e3,
        baud_length_s=rng * 2 / sc.c,
        excess_rx_noise_K=x_tsys_mono,
        gain_tx_dB=gn,
        gain_rx_dB=gn,
        frequency_Hz=440e6,
        efficiency_tx=1.0,
        efficiency_rx=1.0,
        bandwidth_factor=1.0,
        tx_to_target_range_m=rng,
        target_to_rx_range_m=rng,
        Ne=1e11,
        Te=1000.0,
        Ti=1000.0,
        tsys_type="fixed_medium",
        estimation_error_stdev=0.05,
        maximum_bulk_doppler=1500.0,
        monostatic=True,
        tx_target_rx_angle=0,
        bistatic_volume_factor=1.0,
        quick_bandwidth_estimate=True,
        calculate_plasma_parameter_errors=False,
    )
    paramvalues["O+"] = 1.0

    xtra_vals = dict(n_elements=n_elements, element_pwr=tavg_pwr)
    paramvalues.update(xtra_vals)

    coorddict = {"duty_cycle": duty_swp, "peak_power_W": ("duty_cycle", pwr_swp)}

    data_dims = {"duty_cycle": len(duty_swp)}
    dataset_m = simulate_data(data_dims, coorddict, paramvalues)

    datapath = Path("model_runs")
    datapath.mkdir(exist_ok=True)
    dataset_m.to_netcdf(
        datapath.joinpath("model_run_4m.nc"), engine="h5netcdf", invalid_netcdf=True
    )

    paramvalues["excess_rx_noise_K"] = x_tsys_bistat
    paramvalues["monostatic"] = False
    dataset_b = simulate_data(data_dims, coorddict, paramvalues)
    dataset_b.to_netcdf(
        datapath.joinpath("model_run_4b.nc"), engine="h5netcdf", invalid_netcdf=True
    )

    # monostatic limit at 5x range
    # This is an assumption about the self clutter distance
    # and radar transmission patterns. A monostatic radar
    # cannot be transmitting when it is receiving signals...

    duty_max_for_monostatic = 0.2

    f = pylab.figure()
    ax = [pylab.gca()]
    ax[0].semilogy(
        dataset_b["duty_cycle"].values * 100,
        dataset_b["measurement_time"].values,
        "b",
        label="Locally bistatic",
    )

    indx = np.nonzero(duty_swp * 100 <= duty_max_for_monostatic * 100)[0][-1] + 1

    ax[0].semilogy(
        dataset_m["duty_cycle"].values[:indx] * 100,
        dataset_m["measurement_time"].values[:indx],
        "r",
        label="monostatic",
    )
    ax[0].set_xlim(1, 100)
    ax[0].set_ylim(1, 100)
    yl = ax[0].get_ylim()
    ax[0].vlines(duty_max_for_monostatic * 100, yl[0], yl[1], "g")
    ax[0].text(25, 46, "bistatic\n(can be clutter limited)\n")
    ax[0].text(2, 65, "mono-static")

    attrs = dataset_m.attrs
    lstr = (
        "%i elements\n%.0f W/element\n%.0f km alt\npl=%.0f ms\nne=%.0e m^-3\nTe=%.0f,Ti=%.0f"
        % (
            attrs["n_elements"],
            attrs["element_pwr"],
            rng / 1e3,
            attrs["baud_length_s"] * 1e3,
            attrs["Ne"],
            attrs["Te"],
            attrs["Ti"],
        )
    )

    ax[0].text(82, 20.0, lstr, fontsize=8, backgroundcolor=(1, 1, 1, 0.5))

    ax[0].legend(loc="upper right")
    ax[0].set_title("IS Radar Performance: Duty Cycle at Constant Average Power")
    ax[0].set_xlabel("Duty cycle, percent")
    ax[0].set_ylabel("Time to 5% error (sec)")
    ax[0].grid(True)

    f.savefig("figures/is_sim_constant_avg_pwr.png")


def model_run_5():
    """
    Model run 5: sweep element count, power for constant total TX power
    """

    print("model 5 : sweep element count and power for constant total TX power")

    total_pwr = 1e6
    n_elements = np.arange(100, 10000, 100)

    rng = 300e3

    (pwr_swp, gn_swp) = simple_array(n_elements, total_pwr / n_elements)

    x_tsys = 0.0
    paramvalues = dict(
        maximum_range_m=800e3,
        baud_length_s=rng * 2 / sc.c,
        duty_cycle=0.1,
        excess_rx_noise_K=x_tsys,
        frequency_Hz=440e6,
        efficiency_tx=1.0,
        efficiency_rx=1.0,
        bandwidth_factor=1.0,
        tx_to_target_range_m=rng,
        target_to_rx_range_m=rng,
        Ne=1e11,
        Te=1000.0,
        Ti=1000.0,
        tsys_type="fixed_medium",
        estimation_error_stdev=0.05,
        maximum_bulk_doppler=1500.0,
        monostatic=True,
        tx_target_rx_angle=0,
        bistatic_volume_factor=1.0,
        quick_bandwidth_estimate=True,
        calculate_plasma_parameter_errors=False,
    )
    paramvalues["O+"] = 1.0

    coorddict = {
        "n_elements": n_elements,
        "peak_power_W": ("n_elements", pwr_swp),
        "gain_tx_dB": ("n_elements", gn_swp),
        "gain_rx_dB": ("n_elements", gn_swp),
    }

    data_dims = {"n_elements": len(n_elements)}
    dataset = simulate_data(data_dims, coorddict, paramvalues)

    datapath = Path("model_runs")
    datapath.mkdir(exist_ok=True)
    dataset.to_netcdf(
        datapath.joinpath("model_run_5.nc"), engine="h5netcdf", invalid_netcdf=True
    )

    f, ax = pylab.subplots(2, 1, sharex=True)

    ax[0].plot(n_elements, dataset["snr"].values)

    attrs = dataset.attrs
    lstr = "%.0f km alt\npl=%.0f ms\nne=%.0e m^-3\n%.0fMW peak\nTe=%.0f,Ti=%.0f" % (
        rng / 1e3,
        attrs["baud_length_s"] * 1e3,
        attrs["Ne"],
        total_pwr / 1e6,
        attrs["Te"],
        attrs["Ti"],
    )

    ax[0].set_ylabel("SNR")
    ax[0].grid(True)
    ax[0].set_title("IS Radar Performance: Element Count at Fixed Total Power ")

    ax[1].semilogy(n_elements, dataset["measurement_time"].values)
    ax[1].set_xlabel("Number of elements")
    ax[1].set_ylabel("Time to 5% error (sec)")
    ax[1].set_ylim(1, 100)
    ax[1].set_xlim(1, 10000)
    ax[1].grid(True)

    f.savefig("figures/is_sim_fixed_total_pwr.png")


def model_run_6():
    """
    Model run 6: Locally bistatic UHF (440 MHz) ISR versus
    AMISR design 450 MHz monostatic radar.
    Uses IRI-2012 for plasma parameters over Ethiopia radar
    site @ Bahir Dar.
    """

    print("model 6 : Locally bistatic UHF radar at Ethiopia site")

    t = datetime.datetime(2015, 1, 15, 15, 0, 0)
    bhd_lat = 11.587  # deg
    bhd_lon = 37.357  # deg
    salt = 90  # km
    ealt = 1000  # km
    dalt = 10  # km

    # use the online model for ease and as an example
    # example for the local IRI model is in the mapping code...
    m = iri2016py(t, bhd_lat, bhd_lon, salt, ealt, dalt)
    rng_m = m["alt_km"].values * 1000
    alt_km = m["alt_km"].values
    # Concept at 440 MHz (pulsed)
    pwr = 1.25e6
    fswp = 440e6
    tpulse = 480e-6
    duty = 0.2
    gn = 36.0
    ncount_440 = 1200
    eff_tx = 1
    eff_rx = 1
    bw_fac = 1.0
    x_tsys = 0.0
    est_err = 0.05

    paramvalues = dict(
        peak_power_W=pwr,
        baud_length_s=tpulse,
        duty_cycle=duty,
        gain_tx_dB=gn,
        gain_rx_dB=gn,
        efficiency_tx=eff_tx,
        efficiency_rx=eff_rx,
        bandwidth_factor=bw_fac,
        frequency_Hz=fswp,
        excess_rx_noise_K=x_tsys,
        tsys_type="fixed_vlow",
        estimation_error_stdev=est_err,
        maximum_bulk_doppler=1500.0,
        monostatic=True,
        tx_target_rx_angle=0.0,
        bistatic_volume_factor=1.0,
        quick_bandwidth_estimate=False,
        calculate_plasma_parameter_errors=False,
    )
    paramvalues["O+"] = 1.0

    coorddict = {
        "gdalt": rng_m / 1000,
        "tx_to_target_range_m": ("gdalt", rng_m),
        "target_to_rx_range_m": ("gdalt", rng_m),
        "Ne": ("gdalt", m["ne"].values),
        "Te": ("gdalt", m["Te"].values),
        "Ti": ("gdalt", m["Ti"].values),
        "maximum_range_m": ("gdalt", rng_m),
    }

    data_dims = {"gdalt": len(rng_m)}
    datasetni_p = simulate_data(data_dims, coorddict, paramvalues)

    # Concept UHF RAdar at 440 MHz (CW)
    pwr = 0.250e6
    fswp = 440e6
    tpulse = 480e-6
    duty = 1.0
    gn = 36.0  # hard coded gain
    ncount_440_cw = 1200
    eff_tx = 1
    eff_rx = 1
    bw_fac = 1.0
    x_tsys = 0.0
    est_err = 0.05

    paramvalues = dict(
        peak_power_W=pwr,
        baud_length_s=tpulse,
        duty_cycle=duty,
        gain_tx_dB=gn,
        gain_rx_dB=gn,
        efficiency_tx=eff_tx,
        efficiency_rx=eff_rx,
        bandwidth_factor=bw_fac,
        frequency_Hz=fswp,
        excess_rx_noise_K=x_tsys,
        tsys_type="fixed_vlow",
        estimation_error_stdev=est_err,
        maximum_bulk_doppler=1500.0,
        monostatic=True,
        tx_target_rx_angle=0.0,
        bistatic_volume_factor=1.0,
        quick_bandwidth_estimate=False,
        calculate_plasma_parameter_errors=False,
    )
    paramvalues["O+"] = 1.0

    datasetni_cw = simulate_data(data_dims, coorddict, paramvalues)

    # AMISR monostatic parameters: ISR at 449 MHz
    # FDL comments:
    #  actual AMISR systems run about 1.6 MW and 41 dB with messed up
    #  patterns due to non-working modules / panels. So this is an
    #  overestimate of real world performance.
    #  In reality only 87% of elements are typically active.

    pwr = 2e6
    fswp = 449e6
    tpulse = 480e-6
    duty = 0.1
    gn = 42.0  # hard coded to actual AMISR ideal gain
    ncount_449 = 4000
    eff_tx = 1
    eff_rx = 1
    bw_fac = 1.0
    x_tsys = 10.0  # extra RX loss in actual system
    est_err = 0.05

    paramvalues = dict(
        peak_power_W=pwr,
        baud_length_s=tpulse,
        duty_cycle=duty,
        gain_tx_dB=gn,
        gain_rx_dB=gn,
        efficiency_tx=eff_tx,
        efficiency_rx=eff_rx,
        bandwidth_factor=bw_fac,
        frequency_Hz=fswp,
        excess_rx_noise_K=x_tsys,
        tsys_type="fixed_vlow",
        estimation_error_stdev=est_err,
        maximum_bulk_doppler=1500.0,
        monostatic=True,
        tx_target_rx_angle=0.0,
        bistatic_volume_factor=1.0,
        quick_bandwidth_estimate=False,
        calculate_plasma_parameter_errors=False,
    )
    paramvalues["O+"] = 1.0

    datasetamisr = simulate_data(data_dims, coorddict, paramvalues)

    f, ax = pylab.subplots(1, 3, sharey=True)

    ax[0].plot(np.log10(m["ne"].values), alt_km, label="Ne")
    ax[0].legend(fontsize=8)
    ax[0].set_xlabel("m^-3")
    ax[0].set_ylabel("Altitude, km")
    ax[0].set_title("IRI-2012")
    ax[0].grid(True)
    ax[0].set_xticks(np.arange(10, 13, 1))
    ax[1].plot(m["Te"].values, alt_km, "g", label="Te")
    ax[1].plot(m["Ti"].values, alt_km, "r", label="Ti")
    ax[1].set_title("Bahir Dar")
    ax[1].legend(fontsize=8)
    ax[1].set_xticks(np.arange(1000, 5000, 1000))
    ax[1].set_xlabel("K")
    ax[1].grid(True)
    ax[2].plot(m["Te"].values / m["Ti"].values, alt_km, label="Tr")
    ax[2].legend(fontsize=8)
    ax[2].set_title(t.isoformat())
    ax[2].set_xlabel("Temp Ratio")
    ax[2].grid(True)
    ax[2].set_xticks(np.arange(1, 4, 0.5))

    f.savefig("figures/is_sim_ethiopia_plasma_param.png")

    f, ax = pylab.subplots(2, 1)

    ax[0].plot(
        datasetni_p["snr"].values,
        alt_km,
        "b",
        label="Local Bistatic UHF (440 MHz,Gn=36.0dB,1.25MW @ 25%)",
    )
    ax[0].plot(
        datasetni_cw["snr"].values,
        alt_km,
        "m--",
        label="Local Bistatic UHF(440 MHz,Gn=36.0dB,250KW @ 100%)",
    )
    ax[0].plot(
        datasetamisr["snr"].values,
        alt_km,
        "g",
        label="AMISR (449 MHz,Gn=42.0dB,2.0MW @ 10%)",
    )
    ax[0].legend(loc="upper right", fontsize=8)
    ax[0].set_ylabel("Altitude (km)")
    ax[0].set_xlabel("SNR")
    ax[0].grid(True)
    ax[0].set_title("IS Radar: Bahir Dar (zenith; %s UTC)" % t.isoformat())

    ax[1].semilogx(
        datasetni_p["measurement_time"].values,
        alt_km,
        "b",
        label="440 MHz,1.25MW @ 20%%, %i TX/RX elem" % ncount_440,
    )
    ax[1].semilogx(
        datasetni_cw["measurement_time"].values,
        alt_km,
        "m--",
        label="440 MHz,250kW @ 100%%, %i TX/RX elem" % ncount_440_cw,
    )
    ax[1].semilogx(
        datasetamisr["measurement_time"].values,
        alt_km,
        "g",
        label="449 MHz,2.0MW @ 10%%, %i TX/RX elem" % ncount_449,
    )
    ax[1].set_xlabel("Time to 5% error (sec)")
    ax[1].set_ylabel("Altitude (km)")
    ax[1].grid(True)
    ax[1].legend(loc="lower right", fontsize=8)

    f.savefig("figures/is_sim_ethiopia_uhf_isr_amisr.png")


def model_run_7():
    """
    Model run 7: Millstone Hill 68m zenith antenna
    Compare calculated SNR with Madrigal values for
    actual daytime vertical profile experiment.
    """

    print("model 7 : Millstone Hill Zenith comparison")

    # get typical data record

    startTime = "2015-06-18T16:00:00"
    sd = dateutil.parser.parse(startTime)
    endTime = "2015-06-18T16:15:00"
    ed = dateutil.parser.parse(endTime)

    madObj = madrigalWeb.madrigalWeb.MadrigalData(
        "http://millstonehill.haystack.mit.edu/"
    )
    exps = madObj.getExperiments(
        30,
        sd.year,
        sd.month,
        sd.day,
        sd.hour,
        sd.minute,
        sd.second,
        ed.year,
        ed.month,
        ed.day,
        ed.hour,
        ed.minute,
        ed.second,
    )
    exps.sort()
    print(exps[0])

    estart = datetime.datetime(
        exps[0].startyear, exps[0].startmonth, exps[0].startday, 0, 0, 0
    )
    ds = sd - estart
    suth = ds.days * 24.0 + ds.seconds / 3600.0
    de = ed - estart
    euth = de.days * 24.0 + de.seconds / 3600.0

    expfiles = madObj.getExperimentFiles(exps[0].id)
    for ef in expfiles:
        if ef.name.find("i.00") > 0:  # zenith single pulse
            break

    print(ef)

    parms = "gdalt,uth,ne,te,dte,ti,dti,tr,dtr,vo,snp3,systmp,power,dut21"
    fstr = "filter=gdalt,200,600 filter=uth,%f,%f badval=-1e30" % (suth, euth)

    data = madObj.isprint(
        ef.name, parms, fstr, "Phil Erickson", "pje@haystack.mit.edu", "MIT"
    )

    md = astropy.io.ascii.read(data, names=parms.split(","))

    dt = estart + datetime.timedelta(seconds=md["uth"][0] * 3600)
    print(md)

    #######

    fswp = 440.2e6
    pwr = md["power"][0] * 1e3
    gn = 49.72

    tpulse = 480e-6

    duty = 0.053872
    eff_tx = 0.51  # empirically determined by SNR model-to-data match
    eff_rx = 0.51  # empirically determined by SNR model-to-data match
    bw_fac = 1.0
    est_err = 0.01
    vdopp_max = md["vo"]

    # set x_tsys to match the measured system temperature
    # Remember that madrigal Tsys for Millstone includes the sky temperature!
    # It is necessary to offset this out as a model temperature is added back in!
    # The reported temperature was uncalibrated for this experiment but was close.
    # Millstone Zenith Tsys(+sky) runs 155K typically including the sky (~ 30 to 40K)

    x_tsys = (
        md["systmp"][0]
        - rx_temperature_model(fswp, "fixed_medium")
        - sky_temperature_model(fswp)
    )
    print("x_tsys: " + str(x_tsys))

    paramvalues = dict(
        peak_power_W=pwr,
        baud_length_s=tpulse,
        duty_cycle=duty,
        gain_tx_dB=gn,
        gain_rx_dB=gn,
        efficiency_tx=eff_tx,
        efficiency_rx=eff_rx,
        bandwidth_factor=bw_fac,
        frequency_Hz=fswp,
        excess_rx_noise_K=x_tsys,
        tsys_type="fixed_medium",
        estimation_error_stdev=est_err,
        maximum_bulk_doppler=1500.0,
        monostatic=True,
        tx_target_rx_angle=0.0,
        bistatic_volume_factor=1.0,
        quick_bandwidth_estimate=False,
        calculate_plasma_parameter_errors=False,
    )
    paramvalues["O+"] = 1.0
    rng_m = np.array(md["gdalt"]) * 1000

    coorddict = {
        "gdalt": rng_m / 1000,
        "tx_to_target_range_m": ("gdalt", rng_m),
        "target_to_rx_range_m": ("gdalt", rng_m),
        "Ne": ("gdalt", np.array(md["ne"])),
        "Te": ("gdalt", np.array(md["te"])),
        "Ti": ("gdalt", np.array(md["ti"])),
        "maximum_range_m": ("gdalt", rng_m),
    }

    data_dims = {"gdalt": len(rng_m)}
    dataset = simulate_data(data_dims, coorddict, paramvalues)

    # correct SNR for difference between the optimum bandwidth used
    # in the model and the fixed 50 kHz bandwidth used by the signal processing chain
    # this is Millstone MIDAS-W single pulse specific for this particular data interval.

    dataset["snr"].data = dataset["snr"].data * dataset["echo_bandwidth"] / 50e3

    # estimate the measurement error for the given Madrigal record,
    # scaling off the ratio of actual measurement time to model measurement time
    meas_est_err = est_err * (dataset["measurement_time"] / md["dut21"]) ** 0.5

    f, ax = pylab.subplots(1, 3, sharey=True)

    ax[0].plot(np.log10(md["ne"]), md["gdalt"], label="Ne")
    ax[0].legend(fontsize=8)
    ax[0].set_xlabel("m^-3")
    ax[0].set_ylabel("Altitude, km")
    ax[0].set_title("MHO Zenith")
    ax[0].grid(True)
    ax[0].set_xticks(np.arange(10, 13, 1))
    ax[1].plot(md["te"], md["gdalt"], "g", label="Te")
    ax[1].plot(md["ti"], md["gdalt"], "r", label="Ti")
    ax[1].set_title("480us")
    ax[1].legend(fontsize=8)
    ax[1].set_xticks(np.arange(1000, 5000, 1000))
    ax[1].set_xlabel("K")
    ax[1].grid(True)
    ax[2].plot(md["te"] / md["ti"], md["gdalt"], label="Tr")
    ax[2].legend(fontsize=8)
    ax[2].set_title(dt.isoformat())
    ax[2].set_xlabel("Temp Ratio")
    ax[2].grid(True)
    ax[2].set_xticks(np.arange(1, 4, 0.5))

    f.savefig("figures/is_sim_mho_zenith_plasma_param.png")

    f = pylab.figure()
    ax = [pylab.gca()]

    ax[0].plot(md["snp3"], md["gdalt"], "b", label="MHO")
    ax[0].plot(dataset["snr"].values, md["gdalt"], "r", label="Model")
    ax[0].text(
        2,
        450,
        "Zenith TX Eff: %.2f\nZenith RX Eff: %.2f\nZenith Tsys: %.1f\nZenith Peak Pwr: %.1f MW"
        % (eff_tx, eff_rx, md["systmp"][0], md["power"][0] / 1e3),
    )
    ax[0].set_xlabel("SNR")
    ax[0].set_ylabel("Altitude, km")
    ax[0].set_title("MHO Zenith " + dt.isoformat() + " UTC")
    ax[0].grid(True)
    ax[0].legend(fontsize=8)

    f.savefig("figures/is_sim_mho_zenith_snr.png")

    f = pylab.figure()
    ax = [pylab.gca()]

    ax[0].semilogx(
        dataset["measurement_time"].values, md["gdalt"], label="1% Error Time"
    )
    ax[0].grid(True)
    ax[0].legend(fontsize=8)
    ax[0].set_xlabel("Modeled time for 1% accuracy, seconds")
    ax[0].set_ylabel("Altitude, km")
    ax[0].set_title("MHO Zenith " + dt.isoformat() + " UTC")

    f.savefig("figures/is_sim_mho_zenith_mtime.png")

    f = pylab.figure()
    ax = [pylab.gca()]

    ax[0].plot(100 * md["dti"] / md["ti"], md["gdalt"], "r", label="INSCAL dTi/Ti")
    ax[0].plot(100 * md["dte"] / md["te"], md["gdalt"], "g", label="INSCAL dTe/Te")
    ax[0].plot(100 * md["dtr"] / md["tr"], md["gdalt"], "b", label="INSCAL dTr/Tr")
    ax[0].plot(100 * meas_est_err, md["gdalt"], "m", label="Model Est accuracy")
    ax[0].text(1.75, 400, "Integration time: %.0f sec" % md["dut21"][0])
    ax[0].grid(True)
    ax[0].legend(loc="lower right", fontsize=8)
    ax[0].set_xlabel("Accuracy, percent")
    ax[0].set_ylabel("Altitude, km")
    ax[0].set_title("MHO Zenith " + dt.isoformat() + " UTC")

    f.savefig("figures/is_sim_mho_zenith_accuracy.png")

    f, ax = pylab.subplots(1, 2, sharey=True)

    ax[0].plot(md["snp3"] / dataset["snr"].values, md["gdalt"], label="MHO/Model")
    ax[0].set_xlabel("Measured/Model SNR Ratio")
    ax[0].set_ylabel("Altitude, km")
    ax[0].set_title("MHO Zenith")
    ax[0].grid(True)
    ax[0].set_xlim(0.9, 1.1)
    ax[0].legend(fontsize=8)
    ax[1].plot(
        50e3 / dataset["echo_bandwidth"].values, md["gdalt"], label="Model BW / 50 kHz"
    )
    ax[1].set_xlabel("BW Ratio")
    ax[1].set_title(dt.isoformat())
    ax[1].grid(True)
    ax[1].legend(fontsize=8)

    f.savefig("figures/is_sim_mho_zenith_ratio_bw.png")


def model_run_8():
    """
    Model run 8: Millstone Hill 46m MISA antenna
    Compare calculated SNR with Madrigal values for
    actual daytime vertical profile experiment.
    """

    print("model 8 : Millstone Hill MISA comparison")

    # get typical data record

    startTime = "2015-06-18T16:20:00"
    sd = dateutil.parser.parse(startTime)
    endTime = "2015-06-18T16:25:00"
    ed = dateutil.parser.parse(endTime)

    madObj = madrigalWeb.madrigalWeb.MadrigalData(
        "http://millstonehill.haystack.mit.edu/"
    )
    exps = madObj.getExperiments(
        30,
        sd.year,
        sd.month,
        sd.day,
        sd.hour,
        sd.minute,
        sd.second,
        ed.year,
        ed.month,
        ed.day,
        ed.hour,
        ed.minute,
        ed.second,
    )
    exps.sort()
    print(exps[0])

    estart = datetime.datetime(
        exps[0].startyear, exps[0].startmonth, exps[0].startday, 0, 0, 0
    )
    ds = sd - estart
    suth = ds.days * 24.0 + ds.seconds / 3600.0
    de = ed - estart
    euth = de.days * 24.0 + de.seconds / 3600.0

    expfiles = madObj.getExperimentFiles(exps[0].id)
    for ef in expfiles:
        if ef.name.find("k.00") > 0:  # MISA single pulse
            break

    print(ef)

    parms = "gdalt,uth,ne,te,dte,ti,dti,tr,dtr,vo,snp3,systmp,power,dut21"
    fstr = "filter=gdalt,200,600 filter=uth,%f,%f badval=-1e30" % (suth, euth)

    data = madObj.isprint(
        ef.name, parms, fstr, "Phil Erickson", "pje@haystack.mit.edu", "MIT"
    )
    md = astropy.io.ascii.read(data, names=parms.split(","))

    dt = estart + datetime.timedelta(seconds=md["uth"][0] * 3600)

    print(md)

    #######

    fswp = 440.2e6
    pwr = md["power"][0] * 1e3
    gn = 44.7

    tpulse = 480e-6
    duty = 0.053872
    # MISA value is too high and we don't quite understand it at this point.
    eff_tx = 0.6  # empirically determined by SNR model-to-data match
    eff_rx = 0.6  # empirically determined by SNR model-to-data match

    bw_fac = 1.0
    est_err = 0.01
    vdopp_max = md["vo"]

    # set x_tsys to match the measured system temperature
    # Remember that madrigal Tsys for Millstone includes the sky temperature!
    # It is necessary to offset this out as a model temperature is added back in!
    # The reported temperature was uncalibrated for this experiment.
    # Millstone MISA Tsys(+ feed loss + Tsky) runs 165K typically including the sky (~ 30 to 40K)
    # Offset to likely value range for Tsys(+sky) of ~ 165K.
    md["systmp"][0] = 145.0
    x_tsys = (
        md["systmp"][0]
        - rx_temperature_model(fswp, "fixed_medium")
        - sky_temperature_model(fswp)
    )
    print("x_tsys: " + str(x_tsys))

    paramvalues = dict(
        peak_power_W=pwr,
        baud_length_s=tpulse,
        duty_cycle=duty,
        gain_tx_dB=gn,
        gain_rx_dB=gn,
        efficiency_tx=eff_tx,
        efficiency_rx=eff_rx,
        bandwidth_factor=bw_fac,
        frequency_Hz=fswp,
        excess_rx_noise_K=x_tsys,
        tsys_type="fixed_medium",
        estimation_error_stdev=est_err,
        maximum_bulk_doppler=1500.0,
        monostatic=True,
        tx_target_rx_angle=0.0,
        bistatic_volume_factor=1.0,
        quick_bandwidth_estimate=False,
        calculate_plasma_parameter_errors=False,
    )
    paramvalues["O+"] = 1.0
    rng_m = np.array(md["gdalt"]) * 1000

    coorddict = {
        "gdalt": rng_m / 1000,
        "tx_to_target_range_m": ("gdalt", rng_m),
        "target_to_rx_range_m": ("gdalt", rng_m),
        "Ne": ("gdalt", np.array(md["ne"])),
        "Te": ("gdalt", np.array(md["te"])),
        "Ti": ("gdalt", np.array(md["ti"])),
        "maximum_range_m": ("gdalt", rng_m),
        "maximum_bulk_doppler": ("gdalt", vdopp_max),
    }

    data_dims = {"gdalt": len(rng_m)}
    dataset = simulate_data(data_dims, coorddict, paramvalues)

    # correct SNR for difference between the optimum bandwidth used
    # in the model and the fixed 50 kHz bandwidth used by the signal processing chain
    # this is Millstone MIDAS-W single pulse specific for this particular data interval.

    dataset["snr"].data = dataset["snr"].data * dataset["echo_bandwidth"] / 50e3

    # estimate the measurement error for the given Madrigal record,
    # scaling off the ratio of actual measurement time to model measurement time
    meas_est_err = est_err * (dataset["measurement_time"] / md["dut21"]) ** 0.5

    f, ax = pylab.subplots(1, 3, sharey=True)

    ax[0].plot(np.log10(md["ne"]), md["gdalt"], label="Ne")
    ax[0].legend(fontsize=8)
    ax[0].set_xlabel("m^-3")
    ax[0].set_ylabel("Altitude, km")
    ax[0].set_title("MHO Zenith")
    ax[0].grid(True)
    ax[0].set_xticks(np.arange(10, 13, 1))
    ax[1].plot(md["te"], md["gdalt"], "g", label="Te")
    ax[1].plot(md["ti"], md["gdalt"], "r", label="Ti")
    ax[1].set_title("480us")
    ax[1].legend(fontsize=8)
    ax[1].set_xticks(np.arange(1000, 5000, 1000))
    ax[1].set_xlabel("K")
    ax[1].grid(True)
    ax[2].plot(md["te"] / md["ti"], md["gdalt"], label="Tr")
    ax[2].legend(fontsize=8)
    ax[2].set_title(dt.isoformat())
    ax[2].set_xlabel("Temp Ratio")
    ax[2].grid(True)
    ax[2].set_xticks(np.arange(1, 4, 0.5))

    f.savefig("figures/is_sim_mho_misa_plasma_param.png")

    f = pylab.figure()
    ax = [pylab.gca()]

    ax[0].plot(md["snp3"], md["gdalt"], "b", label="MHO")
    ax[0].plot(dataset["snr"].values, md["gdalt"], "r", label="Model")
    ax[0].text(
        2,
        450,
        "Zenith TX Eff: %.2f\nZenith RX Eff: %.2f\nZenith Tsys: %.1f\nZenith Peak Pwr: %.1f MW"
        % (eff_tx, eff_rx, md["systmp"][0], md["power"][0] / 1e3),
    )
    ax[0].set_xlabel("SNR")
    ax[0].set_ylabel("Altitude, km")
    ax[0].set_title("MHO Zenith " + dt.isoformat() + " UTC")
    ax[0].grid(True)
    ax[0].legend(fontsize=8)

    f.savefig("figures/is_sim_mho_zenith_snr.png")

    f = pylab.figure()
    ax = [pylab.gca()]

    ax[0].semilogx(
        dataset["measurement_time"].values, md["gdalt"], label="1% Error Time"
    )
    ax[0].grid(True)
    ax[0].legend(fontsize=8)
    ax[0].set_xlabel("Modeled time for 1% accuracy, seconds")
    ax[0].set_ylabel("Altitude, km")
    ax[0].set_title("MHO MISA " + dt.isoformat() + " UTC")

    f.savefig("figures/is_sim_mho_misa_mtime.png")

    f = pylab.figure()
    ax = [pylab.gca()]

    ax[0].plot(100 * md["dti"] / md["ti"], md["gdalt"], "r", label="INSCAL dTi/Ti")
    ax[0].plot(100 * md["dte"] / md["te"], md["gdalt"], "g", label="INSCAL dTe/Te")
    ax[0].plot(100 * md["dtr"] / md["tr"], md["gdalt"], "b", label="INSCAL dTr/Tr")
    ax[0].plot(100 * meas_est_err, md["gdalt"], "m", label="Model Est accuracy")
    ax[0].text(1.75, 400, "Integration time: %.0f sec" % md["dut21"][0])
    ax[0].grid(True)
    ax[0].legend(loc="lower right", fontsize=8)
    ax[0].set_xlabel("Accuracy, percent")
    ax[0].set_ylabel("Altitude, km")
    ax[0].set_title("MHO MISA " + dt.isoformat() + " UTC")

    f.savefig("figures/is_sim_mho_misa_accuracy.png")

    f, ax = pylab.subplots(1, 2, sharey=True)

    ax[0].plot(md["snp3"] / dataset["snr"].values, md["gdalt"], label="MHO/Model")
    ax[0].set_xlabel("Measured/Model SNR Ratio")
    ax[0].set_ylabel("Altitude, km")
    ax[0].set_title("MHO MISA")
    ax[0].grid(True)
    ax[0].set_xlim(0.9, 1.1)
    ax[0].legend(fontsize=8)
    ax[1].plot(
        50e3 / dataset["echo_bandwidth"].values, md["gdalt"], label="Model BW / 50 kHz"
    )
    ax[1].set_xlabel("BW Ratio")
    ax[1].set_title(dt.isoformat())
    ax[1].grid(True)
    ax[1].legend(fontsize=8)

    f.savefig("figures/is_sim_mho_misa_ratio_bw.png")


def model_run_9():
    """
    Model run 9: Millstone Hill 46m MISA antenna and AMISR comparison
    IRI-2012 parameters for Te, Ti
    Keith Groves requested Ne = 1E12 m^-3
    30 deg elevation, 500 km range (265.69 km altitude)
    """

    t = datetime.datetime(2015, 10, 7, 18, 0, 0)
    # looker tells us with MHO = observing point,
    # geodetic observation coordinates are
    #  (46.35499751681627, -71.49199999995793, 265.6915002226515)
    #  for az = 0.0, el = 30.0, range = 500 km
    mho_az = 0.0
    mho_el = 30.0
    mho_range = 500.0e3
    (gdlat, gdlon, gdalt) = (46.35499751681627, -71.49199999995793, 265.6915002226515)
    m = iri2016py(t, gdlat, gdlon, gdalt, gdalt + 20, 10)
    print(m)

    # set identical plasma parameters
    ne = 1e12
    te = m["Te"][0]
    ti = m["Ti"][0]

    # pulse length sweep
    tpulse = np.linspace(1e-6, 100e-6, 100)

    # uncertainty
    est_err = 0.05

    # AMISR class ISR at 449 MHz (pulsed)
    pwr = 2.0e6
    fswp = 449e6
    duty = 0.1
    gn = 42.0 - 0.624  # modify for cosine scan angle behavior at 30 deg

    x_tsys = 10.0

    paramvalues = dict(
        peak_power_W=pwr,
        maximum_range_m=800e3,
        duty_cycle=duty,
        baud_length_s=mho_range * 2 / sc.c,
        excess_rx_noise_K=x_tsys,
        gain_tx_dB=gn,
        gain_rx_dB=gn,
        frequency_Hz=440e6,
        efficiency_tx=1.0,
        efficiency_rx=1.0,
        bandwidth_factor=1.0,
        tx_to_target_range_m=mho_range,
        target_to_rx_range_m=mho_range,
        Ne=ne,
        Te=te,
        Ti=ti,
        tsys_type="amisr",
        estimation_error_stdev=est_err,
        maximum_bulk_doppler=1500.0,
        monostatic=True,
        tx_target_rx_angle=0,
        bistatic_volume_factor=1.0,
        quick_bandwidth_estimate=True,
        calculate_plasma_parameter_errors=False,
    )
    paramvalues["O+"] = 1.0

    coorddict = {"baud_length_s": tpulse}

    data_dims = {"baud_length_s": len(tpulse)}
    datasetamisr = simulate_data(data_dims, coorddict, paramvalues)

    # MISA ISR at 440 MHz (pulsed)
    pwr = 2e6
    fswp = 440.2e6
    gn = 46.5
    eff_tx = 0.65  # empirically determined by SNR model-to-data match
    eff_rx = 0.65  # empirically determined by SNR model-to-data match
    duty = 0.06

    # set x_tsys to match a 165 K system temperature
    x_tsys = (
        165 - rx_temperature_model(fswp, "fixed_medium") - sky_temperature_model(fswp)
    )
    print("x_tsys: " + str(x_tsys))

    paramvalues = dict(
        peak_power_W=pwr,
        maximum_range_m=800e3,
        duty_cycle=duty,
        baud_length_s=mho_range * 2 / sc.c,
        excess_rx_noise_K=x_tsys,
        gain_tx_dB=gn,
        gain_rx_dB=gn,
        frequency_Hz=440e6,
        efficiency_tx=eff_tx,
        efficiency_rx=eff_rx,
        bandwidth_factor=1.0,
        tx_to_target_range_m=mho_range,
        target_to_rx_range_m=mho_range,
        Ne=ne,
        Te=te,
        Ti=ti,
        tsys_type="fixed_medium",
        estimation_error_stdev=est_err,
        maximum_bulk_doppler=1500.0,
        monostatic=True,
        tx_target_rx_angle=0,
        bistatic_volume_factor=1.0,
        quick_bandwidth_estimate=True,
        calculate_plasma_parameter_errors=False,
    )
    paramvalues["O+"] = 1.0
    datasetmho = simulate_data(data_dims, coorddict, paramvalues)

    f = pylab.figure()
    ax = [pylab.gca()]

    ax[0].loglog(
        datasetamisr["measurement_time"].values,
        tpulse * sc.c / 2e3,
        "r-x",
        label="AMISR",
    )
    ax[0].loglog(
        datasetmho["measurement_time"].values, tpulse * sc.c / 2e3, "g-o", label="MISA"
    )
    ax[0].grid(True)
    ax[0].text(
        1e2,
        10,
        "El = 30 deg\nAMISR boresight, fixed pointing assumed\nNo ionosphere beyond 1000 km range\nPower only\nCoding required for full spectrum\n(or a known Te/Ti)\n",
    )
    ax[0].legend(fontsize=9)
    ax[0].set_xlabel(
        "Measurement time to achieve %.0f%% accuracy, seconds" % (est_err * 100)
    )
    ax[0].set_ylabel("Range resolution, km")
    ax[0].set_title(
        "Range=%.0f km  Ne=%.1e m-3  Te=%.1f K  Ti=%.1f K"
        % (mho_range / 1e3, ne, te, ti)
    )

    f.savefig("figures/is_snr_amisr_vs_misa_groves_mtime.png")

    f = pylab.figure()
    ax = [pylab.gca()]

    ax[0].loglog(datasetamisr["snr"].values, tpulse * sc.c / 2e3, "r-x", label="AMISR")
    ax[0].loglog(datasetmho["snr"].values, tpulse * sc.c / 2e3, "g-o", label="MISA")
    ax[0].grid(True)
    ax[0].legend(fontsize=9)
    ax[0].set_xlabel("SNR")
    ax[0].set_ylabel("Range resolution, km")
    ax[0].set_title(
        "Range=%.0f km  Ne=%.1e m-3  Te=%.1f K  Ti=%.1f K"
        % (mho_range / 1e3, ne, te, ti)
    )

    f.savefig("figures/is_snr_amisr_vs_misa_groves_snr.png")


def plot_rxnoise():
    """
    Plot sky noise versus frequency.
    """
    freqs = np.linspace(10.0, 1000.0, num=1000) * 1e6
    rx_noise = rx_temperature_model(freqs) + np.zeros(len(freqs))
    sky_noise = sky_temperature_model(freqs)
    pylab.semilogy(freqs / 1e6, rx_noise + sky_noise, label="sky noise + rx noise")
    pylab.semilogy(freqs / 1e6, rx_noise, label="rx noise")
    pylab.semilogy(freqs / 1e6, sky_noise, label="sky noise")
    pylab.legend()
    pylab.xlabel("Frequency (MHz)")
    pylab.ylabel("Noise temperature (K)")
    pylab.savefig("figures/is_system_noise.png")


if __name__ == "__main__":

    print(version_str)
    plot_rxnoise()
    model_run_1()
    model_run_2()
    model_run_3()
    model_run_4()
    model_run_5()
    model_run_6()
    model_run_7()
    model_run_8()
    model_run_9()
