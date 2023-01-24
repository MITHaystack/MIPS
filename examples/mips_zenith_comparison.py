"""
    mips_zenith_comparison.py

    IS radar performance comparison to Millstone Zenith for the MIPS
    model.

"""

import datetime
import dateutil.parser
import numpy as np
import pylab
import madrigalWeb.madrigalWeb
import astropy.io.ascii
from mips import (
    simulate_data,
    simple_array,
    rx_temperature_model,
    sky_temperature_model,
)


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
    gn = 49.9

    tpulse = 480E-6

    duty = 0.053872
    eff_tx = 0.475  # empirically determined by SNR model-to-data match
    eff_rx = 0.475 # empirically determined by SNR model-to-data match
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

    snr_swp = np.zeros(len(md["gdalt"]))
    PApeak_swp = np.zeros(len(md["gdalt"]))
    PAavg_swp = np.zeros(len(md["gdalt"]))
    ld_ratio_swp = np.zeros(len(md["gdalt"]))
    bandwidth_swp = np.zeros(len(md["gdalt"]))
    mtime_swp = np.zeros(len(md["gdalt"]))

    paramvalues = dict(
        peak_power_W=pwr,
        n_bauds=1,
        pulse_length_s=tpulse,
        duty_cycle=duty,
        gain_tx_dB=gn,
        gain_rx_dB=gn,
        efficiency_tx=eff_tx,
        efficiency_rx=eff_tx,
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
        mtime_estimate_method="standard",
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
        33e3 / dataset["echo_bandwidth"].values, md["gdalt"], label="Model BW / 33 kHz"
    )
    ax[1].set_xlabel("BW Ratio")
    ax[1].set_title(dt.isoformat())
    ax[1].grid(True)
    ax[1].legend(fontsize=8)

    f.savefig("figures/is_sim_mho_zenith_ratio_bw.png")


if __name__ == "__main__":

    model_run_7()
