"""
    MIPS ISR Performance estimation library.

    This is the primary library for computation of IS radar performance based
    on a radar configuration, model ionosphere, and model configuration
    parameters. An IRI2016 interface is used to provide access to realistic
    ionospheric parameters.

"""

import scipy.constants as sc
import scipy.interpolate
import math
import numpy as np
import xarray
import urllib.request, urllib.parse, urllib.error
import urllib.request, urllib.error, urllib.parse
import datetime
import dateutil.parser
import astropy.io.ascii
import madrigalWeb.madrigalWeb
import ISRSpectrum
import iri2016.base as iri2016_base  # pip install - written by Michael Hirsch; avoiding name conflict with old method

from .coord import geographic_to_cartesian, antenna_to_cartesian, cartesian_to_geographic

version_str = "MIPS V2.2.1"

AMUDICT = {16: "O+", 30: "NO+", 28: "N2+", 32: "O2+", 14: "N+", 1: "H+", 4: "He+"}


def rx_temperature_model(frequency, mtype="fixed_medium"):
    """This is a simplified system temperature model, either fixed or
    variable based on a minicircuits ZX60-P103 LN+.
    it is possible to build a more optimized amplifier but this is
    representative of a typical E-phemt technology.

    Parameters
    ----------
    float
        Frequency in Hz

    Returns
    -------
    float
        Rx effective temperature, K
    """

    if mtype == "fixed_zero":
        return 0.0

    elif mtype == "fixed_cooled":

        return 20.0

    elif mtype == "fixed_vlow":

        return 31.0

    elif mtype == "fixed_low":

        return 40.0

    elif mtype == "fixed_medium":

        return 50.0

    elif mtype == "fixed_high":

        return 70.0

    elif mtype == "amisr":

        return 120.0

    elif mtype == "zx60-p103":

        # variable model
        # base model for the amplifier from data sheet
        lna_freq = np.array(
            [
                20.0e6,
                60.0e6,
                100.0e6,
                200.0e6,
                300.0e6,
                400.0e6,
                500.0e6,
                600.0e6,
                700.0e6,
                800.0e6,
                900.0e6,
                1000.0e6,
                1100e6,
                1200e6,
                1300e6,
                1400e6,
                1500e6,
                2000e6,
                2500e6,
                3000e6,
            ]
        )
        lna_nf = np.array(
            [
                2.68,
                1.18,
                0.82,
                0.57,
                0.81,
                0.60,
                0.67,
                0.66,
                0.71,
                0.79,
                0.97,
                0.86,
                0.91,
                1.03,
                1.07,
                1.17,
                1.31,
                1.35,
                1.40,
                1.80,
            ]
        )
        T0 = 290.0
        lna_ntemp = (10 ** (lna_nf / 10) - 1) * T0

        splintp = scipy.interpolate.interp1d(lna_freq, lna_ntemp, kind="cubic")

        return splintp(frequency)


def sky_temperature_model(frequency):
    """
    This model returns a modeled sky temperature for a given frequency.
    Valid from 100 to 500 MHz and ignores spatial variations. Possibly valid
    to HF frequencies but some observations have a different spectral index
    of 2.4 or so.

    Note the galaxy is neglected here as it is considered 'foreground'; this
    can provide quite a bit of noise (especially at low frequencies).

    From:
     Alan E. E. Rogers and Judd D. Bowman (2008), The Astronomical Journal, 136, 641
     http://dx.doi.org/10.1088/0004-6256/136/2/641

    Parameters
    ----------
    frequency : float
        Frequency in Hz

    Returns
    -------
    float
        Sky effective temperature, K
    """
    t150 = 283.2
    beta = 2.5  # spectral index
    tcmb = 2.725  # cosmic microwave background

    return t150 * (frequency / 150.0e6) ** (-beta) + tcmb


def simple_array(N, pwr):
    """Model a simple phased array.
       NOTE: this is just for array element count sweeps. Use a real
       antenna simulation for detailed design.

    Parameters
    ----------
    N : float
        element count
    pwr : float
        peak power per element, W  (6 dB per-element gain assumed)

    Returns
    -------
    power : float
        total peak power in boresight direction, W
    gain : float
        total gain in boresight direction, dB
    """
    return (N * pwr, 10.0 * np.log10(N) + 6.0)


def iri2016py(dt, gdlat, gdlon, gdalt_start, gdalt_end, gdalt_step):
    """
    Return the IRI 2016 output using the python module iri2016 written by
    Michael Hirsch.

        Note: to run this method for recent dates, you will need to update the files
        apf107.dat and ig_rz.dat from http://irimodel.org/indices/ and overwrite the
        old versions in site-packages/iri2016/data/index

    Parameters
    ----------
    dt : datetime.datetime
        UNIVERSAL time desired
    gdlat : float
        geodetic latitude, deg
    gdlon : float
        geodetic longitude, deg
    gdalt_start : float
        start geodetic altitude, km
    gdalt_end : float
        end geodetic altitude, km
    gdalt_step : float
        step geodetic altitude, km

    Returns
    -------
    <xarray.Dataset>
        Dimensions:         (alt_km, time)
        Coordinates:
          * time            (time) datetime64[ns]
          * alt_km          (alt_km) float64
            glat            int64
            glon            float64
        Data variables:
            ne              (alt_km) float64
            Tn              (alt_km) float64
            Ti              (alt_km) float64
            Te              (alt_km) float64
            nO+             (alt_km) float64
            nH+             (alt_km) float64
            nHe+            (alt_km) float64
            nO2+            (alt_km) float64
            nNO+            (alt_km) float64
            nCI             (alt_km) float64
            nN+             (alt_km) float64
            NmF2            (time) float64
            hmF2            (time) float64
            NmF1            (time) float64
            hmF1            (time) float64
            NmE             (time) float64
            hmE             (time) float64
            TEC             (time) float64
            EqVertIonDrift  (time) float64
            foF2            (time) float64
        Attributes:
            f107:     [67.9130402]
            ap:       [-11.0]
    """

    alt_km_range = (gdalt_start, gdalt_end, gdalt_step)
    return iri2016_base.IRI(dt, alt_km_range, gdlat, gdlon)


def iri2016(dt, gdlat, gdlon, gdalt_start, gdalt_end, gdalt_step):

    """
    Return the IRI-2016 model output for the specified 4D coordinates
    through a call to the CCMC web interface.

    Parameters
    ----------
    dt : datetime.datetime
        UNIVERSAL time desired
    gdlat : float
        geodetic latitude, deg
    gdlon : float
        geodetic longitude, deg
    gdalt_start : float
        start geodetic altitude, km
    gdalt_end : float
        end geodetic altitude, km
    gdalt_step : float
        step geodetic altitude, km

    Returns
    -------
    astropy.io.ascii.Table with the following data in each column:
        1. Electron density (Ne) in m^-3.
        2. Ratio of Ne to the F2 peak density (NmF2).
        3. Neutral temperature (Tn) in K.
        4. Ion temperature (Ti) in K.
        5. Electron temperature (Te) in K.
        6. Atomic oxygen ions (O+) percentage.
        7. Atomic hydrogen ions (H+) percentage.
        8. Atomic helium ions (He+) percentage.
        9. Molecular oxygen ions (02+) percentage.
        10. Nitric oxide ions (NO+) percentage.
        11. Cluster ions percentage.
        12. Atomic nitrogen ions (N+) percentage.
        13. Total electron content (TEC) in 10^16 m^-2.
        14. TEC top percentage.
        15. Height of the F2 peak (hmF2) in km.
        16. Height of the F1 peak (hmF1) in km.
        17. Height of the E peak (hmE) in km.
        18. Height of the D peak (hmD) in km.
        19. Density of the F2 peak (NmF2) in m^-3.
        20. Density of the F1 peak (NmF1) in m^-3.
        21. Density of the E peak (NmE) in m^-3.
        22. Density of the D peak (NmD) in m^-3.
        23. Propagation factor M(3000)F2.
        24. Bottomside thickness (B0) in km.
        25. Bottomside shape (B1).
        26. E-valley width in km.
        27. E-valley depth (Nmin/NmE) in km.
        28. F2 plasma frequency (foF2) in MHz.
        29. F1 plasma frequency (foF1) in MHz.
        30. E plasma frequency (foE) in MHz.
        31. D plasma frequency (foD) in MHz.
        32. Equatorial vertical ion drift in m/s.
        33. Ratio of foF2 storm to foF2 quiet.
        34. CGM latitude of auroral oval boundary.
        35. F1 probability.
        36. Ratio of foE storm to foE quiet.
        37. Spread F probability.
        38. 12-month running mean of sunspot number Rz12 used by the model.
        39. Ionospheric index IG12 used by the model.
        40. Daily solar radio flux F107D used by the model.
        41. 81 day solar radio flux F107_81D used by the model.
        42. 3 hour ap index used by the model.
        43. Daily ap index used by the model.
        44. 3 hour Kp index used by the model.

    A value of -1 for any output indicates that the parameter is not available for the specified range. TEC = -1 means you have not entered an upper boundary height for TEC_HMAX.

    NB: as of summer 2018, the VITMO web interface to IRI-2012 was shut down due to repeated overloading.
    NB: IRI-2016 is back online as of spring 2020 at CCMC
    """

    if gdlon < 0:
        gdlon += 360
    url = "http://ccmc.gsfc.nasa.gov/cgi-bin/modelweb/models/vitmo_model.cgi"
    values = {
        "model": "iri2016",
        "year": "%i"
        % dt.year,  # note this is UNIVERSAL time in the default call! (tested on web site)
        "month": "%i" % dt.month,
        "day": "%i" % dt.day,
        "hour": "%.4f" % (dt.hour + dt.minute / 60.0 + dt.second / 3600.0),
        "geo_flag": 0,  # geographic coordinates
        "time_flag": 0,  # Universal time
        "latitude": gdlat,
        "longitude": gdlon,
        "height": gdalt_start,
        "profile": 1,  # generate altitude profile
        "start": gdalt_start,
        "stop": gdalt_end,
        "step": gdalt_step,
        "sun_n": "",  # Sunspot number, R12 (0. - 400.)
        "ion_n": "",  # Ionospheric index, IG12 (-50. - 400.)
        "radio_f": "",  # F10.7 radio flux, daily (0. - 400.)
        "radio_f81": "",  # F10.7 radio flux, 81-day (0. - 400.)
        "htec_max": "",  # Electron content: Upper boundary (50. - 2000. km)
        "ne_top": "",  # Ne topside: use NeQuick default
        "imap": 0,  # Ne F-peak: 0 = URSI model
        "ffof2": 0,  # F-peak storm model: 0 = on
        "hhmf2": 0,  # F-peak height: 0 = AMTB2013
        "ib0": 2,  # Bottomside thickness: 2 = ABT-2009
        "probab": 0,  # F1 occurrence probability: 0 = Scotto-1997 no L
        "fauroralb": 1,  # Auroral boundaries: 1 = off
        "ffoE": 0,  # E-peak auroral storm model: 0 = off
        "dreg": 0,  # D-region model
        "tset": 0,  # Te topside: 0 = TBT2012+SA
        "icomp": 0,  # Ion composition: 0 = RBV10/TBT15
        "nmf2": 0,  # F2 peak density (NmF2): 0 = model determines
        "hmf2": 0,  # F2 peak height (hmf2): 0 = model determines
        "user_nme": 0,  # E peak density (NmE): 0 = model determines
        "user_hme": 0,  # E peak density (hmE): 0 = model determines
        "user_B0": 0,
    }  # Bottomside thickness (B0): 0 = model determines
    parnames = [
        "ne",
        "ratio_ne_nmF2",
        "tn",
        "ti",
        "te",
        "pct_o+",
        "pct_h+",
        "pct_he+",
        "pct_o2+",
        "pct_no+",
        "pct_cluster",
        "pct_n+",
        "tec",
        "tec_top_pct",
        "hmF2",
        "hmF1",
        "hmE",
        "hmD",
        "nmF2",
        "nmF1",
        "nmE",
        "nmD",
        "m3000F2",
        "B0",
        "B1",
        "E_valley_width",
        "E_valley_depth",
        "f0F2",
        "f0F1",
        "f0E",
        "f0D",
        "eq_vertical_drift",
        "f0F2_storm_to_quiet",
        "oval_cgm_lat",
        "prob_F1",
        "f0E_storm_to_quiet",
        "prob_ESF",
        "RZ12",
        "IG12",
        "F107D",
        "F107_81D",
        "Ap3",
        "Ap",
        "Kp",
    ]
    data = urllib.parse.urlencode(values)
    for vk in range(17, 61):
        data += "&vars=%i" % vk
    data = urllib.request.urlopen(url + "?" + data)
    d = data.read()
    ds = d.split("<pre>")[1].split("</pre>")[0]
    indx = ds.find("44 3-h_kp")
    ds = ds[indx:]
    dlines = ds.split("\n")[3:]
    a = astropy.io.ascii.read(dlines, names=parnames)
    a["gdalt"] = np.arange(gdalt_start, gdalt_end + gdalt_step, gdalt_step)
    a["gdlat"] = gdlat * np.ones(len(a), np.float)
    a["gdlon"] = gdlon * np.ones(len(a), np.float)
    return a


def speccheck(ion_species, pfunc=print):
    """Used for assert statments to check if species is valid.

    Parameters
    ----------
    ion_species : list
        Names of ion species used. Includes 'O+', 'NO+', 'N2+', 'O2+', 'N+', 'H+', 'He+'.

    Returns
    -------
    allgood : boolean
        True if all list members are valid species.
    """

    maslist = ["O+", "NO+", "N2+", "O2+", "N+", "H+", "He+"]

    allgood = True
    for ion in ion_species:
        if not ion in maslist:
            pfunc("{} not a recognized ion species".format(ion))
            allgood = False
    return allgood


def sweep_iri_along_beam(dt, latitude, longitude, altitude, az_dir, el_dir, start_range, end_range, delta_range):
    """ Sweep the IRI model along a radar beam and compute the model outputs.

        dt, datetime for the model run time
        latitude, radar latitude
        longitude, radar longitude
        altitude, radar altitude
        az_dir, radar pointing direction in azimuth
        el_dir, radar pointing direction in elevation angle
        start_range, range along beam to start sweep in km
        end_range, range along beam to end sweep in km
        delta_range, range step in km

        return xarray model evaluation versus range with latitude, longitude, and altitude evaluated

    """

    # convert to ECF (x,y,z) start point
    radar_x, radar_y = geographic_to_cartesian(longitude, latitude, {'lon_0':longitude,'lat_0':latitude,'proj':'pyart_aeqd'})
    radar_z = altitude # in meters
    # compute cartesian coordinates
    ranges = np.arange(start_range, end_range, delta_range)
    # ranges are in km
    beam_x, beam_y, beam_z = antenna_to_cartesian(ranges/1000.0, az_dir, el_dir)
    # note result is in meters

    # offset to ECF
    sweep_x = radar_x + beam_x
    sweep_y = radar_y + beam_y
    sweep_z = radar_z + beam_z

    # convert ECF to geodetic lat, lon, alt
    sweep_lon, sweep_lat = cartesian_to_geographic(sweep_x, sweep_y, {'lon_0':longitude,'lat_0':latitude,'proj':'pyart_aeqd'})
    sweep_alt = sweep_z

    #print(sweep_lat)
    #print(sweep_lon)
    #print(ranges)
    #print(sweep_alt)

    # Sweep IRI model in altitude along beam locations
    ionosphere = None
    for idx, alt in enumerate(sweep_alt):
        iri = iri2016py(dt,sweep_lat[idx],sweep_lon[idx],alt/1e3,alt/1e3,1.0)
        #print(iri)
        if ionosphere is None:
            ionosphere = iri
        else:
            ionosphere = xarray.concat((ionosphere, iri), dim="alt_km")

    return ionosphere, ranges


def is_bandwidth_estimate(
    frequency_Hz,
    Ne,
    Te,
    Ti,
    ion_species,
    ion_fraction,
    tx_target_rx_angle,
    maximum_bulk_doppler,
    bandwidth_factor,
    quick_estimate_mode=True,
    pfunc=print
):

    """
    is_bandwidth_estimate() calculates an estimate of the incoherent scatter spectral bandwidth including bulk Doppler effects.

    Parameters
    ----------
    frequency_Hz : float
        Center frequency of radar in Hz.
    Ne : float
        Electron density at the given altitude, m^-3
    Te : float
        Electron temperature in K.
    Ti : float
        Ion temperature in K.
    ion_species : list
        Names of ion species used. Includes 'O+', 'NO+', 'N2+', 'O2+', 'N+', 'H+', 'He+'.
    ion_fraction : list
        Fractions of each ion species. Sum must equal 1.
    maximum_bulk_doppler :
        maximum line-of-sight Doppler velocity expected, m/s
    tx_target_rx_angle : float
        angle between transmitter target and target observer lines, deg
    bandwidth_factor : float
        System Bandwidth factor (multiplies IS bandwidth; post processing), unitless
    quick_estimate_mode : bool
        Default = True: use a first principles quick estimate; False = use a full IS theoretical spectral calculation (NB: magnetic field effects are disabled)

    Returns
    -------
    line_shift : float

    line_shift : float
        normal radar doppler shift for the ion acoustic speed and effective scattering wavelength, Hz
    bandwidth : float
        total spectral bandwidth including bulk Doppler  shift, Hz
    h_lambda_inv : float
        Inverse of wavelength scaled by the bistatic angle in 1/m. Reciprical used to avoid divide by zero errors from bistatic angle.
    """

    # sanity checks
    assert (np.array(ion_fraction) >= 0.0).all(), "Can't have negative ion fractions."
    assert ISRSpectrum.ioncheck(ion_species), "Invalid ion species"
    assert sum(ion_fraction) < 1.1 and sum(ion_fraction) > 0.9, "Ion fractions must equal 1."

    # Estimate from first principles without an actual IS theoretical calculation.

    # Compute ion line bandwidth assuming a single ion species, and double for two
    # sided line doppler shift. This corresponds to the minimum required receiver bandwidth and
    # with modern signal processing this can be achieved with digital filters.
    # Also apply a radar specific, user requested bandwidth factor (bwf). This can account for fixed
    # capability receivers or factors for bulk doppler shift due to drifts.

    # radar wavelength
    wavelength = sc.c / frequency_Hz

    # ion-acoustic velocity
    i_ion = np.argmax(ion_fraction)
    ion_mass = ISRSpectrum.getionmass(ion_species[i_ion])
    v_ion_acoustic = (((sc.k * Ti) / (ion_mass * sc.m_p)) * (1.0 + Te / Ti)) ** 0.5

    # Doppler shift due to i-a velocity
    # take into account bistatic Bragg length
    # h_lambda = wavelength / (
    #     2.0 * np.cos((np.pi * tx_target_rx_angle / 180.0) / 2.0)
    # )

    #
    h_lambda_inv = 2.0 * np.cos((np.pi * tx_target_rx_angle / 180.0) / 2.0) / wavelength
    # normal radar doppler shift for the ion acoustic speed and effective scattering wavelength
    line_shift = np.array(2.0 * v_ion_acoustic * h_lambda_inv)

    # flow shift - for bulk line of sight Doppler shifts
    flow_shift = maximum_bulk_doppler * h_lambda_inv

    if not quick_estimate_mode:
        # more realistic estimate using two-ion incoherent scatter spectral calculation.
        try:

            # spectrum becomes more narrow due to bistatic k-vector
            bistatic_effective_freq = frequency_Hz * ((wavelength * 0.5) * h_lambda_inv)
            iss = ISRSpectrum.Specinit(
                centerFrequency=bistatic_effective_freq,
                nspec=4096,
                sampfreq=128 * line_shift,
                dFlag=False,
            )
            omega, spec = iss.getspecsimple(
                Ne, Te, Ti, ion_species, ion_fraction, vel=0.0, rcsflag=False
            )

            spec = spec / spec.max()

            # fix problems with non-zero spectra
            indx = np.where(spec < 0)[0]
            spec[indx] = 0.0
            # working backwards from high frequency end of the spectrum,
            # find the place where the spectrum is 5% of the peak value
            indx = np.nonzero(spec == spec.max())[0][-1]
            spec_intp = scipy.interpolate.interp1d(
                spec[indx:][::-1], omega[indx:][::-1]
            )
            line_shift = spec_intp(0.05)

        except:

            #raise ValueError(
            #    "Could not find bandwidth in realistic spectral case.  Try increasing sampling frequency."
            #)
            pfunc("Could not find bandwidth in realistic spectral case. Default to simple spectral case.")

    # common calculations regardless of mode

    # total Doppler shift due to scatter broadening plus line of sight velocity
    radar_doppler_shift = line_shift + flow_shift

    # effective bandwidth needed (up and downshifted, positive and negative flow),
    # including user multiplication factor bwf
    bandwidth = 2.0 * radar_doppler_shift * bandwidth_factor

    return (line_shift, bandwidth, h_lambda_inv)


def is_calculate_spectrum(velocity_ms, frequency_Hz, Ti, Te, Ne, ion_species, ion_fraction):
    """
    is spectrum to obtain linearized errors

    Parameters
    ----------
    velocity_ms : float
        Velocity in m/s
    frequency_Hz : float
        Center frequency of radar in Hz.
    Ti : float
        Ion temperature in K.
    Te : float
        Electron temperature in K.
    Ne : float
        Electron density at the given altitude, m^-3
    ion_species : list
        Names of ion species used. Includes 'O+', 'NO+', 'N2+', 'O2+', 'N+', 'H+', 'He+'.
    ion_fraction : list
        Fractions of each ion species. Sum must equal 1.

    Returns
    -------
    spectrum_freqs : array_like
        Frequency sampling of the spectrum in Hz.
    spectrum : array_like
        IS spectrum with highest value normalized to 1.
    """
    doppler_Hz = 2.0 * velocity_ms * frequency_Hz / sc.c
    # nspec is ballpark estimate of coding but should actually reflect the bandwidth, pulse length, and coding.
    # we need to improve this in the future...
    # iss=ISSpectrum.ISSpectrum(centerFrequency=frequency_Hz/1e6,nspec=32)
    iss = ISRSpectrum.Specinit(
        centerFrequency=frequency_Hz, nspec=64, sampfreq=50e3, dFlag=False
    )
    spectrum_freqs, spectrum = iss.getspecsimple(
        Ne, Te, Ti, ion_species, ion_fraction, vel=velocity_ms, rcsflag=False
    )

    spectrum = spectrum / spectrum.max()

    return (spectrum_freqs, spectrum)


# Linearized errors for plasma parameter fit
def is_calculate_plasma_parameter_errors(
    velocity_ms,
    frequency_Hz,
    Ti,
    Te,
    Ne,
    ion_species,
    ion_fraction,
    snr,
    estimation_error_stdev,
    debug_print=False,
):
    """
    Compute is plasma parameter errors

    Parameters
    ----------
    velocity_ms : float
        Velocity in m/s
    frequency_Hz : float
        Center frequency of radar in Hz.
    Ti : float
        Ion temperature in K.
    Te : float
        Electron temperature in K.
    Ne : float
        Electron density at the given altitude, m^-3
    ion_species : list
        Names of ion species used. Includes 'O+', 'NO+', 'N2+', 'O2+', 'N+', 'H+', 'He+'.
    ion_fraction : list
        Fractions of each ion species. Sum must equal 1.
    snr : float
        signal-to-noise ratio linear scale, unitless
    estimation_error_stdev : float
        estimation error for speed computation in fraction of final plasma parameter (e.g. 0.1 -> 10 percent error), unitless
    debug_print : bool
        Debug print outs.

    Returns
    -------
    dNe : float
        Error in electron density m^-3
    dTi : float
        Error in ion temperature in K.
    dTe : float
        Error in electron temperature in K.
    dV : float
        Error in velocity in m/s.
    """
    Ne0 = Ne
    Te0 = Te
    Ti0 = Ti
    vel0 = velocity_ms

    # step of numerical estimation of Jacobian
    df = np.array([0.01 * Ne0, 0.01 * Ti0, 0.01 * Te0, 10.0])

    (freqs, spec0) = is_calculate_spectrum(
        vel0, frequency_Hz, Ti0, Te0, Ne0, ion_species, ion_fraction
    )

    spec_scale = np.max(spec0)
    n_spec = len(spec0)

    # Jacobian for four parameter fit (ne, ti, te, vel)
    J = np.zeros([n_spec, 4])

    pars0 = np.array([Ne0, Ti0, Te0, vel0])
    for i in range(4):
        pars1 = np.copy(pars0)
        pars1[i] = pars1[i] + df[i]
        Ne = pars1[0]
        Ti = pars1[1]
        Te = pars1[2]
        vel = pars1[3]
        (freqs, spec1) = is_calculate_spectrum(
            vel, frequency_Hz, Ti, Te, Ne, ion_species, ion_fraction
        )
        spec1 = spec1 * (Ne / Ne0)
        J[:, i] = (spec1 / spec_scale - spec0 / spec_scale) / df[i]

    sigma = np.zeros(n_spec)
    sigma[:] = estimation_error_stdev**2.0
    Sigma_inv = np.diag(1.0 / sigma)
    try:
        Sigma_post = np.linalg.inv(np.dot(np.dot(np.transpose(J), Sigma_inv), J))
    except:
        Sigma_post = np.nan * np.ones((4, 4))

    if debug_print:

        corr_mat = np.copy(Sigma_post)
        for i in range(4):
            for j in range(4):
                corr_mat[i, j] = Sigma_post[i, j] / (
                    np.sqrt(Sigma_post[i, i]) * np.sqrt(Sigma_post[j, j])
                )
        print("Correlation matrix")
        print(corr_mat)
        print("Diagonal 2-sigma errors")
        print((2.0 * np.sqrt(np.diag(Sigma_post))))
        print(
            (
                "dNe/Ne %1.2f %% dTi %1.2f K dTe %1.2f K dv %1.2f m/s"
                % (
                    np.sqrt(Sigma_post[0, 0]) / Ne0,
                    np.sqrt(Sigma_post[1, 1]),
                    np.sqrt(Sigma_post[2, 2]),
                    np.sqrt(Sigma_post[3, 3]),
                )
            )
        )

    return {
        "dNe": np.sqrt(Sigma_post[0, 0]) / Ne0,
        "dTi": np.sqrt(Sigma_post[1, 1]),
        "dTe": np.sqrt(Sigma_post[2, 2]),
        "dV": np.sqrt(Sigma_post[3, 3]),
    }


def is_snr(
    peak_power_W,
    maximum_range_m,
    pulse_length_ns,
    n_bauds,
    duty_cycle,
    gain_tx_dB,
    gain_rx_dB,
    efficiency_tx,
    efficiency_rx,
    frequency_Hz,
    bandwidth_factor,
    tx_to_target_range_m,
    target_to_rx_range_m,
    Ne,
    Te,
    Ti,
    excess_rx_noise_K,
    tsys_type,
    estimation_error_stdev,
    maximum_bulk_doppler,
    monostatic,
    tx_target_rx_angle,
    bistatic_volume_factor,
    ion_species,
    ion_fraction,
    quick_bandwidth_estimate,
    calculate_plasma_parameter_errors,
    mtime_estimate_method,
    pfunc=print,
):

    """SNR estimation equation

    Parameters
    ----------
    peak_power_W : Float
        Average TX Power, W
    maximum_range_m : Float
        maximum range that produces measureable signal, m
    pulse_length_ns : int
        radar transmit pulse length, nanoseconds
    n_bauds : int
        Number of bauds in pulse
    duty_cycle float : float
        estimation duty cycle, unitless
    gain_tx_dB : float
        TX gain, dB
    gain_tx_dB : float
        RX gain, dB
    efficiency_tx : float
        Efficiency of TX aperture, unitless
    efficiency_rx : float
        Efficiency of RX aperture, unitless
    frequency_Hz : float
        TX and RX frequency, Hz
    bandwidth_factor : float
        System Bandwidth factor (multiplies IS bandwidth; post processing), unitless
    tx_to_target_range_m : float
        range from tx to target, m
    target_to_rx_range_m : float
        range from target to rx, m
    Ne : float
        electron density at the given altitude, m^-3
    Te : float
        electron temperature, K
    Ti : float
        ion temperature, K
    excess_rx_noise_K : float
        extra Tsys to account for system specific components (e.g. T/R switch, calibrator, diode protection, combiner network losses, etc)., K
    estimation_error_stdev : float
        estimation error for speed computation in fraction of final plasma parameter (e.g. 0.1 -> 10 percent error), unitless
    maximum_bulk_doppler : float
        maximum line-of-sight Doppler velocity expected, m/s
    tx_target_rx_angle : float
        angle between transmitter target and target observer lines, deg
    bistatic_volume_factor : float
        the fraction of bistatic volume compared with monostatic measurement volume (V_bistatic/V_monostatic), unitless
    ion_species : list
        Names of ion species used. Includes 'O+', 'NO+', 'N2+', 'O2+', 'N+', 'H+', 'He+'.
    ion_fraction : list
        Fractions of each ion species. Sum must equal 1.
    quick_bandwidth_estimate : bool
        if True (default), a single ion is assumed with AMU = ion mass 1. If False, a full 2-ion IS spectral calculation is used for bandwidth estimation.
    mtime_estimate_method : str
        String to determine measurement time estimation method. Can be std, mracf.

    Returns
    -------
    snr : float
        signal-to-noise ratio linear scale, unitless
    power_aperture_to_temperature : float
        peak power aperture to temperature ratio, MW m^2 / K
    avg_power_aperture_to_temperature : float
        average power aperture to temperature ratio, MW m^2 / K
    wavelength_to_debye_length_ratio : float
        ratio of radar wavelength to Debye length for provided plasma parameters and radar frequency, unitless
    echo_bandwidth : float
        estimated signal bandwidth, Hz
    measurement_time : float
        measurement time required to achieve the requested statistical estimation error, s
    plasma_parameter_errors : tuple
        errors of the plasma parameters dNe, dTi, dTe, dV (optional)

    (snr, power_aperture_to_temperature, avg_power_aperture_to_temperature,
     wavelength_to_debye_length_ratio, echo_bandwidth, measurement_time)

    Previous default parameters:
        peak_power_W=1e6,
        maximum_range_m=600e3,
        pulse_length_ns=500000,
        duty_cycle=0.05,
        gain_tx_dB=42.0,
        gain_rx_dB=42.0,
        efficiency_tx=1.0,
        efficiency_rx=1.0,
        frequency_Hz=440e6,
        bandwidth_factor=1.0,
        tx_to_target_range_m=300e3,
        target_to_rx_range_m=300e3,
        Ne=4.2e11,
        Te=1000.0,
        Ti=500.0,
        excess_rx_noise_K=0.0,
        tsys_type='fixed_medium',
        estimation_error_stdev=0.05,
        maximum_bulk_doppler=1500.0,
        monostatic=True,
        tx_target_rx_angle=0.0,
        bistatic_volume_factor=1.0,
        ion_species=['O+'],
        ion_fraction = [1.],
        quick_bandwidth_estimate=True,
        calculate_plasma_parameter_errors=True
    """

    baud_length_s = 1e-9 * float(pulse_length_ns) / n_bauds
    p_length = pulse_length_ns * 1e-9

    # radar wavelength
    wavelength = sc.c / frequency_Hz

    # absolute TX and RX gain
    gain_tx = 10 ** (gain_tx_dB / 10.0)
    gain_rx = 10 ** (gain_rx_dB / 10.0)

    # RX and TX beamwidth in degrees

    # Start with a sphere having 41253 square degrees (i.e 4pi * (57.29577951 deg/rad)^2)
    #
    # gain = area of sphere / area of beam pattern calculation

    # With a circular beam shape being assumed. Approximate beam pattern from
    # gain value as being at 0.707 for the half power point from the gain
    # maximum. Then apply a loss of the energy to the sidelobes from the main beam.
    #
    # The result is about 45% of the radiated power going through the half power
    # beamwidth. This is good for parabolic antennas and is about the best we
    # can estimate without an antenna pattern model. Overall this
    # is a bit on the conservative side for many some types of apertures. Array
    # apertures do better with taper but that trades off having additional T/R
    # elements relative to the far field gain of the aperture.

    # Note that this is separate from the efficiency of the aperture due to other
    # factors such as blockage.

    rx_beamwidth = (27000.0 / gain_rx) ** 0.5
    tx_beamwidth = (27000.0 / gain_tx) ** 0.5

    # RX and TX beamwidth in radians
    rad_rx_beamwidth = rx_beamwidth * math.pi / 180.0
    rad_tx_beamwidth = tx_beamwidth * math.pi / 180.0

    # range resolution is determined by baud length
    range_resolution_m = sc.c / 2.0 * baud_length_s


    # baud_gain for measurement time
    # J. Stamm, J. Vierinen, J. M. Urco, B. Gustavsson, and J. L. Chau, “Radar imaging with EISCAT 3D,” Annales Geophysicae, vol. 39, no. 1, pp. 119–134, Feb. 2021, doi: 10.5194/angeo-39-119-2021.
    # Note this is likely only true for voltage domain codes. Power domain codes take multiple cycles per measurement.
    # set to unity for the moment until we work out how to handle this. Things like
    # alternating codes go as 1/sqrt(n_bauds) due to incoherent averaging. So this
    # produces very optimistic measurement speed without it being correct.
    #
    if n_bauds < 2:
        baud_gain = 1
    else:
        #baud_gain = n_bauds * (n_bauds - 1) / 2.0
        baud_gain = 1

    # handle mis-matched beams
    if rad_tx_beamwidth <= rad_rx_beamwidth:

        # Illuminated volume (approximation from Peebles, 5.7-19)
        # if we are modeling a bi-static path, include a penalty factor for mismatched volumes
        volume = (
            bistatic_volume_factor
            * math.pi
            * tx_to_target_range_m**2
            * rad_tx_beamwidth**2
            * range_resolution_m
            / (16.0 * math.log(2.0)*(np.cos(np.deg2rad(tx_target_rx_angle)))**2.0)
        )

        # wider RX beam will increase noise collected relative to signal, lowering SNR
        noise_scaling = rad_rx_beamwidth / rad_tx_beamwidth

    else:
        # Illuminated volume (approximation from Peebles, 5.7-19)
        # if we are modeling a bi-static path, include a penalty factor for mismatched volumes
        volume = (
            bistatic_volume_factor
            * math.pi
            * tx_to_target_range_m**2
            * rad_rx_beamwidth**2
            * range_resolution_m
            / (16.0 * math.log(2.0)*(np.cos(np.deg2rad(tx_target_rx_angle)))**2.0)
        )

        # wider TX beam than RX is just diluted by the volume above but collects no extra noise
        noise_scaling = 1.0



    # fundamental electron radius and scattering cross-section
    electron_radius = sc.e**2.0 * sc.mu_0 / (4.0 * sc.pi * sc.m_e)

    # From Beynon and Williams 1978.
    # notice that bistatic depolarization effect is taken into account at a later stage in power
    # lost from a circularly polarized transmission on bistatic receive
    electron_xsection = 4.0 * sc.pi * electron_radius**2.0

    # LNA system temperature, K
    lna_temperature = rx_temperature_model(frequency_Hz, tsys_type)
    # Sky temperature, K ; scale for mis-matched beams
    sky_temperature = sky_temperature_model(frequency_Hz) * noise_scaling

    # Total effective system temperature including user addon, K
    system_temperature = lna_temperature + sky_temperature + excess_rx_noise_K

    # Peak power aperture to temperature ratio, MW m^2 / K
    power_aperture_to_temperature = (
        peak_power_W
        * gain_tx
        * wavelength**2
        / (4 * sc.pi * 1e6 * system_temperature)
    )
    # Average power aperture to temperature ratio, MW m^2 / K
    avg_power_aperture_to_temperature = duty_cycle * power_aperture_to_temperature

    # Debye length, m
    debye_length_m = (sc.epsilon_0 * sc.k * Te / (Ne * sc.e**2)) ** 0.5
    # ratio of radar wavelength to Debye length for provided plasma
    # parameters and radar frequency, unitless
    wavelength_to_debye_length_ratio = wavelength / debye_length_m

    # unit ISR cross-section per electron, m^2
    # From Beynon and Williams 1978, p. 919, take into account debye length effects
    alpha = 4.0 * sc.pi * debye_length_m / wavelength
    unit_xsection = electron_xsection * (
        1.0
        - (1.0 + alpha**2.0) ** (-1.0)
        + ((1.0 + alpha**2.0) * (1.0 + alpha**2.0 + Te / Ti)) ** -1.0
    )



    if np.min(wavelength_to_debye_length_ratio) < 1.0:
        wdval = np.min(wavelength_to_debye_length_ratio)

        pfunc(
            "Warning: Wavelength to Debye length ratio %f is less than 1. Ion line expression not valid. Applicability of the ISR method to this plasma dubious."
            % (wavelength_to_debye_length_ratio)
        )
        pfunc(
            "wavelength ",
            wavelength,
            " debye length ",
            debye_length_m,
            " Te ",
            Te,
            " Ne ",
            Ne,
        )

        # raising an exception here breaks all kinds of sweeps over regular grids

        #    % (wdval)")
        # raise ValueError(
        #    "Wavelength to Debye length ratio %f is less than 5. Ion line expression not valid. Applicability of the ISR method to this plasma dubious."
        #    % (wdval)
        # )

    # Compute ion line bandwidth.  Use estimation mode requested by user - either 1 or 2 ion model.

    (line_shift, bandwidth, h_lambda) = is_bandwidth_estimate(
        frequency_Hz,
        Ne,
        Te,
        Ti,
        ion_species,
        ion_fraction,
        tx_target_rx_angle,
        maximum_bulk_doppler,
        bandwidth_factor,
        quick_estimate_mode=quick_bandwidth_estimate,
    )

    # Final SNR computation. Bistatic radar equation, taking into account polarization effects

    # HACK bistatic depolarization needs to be treated to deal with different types of polariation. Probably the best way is to do a scattering matrix.
    # bistatic depolarization, assuming vertical transmit polarization
    #polarization_loss = 1.0 - np.sin(sc.pi * tx_target_rx_angle / 180.0) ** 2.0
    #polarization_loss = 1.0
    # circular polarization case
    polarization_loss = 1.0 -  0.5*(np.sin(np.deg2rad(tx_target_rx_angle)))**2.0

    # radar cross section
    rcs = polarization_loss * unit_xsection * Ne * volume

    # divide signal and noise into separate components to address self-noise
    # signal power
    s = (
        polarization_loss
        * peak_power_W
        * efficiency_tx
        * efficiency_tx
        * gain_tx
        * gain_rx
        * wavelength**2.0
        * rcs
        / (
            (4.0 * sc.pi) ** 3.0
            * (tx_to_target_range_m**2.0 * target_to_rx_range_m**2.0)
        )
    )
    n = sc.k * system_temperature * bandwidth
    snr = s / n


    # compute the number of independent samples for the integration time
    # we are going to neglect increases in estimation speed from
    # trading off SNR for samples.
    #
    # For the bistatic case, a given time interval of integration can
    # be divided into sub-intervals based on the range
    # measurement resolution and the ionospheric correlation time.
    #
    # For the monostatic case, we add the time of flight to the target range
    # as an additional limiting factor.
    #
    # Sub cases:
    #  1) Range resolution is low: limited by the larger of time it takes light
    #     to span measurement volume and time of flight to measurement volume
    #     (the latter in the case of monostatic)
    #  2) Range resolution is high (bistatic case): limited by medium decorrelation time
    #
    # After this limit occurs, duty cycle de-rating is used to obtain sampling rate.
    #
    # Note that this calculation does NOT warn the user of potential range aliasing
    # in the monostatic case,
    # where previous IPPs show up at the same time as the one being measured.
    # The user should take care to avoid this problem by setting minimum_observation_interval and Rtau
    # properly and by not selecting a duty cycle that is too fast and thus triggers
    # range aliasing problems.

    # decorrelation time of the incoherent scatter process.
    decorrelation_time = 1.0 / (2.0 * line_shift)

    # how many incoherent scatter samples per second
    if monostatic:
        # time of flight to maximum range must be considered as a limiting factor to how often we can
        # transmit pulses.
        minimum_observation_interval = 2.0 * maximum_range_m / sc.c
        sample_rate = 1.0 / np.maximum(
            minimum_observation_interval,
            np.maximum(decorrelation_time / duty_cycle, p_length / duty_cycle),
        )
    else:
        sample_rate = 1.0 / np.maximum(
            decorrelation_time / duty_cycle, p_length / duty_cycle
        )

    #
    # take into account self noise and the fact that we can trade signal for
    # independent samples of the incoherent scatter radar process
    #

    # standard method of measurement
    s_factor = 1.0
    mtime = (s / s_factor + n) ** 2.0 / (
        baud_gain
        * s_factor
        * sample_rate
        * estimation_error_stdev**2.0
        * (s / s_factor) ** 2.0
    )

    # find out how to divide our transmit pulse to obtain minimal measurement time,
    # also known as the Mr. ACF trick.
    # M. P. Sulzer, “A phase modulation technique for a sevenfold statistical improvement in incoherent scatter data‐taking,” Radio Science, vol. 21, no. 4, pp. 737–744, Jul. 1986, doi: 10.1029/RS021i004p00737.

    if mtime_estimate_method == "mracf":
        for s_factor in np.arange(2.0, 10.0):
            mtime = np.minimum(
                mtime,
                (s / s_factor + n) ** 2.0
                / (
                    baud_gain
                    * s_factor
                    * sample_rate
                    * estimation_error_stdev**2.0
                    * (s / s_factor) ** 2.0
                ),
            )

    # debug printout for sanity checks
    #if tx_to_target_range_m > 500.0E3 and tx_to_target_range_m < 650.0E3 and tx_target_rx_angle > 44.0 and tx_target_rx_angle < 46.0:
    #    print("\nrange_resolution_m ", range_resolution_m)
    #    print("tx range ", tx_to_target_range_m)
    #    print("tx to rx angle ", tx_target_rx_angle)
    #    print("tx beamwidth deg", np.rad2deg(rad_tx_beamwidth))
    #    print("rx beamwidth deg", np.rad2deg(rad_rx_beamwidth))
    #    print("volume ", volume)
    #    bv = sc.c / 2.0 * baud_length_s * tx_to_target_range_m**2 * np.sin(np.deg2rad(tx_target_rx_angle)) * rad_tx_beamwidth**2
    #    print("bowles volume ", bv)
    #    print("bv / vol ", bv/volume)
    #    print("noise scale ", noise_scaling)
    #    print("wavelength ", wavelength)
    #    print("debye length ", debye_length_m)
    #    print("gain_tx ", gain_tx)
    #    print("peak power ", peak_power_W)
    #    print("sys temp ", system_temperature)
    #    print("unit_xsection", unit_xsection)
    #    print("rcs ", rcs, " ", 10.0*np.log10(rcs), " dB")
    #    print("snr ", snr)
    #    print("mtime ", mtime)

    if calculate_plasma_parameter_errors:
        (plasma_parameter_errors) = is_calculate_plasma_parameter_errors(
            0.0,
            frequency_Hz,
            Ti,
            Te,
            Ne,
            ion_species,
            ion_fraction,
            snr,
            estimation_error_stdev,
        )
        return (
            snr,
            power_aperture_to_temperature,
            avg_power_aperture_to_temperature,
            wavelength_to_debye_length_ratio,
            bandwidth,
            mtime,
            plasma_parameter_errors,
        )
    else:
        return (
            snr,
            power_aperture_to_temperature,
            avg_power_aperture_to_temperature,
            wavelength_to_debye_length_ratio,
            bandwidth,
            mtime,
        )
