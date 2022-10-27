#!/usr/bin/env python
"""
    ISR Mapping library

    This library provides helper functions to enable easy map generation for IS
    radar performance assesement using the MIPS model.

"""
import time
import string
import datetime
import numpy as np
import numpy.ma as ma
import scipy.constants as c

from .coord import geodetic_to_az_el_r, azel_ecef, geodetic2ecef
from .isr_performance import is_snr, iri2016
from .configtools import build_site_lists, build_radar_lists
from .isr_sim_array import simulate_data

# from mpl_toolkits.basemap import Basemap, shiftgrid
import xarray as xr
import iri2016 as iri

# Helper routines


def llh2ecef(lat, lon, alt):
    """Latitude, longitude, height to ecef.

    Parameters
    ----------
    lat : float
        Latitude in degrees.
    lon : float
        Longitude in degrees.
    alt : float
        Altitude in meters.

    Returns
    -------
    x : float
        ECEF x in meters.
    y : float
        ECEF y in meters.
    z : float
        ECEF z in meters.
    """
    # see http://www.mathworks.de/help/toolbox/aeroblks/llatoecefpositionp.html

    rad = np.float64(6378137.0)  # Radius of the Earth (in meters)
    f = np.float64(1.0 / 298.257223563)  # Flattening factor WGS84 Model
    cosLat = np.cos(lat)
    sinLat = np.sin(lat)
    FF = (1.0 - f) ** 2
    C = 1 / np.sqrt(cosLat**2 + FF * sinLat**2)
    S = C * FF

    x = (rad * C + alt) * cosLat * np.cos(lon)
    y = (rad * C + alt) * cosLat * np.sin(lon)
    z = (rad * S + alt) * sinLat

    return x, y, z


def aer2ecef(obs_lat, obs_long, obs_alt, azimuthDeg, elevationDeg):

    slantRange = 1.0

    # site ecef in meters
    sitex, sitey, sitez = llh2ecef(obs_lat, obs_long, obs_alt)

    # some needed calculations
    slat = np.sin(np.radians(obs_lat))
    slon = np.sin(np.radians(obs_long))
    clat = np.cos(np.radians(obs_lat))
    clon = np.cos(np.radians(obs_long))

    azRad = np.radians(azimuthDeg)
    elRad = np.radians(elevationDeg)

    # az,el,range to sez convertion
    south = -slantRange * np.cos(elRad) * np.cos(azRad)
    east = slantRange * np.cos(elRad) * np.sin(azRad)
    zenith = slantRange * np.sin(elRad)

    x = (slat * clon * south) + (-slon * east) + (clat * clon * zenith) + sitex
    y = (slat * slon * south) + (clon * east) + (clat * slon * zenith) + sitey
    z = (-clat * south) + (slat * zenith) + sitez
    v = np.array([x, y, z])
    return v / np.linalg.norm(v)


def thetaphi2uv(theta, phi, rotate=False):

    if rotate:
        # offset elevation to angle from boresite
        el = 90.0 - el

        # rotate azimuth to be centered at +/- 180.0
        if az > 180.0:
            az = 360.0 - az

    u = np.sin(np.pi / 180.0 * theta) * np.cos(np.pi / 180.0 * phi)
    v = np.sin(np.pi / 180.0 * theta) * np.sin(np.pi / 180.0 * phi)

    return (u, v)


def azel2thetaphi(az, el, rotate=False):

    if rotate:
        # offset elevation to angle from boresite
        el = 90.0 - el

        # rotate azimuth to be centered at +/- 180.0
        if az > 180.0:
            az = 360.0 - az

    theta = (
        np.arccos(np.cos(np.pi / 180.0 * el) * np.cos(np.pi / 180.0 * az))
        * 180.0
        / np.pi
    )

    if el == 0.0:
        phi = 0.0
    else:
        phi = (
            np.arctan(np.tan(np.pi / 180.0 * el) / np.sin(np.pi / 180.0 * az))
            * 180.0
            / np.pi
        )

    return (theta, phi)


def azel2uv(az, el, rotate=False):

    if rotate:
        # offset elevation to angle from boresite
        el = 90.0 - el

        # rotate azimuth to be centered at +/- 180.0
        if az > 180.0:
            az = 360.0 - az

    u = np.cos(np.pi / 180.0 * el) * np.sin(np.pi / 180.0 * az)
    v = np.sin(np.pi / 180.0 * el)

    return (u, v)


def equal_gain_mask(az, el, mask_limits, boresite=np.array([0.0, 0.0])):
    """Dish like radar gain mask. Simplified."""

    if (
        az < mask_limits[0]
        or az > mask_limits[1]
        or el < mask_limits[2]
        or el > mask_limits[3]
    ):
        mask = 0.0
    else:
        mask = 1.0

    return mask


def planar_gain_mask(
    az,
    el,
    mask_limits,
    boresite=np.array([0.0, 0.0]),
    lattice_spacing=np.array([0.0, 0.0]),
):
    """Planar array gain mask with boresite tilt angles. Grating lobe lattice not working yet."""

    # sky elevation angle is actual elevation from horizon
    el = el

    az_bore = boresite[0]

    # boresite elevation angle is indicated as tilt from zenith
    el_bore = 90.0 - boresite[1]

    # aer2ecef
    # convert az el pointing and boresite angles to ecef
    ecf_beam = azel_ecef(0.0, 0.0, 0.0, az, el)
    ecf_bore = azel_ecef(0.0, 0.0, 0.0, az_bore, el_bore)
    ecf_beam_az = azel_ecef(0.0, 0.0, 0.0, az, 0.0)
    ecf_bore_az = azel_ecef(0.0, 0.0, 0.0, az_bore, 0.0)
    ecf_beam_el = azel_ecef(0.0, 0.0, 0.0, 0.0, el)
    ecf_bore_el = azel_ecef(0.0, 0.0, 0.0, 0.0, el_bore)

    bb_prod = np.dot(ecf_beam, ecf_bore) / (
        np.linalg.norm(ecf_beam) * np.linalg.norm(ecf_bore)
    )
    bb_az_prod = np.dot(ecf_beam_az, ecf_bore_az) / (
        np.linalg.norm(ecf_beam_az) * np.linalg.norm(ecf_bore_az)
    )
    bb_el_prod = np.dot(ecf_beam_el, ecf_bore_el) / (
        np.linalg.norm(ecf_bore_el) * np.linalg.norm(ecf_beam_el)
    )

    bb_angle = np.arccos(bb_prod)
    bb_az_angle = np.arccos(bb_az_prod)
    bb_el_angle = np.arccos(bb_el_prod)

    # the mask is the cosine of the angle between the beam and boresite vectors
    # handles both the azimuth and elevation angles due to vector projection
    mask = bb_prod

    # we can't look behind an array
    if mask < 0.0:
        mask = 0.0

    # elevation mask is angle from boresite
    mlim_el = mask_limits[3]
    # mask if the angle between the beam and boresite is greater than el mask
    if np.rad2deg(bb_angle) > mlim_el:
        mask = 0.0

    # now impose grating lobe restrictions
    # need to compute vectors in UV space

    # compute grating lobes for lattice in UV space
    ### NOTE : CURRENTLY WRONG!
    if False:  # lattice_spacing[0] > 0.0:

        u0, v0 = azel2uv(np.rad2deg(bb_az_angle), 90.0 - np.rad2deg(bb_el_angle))
        # u0,v0 = azel2uv(az,el,True)

        # convert to theta phi to UV space
        # theta_beam, phi_beam = azel2thetaphi(az,el)
        # theta_bore, phi_bore = azel2thetaphi(az_bore,el_bore)
        # theta_bore = 0.0
        # phi_bore = 0.0

        # u_beam,v_beam = thetaphi2uv(theta_beam,phi_beam)
        # u_bore,v_bore = thetaphi2uv(theta_bore,phi_bore)

        # u0 = u_bore - u_beam
        # v0 = v_bore - v_beam

        # grating uv for triangular lattice
        # evaluate all offsets n,m near central circle
        # do not evaluate main beam location 0,0

        lattice_pairs = [
            (2, 2),
            (-2, 2),
            (2, -2),
            (-2, -2),
            (0, 2),
            (2, 0),
            (0, -2),
            (-2, 0),
        ]

        for l in lattice_pairs:
            nx, mx = l

            um = u0 + mx * 1.0 / (2.0 * lattice_spacing[0])
            vn = v0 + nx * 1.0 / (2.0 * lattice_spacing[1])

            d = np.sqrt(um**2 + vn**2)

            if d <= 1.0:

                mask = 0.0

    return mask


def combine_parameter_errors(error_mat, max_error, min_error):
    """
    This method takes a set of error maps and combines them.

    """
    n_paths = error_mat.shape[0]
    n_lat = error_mat.shape[1]
    n_lon = error_mat.shape[2]

    combined_error_mat = np.zeros([n_lat, n_lon])

    for path_idx in range(n_paths):

        combined_error_mat += 1.0 / (error_mat[path_idx, :, :])

    combined_error_mat = 1.0 / combined_error_mat

    # bound bad values to NaN
    combined_error_mat[np.where(combined_error_mat > max_error)] = np.nan
    combined_error_mat[np.where(combined_error_mat <= min_error)] = np.nan

    return combined_error_mat


def combine_velocity_errors(
    vel_mat, fov_mask, vel_error_stdev=10.0, max_error=100.0, min_error=0.0
):
    """
    This method takes the velocity matrix k vector representation and the velocity errors. Returns the latitude, longitude array of combined errors.

    """

    n_paths = vel_mat.shape[0]
    n_lat = vel_mat.shape[1]
    n_lon = vel_mat.shape[2]

    vel_err = np.zeros([n_lat, n_lon])

    for lat_i in range(n_lat):
        for lon_i in range(n_lon):
            A = []
            for path_i in range(n_paths):
                if vel_mat[path_i, lat_i, lon_i, 0] != 0.0:
                    A.append(vel_mat[path_i, lat_i, lon_i, :])
            if len(A) > 2:
                A = np.array(A)
                Sinv = np.diag(np.repeat(1.0 / (vel_error_stdev**2.0), len(A)))
                try:
                    Spost = np.linalg.inv(np.dot(np.dot(np.transpose(A), Sinv), A))
                    vel_err[lat_i, lon_i] = np.sqrt(np.max(np.diag(Spost)))
                except:
                    vel_err[lat_i, lon_i] = np.nan
            elif len(A) > 0:
                # only one path with nonzero velocity - use max of velocity error from that path
                vel_err[lat_i, lon_i] = np.max(A[0])
            else:
                # no paths at all with nonzero velocity
                vel_err[lat_i, lon_i] = np.nan

            if fov_mask[lat_i, lon_i] == 0.0:
                vel_err[lat_i, lon_i] = np.nan

            # bound bad values to NaN
            vel_err[np.where(np.abs(vel_err) > max_error)] = np.nan
            vel_err[np.where(np.abs(vel_err) <= min_error)] = np.nan

    return vel_err


"""
    The ISR array simulator takes a set of parameters to simulate a network of IS radars. The parameters of the array are provided as vectors for each item.  The pair list controls which pairs of transmit and receive are computed and combined into the final measurement speed estmiate.
"""


def isr_array_sim(
    tx_lat,
    tx_lon,
    tx_alt,
    rx_lat,
    rx_lon,
    rx_alt,
    tx_el_mask,
    rx_el_mask,
    tx_type,
    tx_boresite,
    tx_mask_limits,
    tx_gain,
    tx_frequency,
    tx_peak_power,
    tx_duty_cycle,
    n_bauds,
    tx_pulse_length,
    rx_type,
    rx_boresite,
    rx_mask_limits,
    rx_gain,
    rx_tsys_type,
    rx_extra_T_sys,
    pair_list,
    eval_grid,
    n_grid_cells,
    max_range,
    v_doppler_max,
    t_max,
    target_estimation_error,
    plasma_parameter_errors,
    ionosphere,
    mtime_estimate_method,
    mpclient=None,
    pfunc=print,
):
    """Compute ISR performance for one or more TX, RX pairs over a given elevation threshold defined. Once all of the inputs are ready they are then put into the xarray based simulator tool.

    Use IRI if indicated. For IRI the times are in Universal Time (Zulu). A local or NASA cgi version of IRI can be used.

    Parameters
    ----------
    tx_lat : array_like
        Listing of latitudes of Tx sites.
    tx_lon : array_like
        Listing of longitudes of Tx sites.
    tx_alt : array_like
        Listing of altitudes of Tx sites.
    rx_lat : array_like
        Listing of latitudes of Rx sites.
    rx_lon : array_like
        Listing of longitudes of Rx sites.
    rx_alt : array_like
        Listing of altitudes of Rx sites.
    tx_el_masking : array_like
        Elevation mask for the Tx.
    rx_el_masking : array_like
        Elevation mask for the Rx.
    tx_type : list
        Transmit antenna type.
    tx_boresite : list
        Az and el location of boresite.
    tx_mask_limites : array_like
        Limits of the tx antennas in degrees. [min_az, max_az,min_el, max_el]
    tx_gain : list
        Tx antenna gain in dB.
    tx_frequency : float
        Transmitter center frequency in Hz.
    tx_peak_power : list
        Tx peak power in W.
    tx_duty_cycle : lists
        Tx duty cyle.
    n_bauds : int
        Number of bauds in the tx pulse.
    tx_pulse_length : int
        Length of each pulse in nanoseconds.
    rx_type : list
        Tx antenna type.
    rx_boresite : list
        Az and el location of boresite.
    rx_mask_limites : array_like
        Limits of the rx antennas in degrees. [min_az, max_az,min_el, max_el]
    rx_gain : list
        Rx antenna gain in dB.
    rx_tsys_type : list
        Type of tsys behavior.
    rx_extra_T_sys : list
        Extra tsys in deg kelvin.
    pair_list : list
        Tx Rx pairs to be evaluated.
    eval_grid : list
        Latitude longitude grid to be evaluated over. [min_lat, max_lat, min_lon, max_lon].
    n_grid_cells : int
        Number of cells per dimension to evaluate over.
    max_range : float
        Maximum range that will be evaluated.
    v_doppler_max : float
        Maximum Doppler used for simulation.
    t_max : float
        Longest measurement time considered.
    target_estimation_error : float
        Desired error in target to set measurement speed.
    plasma_parameter_errors : bool
        If true will also determine plasma parameter errors.
    ionosphere : dict
        How the ionosphere will be handled.
        example = {
            "use_iri": False,
            "iri_type": "local",
            "iri_time": None,
            "alt_m": 300e3,
            "N_e": 2e11,
            "T_e": 1000.0,
            "T_i": 800.0,
        }
    mtime_estimate_method : str
        String to determine measurement time estimation method. Can be std, mracf.
    mpclient : Dask.Client
        Multiprocessing client from dask. If None then standard processing is used.
    pfunc : func
        Print function. defaults to standard print.

    Returns
    -------
    dataset : xarray.Dataset
        Final data set from simulation.
    """
    n_tx = len(tx_lat)
    n_rx = len(rx_lat)

    # pair filter list is supplied
    if not pair_list == []:
        n_paths = len(pair_list)
    else:
        n_paths = n_tx * n_rx

    pfunc(
        (
            "IS radar network : number of transmitters %d, number of receivers %d, number of paths %d\n"
            % (n_tx, n_rx, n_paths)
        )
    )

    lmbda = 2.99792458e8 / tx_frequency

    # duty-cycle
    eff_tx = 1.0  # we should pass this through from the user level
    eff_rx = 1.0  # we should pass this through from the user level

    # smallest fundamental integration period
    t_int = tx_pulse_length / n_bauds

    # bandwidth factor
    bw_fac = 1.0  # we should pass this through from the user level

    # Set up the dimensions for the simulation
    data_dims = dict(pairs=n_paths, lat=n_grid_cells, long=n_grid_cells)
    # Terms that will be constant through out simulation
    const_dict = dict(
        pulse_length_ns=tx_pulse_length,
        n_bauds=n_bauds,
        maximum_range_m=max_range,
        efficiency_tx=1.0,
        efficiency_rx=1.0,
        bandwidth_factor=1.0,
        maximum_bulk_doppler=v_doppler_max,
        frequency_Hz=tx_frequency,
        estimation_error_stdev=target_estimation_error,
        calculate_plasma_parameter_errors=plasma_parameter_errors,
        mean_lat=np.mean(tx_lat),
        mean_lon=np.mean(tx_lon),
        tx_lat=np.array(tx_lat),
        tx_lon=np.array(tx_lon),
        rx_lat=np.array(rx_lat),
        rx_lon=np.array(rx_lon),
        quick_bandwidth_estimate=True,
        mtime_estimate_method=mtime_estimate_method,  #'std'
    )
    const_dict["O+"] = 1.0

    # Set up IRI stuff
    if ionosphere["use_iri"]:
        Ne_arr = np.zeros([len(lats), len(longs)])
        Te_arr = np.zeros([len(lats), len(longs)])
        Ti_arr = np.zeros([len(lats), len(longs)])
        # time in ISO8601 format (simplified)
        # time is GMT!
        t = datetime.datetime.strptime(ionosphere["iri_time"], "%Y-%m-%dT%H:%M:%SZ")

        iri_jf = np.zeros(50)
        iri_oa = np.zeros(100)

        # This is the IRI control variable flag list. See the IRI fortran code.
        iri_jf[0:35] = [
            1,
            1,
            1,
            0,
            0,
            0,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            0,
            0,
            0,
            1,
            1,
            1,
            1,
            0,
            0,
            0,
            1,
            1,
            0,
            0,
            0,
        ]

        tval1 = t.month * 100 + t.day
        # note all times are GMT!
        # indicate GMT to the local IRI model
        tval2 = t.hour + t.minute / 60.0 + t.second / 3600.0 + 25.0

    else:
        const_dict["Ne"] = ionosphere["N_e"]
        const_dict["Te"] = ionosphere["T_e"]
        const_dict["Ti"] = ionosphere["T_i"]

    # target ionosphere grid to evaluate
    lats = np.linspace(eval_grid[0], eval_grid[1], num=n_grid_cells)
    longs = np.linspace(eval_grid[2], eval_grid[3], num=n_grid_cells)

    pairs = np.arange(n_paths)
    if pair_list:
        tx_idx = np.empty(n_paths).astype(int)
        rx_idx = np.empty(n_paths).astype(int)
        for ind, ipair in enumerate(pair_list):
            tx_idx[ind] = int(ipair[0])
            rx_idx[ind] = int(ipair[1])
    else:
        tx_mat, rx_mat = np.meshgrid(
            np.arange(n_tx, dtype=int), np.arange(n_rx, dtype=int)
        )
        tx_idx = tx_mat.flatten()
        rx_idx = rx_mat.flatten()

    lat_bool = np.abs(tx_lat[tx_idx] - rx_lat[rx_idx]) < 0.01
    lon_bool = np.abs(tx_lon[tx_idx] - rx_lon[rx_idx]) < 0.01
    mono_bool = np.logical_and(lat_bool, lon_bool)
    coorddict = dict(
        lat=lats,
        long=longs,
        monostatic=("pairs", mono_bool),
        peak_power_W=("pairs", np.array(tx_peak_power)[tx_idx]),
        duty_cycle=("pairs", np.array(tx_duty_cycle)[tx_idx]),
        excess_rx_noise_K=("pairs", np.array(rx_extra_T_sys)[rx_idx]),
        tsys_type=("pairs", np.array(rx_tsys_type)[rx_idx]),
    )

    rx_gain_mat = np.zeros([n_paths, len(lats), len(longs)])
    tx_gain_mat = np.zeros([n_paths, len(lats), len(longs)])
    rx_el_mat = np.zeros([n_paths, len(lats), len(longs)])
    tx_el_mat = np.zeros([n_paths, len(lats), len(longs)])
    rx_range_mat = np.zeros([n_paths, len(lats), len(longs)])
    tx_range_mat = np.zeros([n_paths, len(lats), len(longs)])
    tx_gain_factors = np.zeros([n_paths, len(lats), len(longs)])
    rx_gain_factors = np.zeros([n_paths, len(lats), len(longs)])
    gammas = np.zeros([n_paths, len(lats), len(longs)])
    dvel_mat = np.zeros([n_paths, len(lats), len(longs)])
    vel_mat = np.zeros([n_paths, len(lats), len(longs), 3])
    fov_mask = np.zeros([len(lats), len(longs)])
    bistatic_volume = np.zeros([n_paths, len(lats), len(longs)])
    # HACK: Can loose loop if coordinate transforms work for arrays.
    # Loop through all of the locations and pairs
    for i, lat in enumerate(lats):
        for j, lon in enumerate(longs):
            for tx_i, rx_i, path_idx in zip(tx_idx, rx_idx, pairs):

                tgt_alt = ionosphere["alt_m"]

                (az_tx, el_tx, r_tx) = geodetic_to_az_el_r(
                    tx_lat[tx_i], tx_lon[tx_i], 0.0, lat, lon, tgt_alt
                )
                (az_rx, el_rx, r_rx) = geodetic_to_az_el_r(
                    rx_lat[rx_i], rx_lon[rx_i], 0.0, lat, lon, tgt_alt
                )

                tx_ecef = geodetic2ecef(tx_lat[tx_i], tx_lon[tx_i], tx_alt[tx_i])
                rx_ecef = geodetic2ecef(rx_lat[rx_i], rx_lon[rx_i], rx_alt[rx_i])
                target_ecef = geodetic2ecef(lat, lon, tgt_alt)
                k_tx = target_ecef - tx_ecef
                k_rx = target_ecef - rx_ecef

                # vector magnitiude
                k_txm = np.sqrt(np.dot(k_tx, k_tx))
                k_rxm = np.sqrt(np.dot(k_rx, k_rx))

                # get the cos of the scattering angle
                inv_angle = np.dot(k_tx, k_rx) / (k_txm * k_rxm)

                # normalized k vector, compute bragg vector
                # just the vector direction
                k_tx0 = k_tx / k_txm
                k_rx0 = -k_rx / k_rxm
                k_bragg = k_rx0 - k_tx0

                # extra debug check with lambda scaled values
                # k_tx_n = np.linalg.norm(2*np.pi*k_tx0/lmbda)
                # k_rx_n = np.linalg.norm(2*np.pi*k_rx0/lmbda)
                # k_bragg_n = np.linalg.norm(2*np.pi*k_bragg/lmbda)

                # print("k ", k_tx_n, k_rx_n, k_bragg_n)
                # print("l ", 2*np.pi/k_tx_n, 2*np.pi/k_rx_n, 2*np.pi/k_bragg_n)
                # print("f ", c.c*k_tx_n/(2*np.pi),c.c*k_rx_n/(2*np.pi),c.c*k_bragg_n/(2*np.pi))

                vel_mat[path_idx, i, j, :] = k_bragg

                # invariant scattering angle
                # degenerate cases where inv_angle is machine epsilon + 1.0
                if inv_angle > 1.0:
                    gamma = 0.0
                else:
                    gamma = 180.0 * np.arccos(inv_angle) / np.pi

                # See appendix in [1] R. de Elía and I. Zawadzki, “Sidelobe Contamination in Bistatic Radars,” Journal of Atmospheric and Oceanic Technology, vol. 17, no. 10, pp. 1313–1329, Oct. 2000, doi: 10.1175/1520-0426(2000)017<1313:SCIBR>2.0.CO;2.

                bistatic_volume[path_idx, i, j] = np.cos(np.deg2rad(gamma / 2)) ** (-2)

                tx_range_mat[path_idx, i, j] = r_tx
                rx_range_mat[path_idx, i, j] = r_rx

                rx_el_mat[path_idx, i, j] = el_rx
                tx_el_mat[path_idx, i, j] = el_tx

                tx_gain_factor = 1.0
                rx_gain_factor = 1.0

                # adjust for boresite angle >?
                az_rx_adj = az_rx
                az_tx_adj = az_tx
                el_rx_adj = el_rx
                el_tx_adj = el_tx

                # rotate azimuth angles
                if az_rx_adj < 0.0:
                    az_rx_adj += 360.0

                if az_tx_adj < 0.0:
                    az_tx_adj += 360.0

                # check for below terrain horizon and apply gain masks
                if el_rx_adj < rx_el_mask[rx_i]:
                    rx_gain_factor = 0.0
                else:
                    if rx_type[rx_i] == "planar_array":
                        # leave out grating masks for the moment
                        rx_gain_factor = planar_gain_mask(
                            az_rx_adj,
                            el_rx_adj,
                            rx_mask_limits[rx_i],
                            rx_boresite[rx_i],
                        )
                    else:
                        rx_gain_factor = equal_gain_mask(
                            az_rx_adj,
                            el_rx_adj,
                            rx_mask_limits[rx_i],
                            rx_boresite[rx_i],
                        )

                if el_tx_adj < tx_el_mask[tx_i]:
                    tx_gain_factor = 0.0
                else:
                    if tx_type[rx_i] == "planar_array":
                        # leave out grating masks for the moment
                        tx_gain_factor = planar_gain_mask(
                            az_tx_adj,
                            el_tx_adj,
                            tx_mask_limits[tx_i],
                            tx_boresite[tx_i],
                        )
                    else:
                        tx_gain_factor = equal_gain_mask(
                            az_tx_adj,
                            el_tx_adj,
                            tx_mask_limits[tx_i],
                            tx_boresite[tx_i],
                        )

                tx_gain_factors[path_idx, i, j] = tx_gain_factor
                rx_gain_factors[path_idx, i, j] = rx_gain_factor

                gammas[path_idx, i, j] = gamma

                # dB with -60 dB regularization
                tx_gain_tmp = 10.0 * np.log10(
                    tx_gain_factor * 10.0 ** (tx_gain[tx_i] / 10.0) + 1e-6
                )
                rx_gain_tmp = 10.0 * np.log10(
                    rx_gain_factor * 10.0 ** (rx_gain[rx_i] / 10.0) + 1e-6
                )
                tx_gain_mat[path_idx, i, j] = tx_gain_tmp
                rx_gain_mat[path_idx, i, j] = rx_gain_tmp
                # compute with IRI if indicated
                if ionosphere["use_iri"]:
                    # return is actually oa variable and iri results
                    if ionosphere["iri_type"] == "local":
                        iri_val = iri.iriFort.iri_sub(
                            iri_jf,
                            0,
                            lat,
                            lon,
                            t.year,
                            tval1,
                            tval2,
                            ionosphere["alt_m"] / 1e3,
                            ionosphere["alt_m"] / 1e3 + 1.0,
                            1.0,
                            iri_oa,
                        )
                        Ne_arr[i, j] = iri_val[0][0][0]
                        Ti_arr[i, j] = iri_val[0][2][0]
                        Te_arr[i, j] = iri_val[0][3][0]

                    else:
                        iri_cgi = iri2016(
                            t, lat, lon, alt_m / 1e3, alt_m / 1e3 + 1.0, 1.0
                        )
                        Ne_arr[i, j] = iri_cgi["ne"]
                        Te_arr[i, j] = iri_cgi["te"]
                        Ti_arr[i, j] = iri_cgi["ti"]

    # final set up and data simulation
    dimttuple = ("pairs", "lat", "long")
    coorddict["tx_to_target_range_m"] = (dimttuple, tx_range_mat)
    coorddict["target_to_rx_range_m"] = (dimttuple, rx_range_mat)
    coorddict["gain_tx_dB"] = (dimttuple, tx_gain_mat)
    coorddict["gain_rx_dB"] = (dimttuple, rx_gain_mat)
    coorddict["tx_target_rx_angle"] = (dimttuple, gammas)
    coorddict["bistatic_volume_factor"] = (dimttuple, bistatic_volume)
    dataset = simulate_data(data_dims, coorddict, const_dict, mpclient, pfunc)

    # Combine all of the errors over lat and long
    mtime = dataset["measurement_time"].values
    nantime = np.isnan(mtime)
    infinitetime = np.logical_not(np.isfinite(mtime))
    mlog = np.logical_or(nantime, infinitetime)
    mtime[mlog] = 1e99
    # zero measurement speed is out of bounds
    mtime[np.where(mtime == 0.0)] = np.nan

    # combine the measurement speed for independent paths

    delta_t_mat_tot = combine_parameter_errors(mtime, t_max, 0.0)

    # Set up error data set
    lld = ["lat", "long"]
    error_data = {"delta_t_mat_tot": (["lat", "long"], delta_t_mat_tot)}
    # create fov mask from speed matrix
    fov_mask[np.where(delta_t_mat_tot <= t_max)] = 1.0
    fov_mask[np.where(delta_t_mat_tot > t_max)] = 0.0
    fov_mask[np.where(delta_t_mat_tot < 0.0)] = 0.0
    fov_mask[np.where(delta_t_mat_tot == np.nan)] = 0.0

    if plasma_parameter_errors:
        tx_mask = tx_gain_factors == 0.0
        rx_mask = rx_gain_factors == 0.0
        err_mask = np.logical_or(tx_mask, rx_mask)
        dataset["dNe"].values[err_mask] = np.nan
        dataset["dTi"].values[err_mask] = np.nan
        dataset["dTe"].values[err_mask] = np.nan
        dataset["dV"].values[err_mask] = 0.0
        # arbitrary upper bounds for the moment
        pfunc("WARNING: Arbitrary upper bounds for parameter errors")

        dNe_t = combine_parameter_errors(dataset["dNe"].values, 1.1, 0.0)
        error_data["dNe_tot"] = (lld, dNe_t)
        dTi_t = combine_parameter_errors(dataset["dTi"].values, 1e2, 0.0)
        error_data["dTi_tot"] = (lld, dTi_t)
        dTe_t = combine_parameter_errors(dataset["dTe"].values, 1e2, 0.0)
        error_data["dTe_tot"] = (lld, dTe_t)
        # for now only support a single error bound for the velocity error
        # this corresponds to fixed ionospheric conditions
        # The worst case bound gets combined with the k-vector info
        maxdV = np.max(np.max(dataset["dV"].values))
        dV_t = combine_velocity_errors(vel_mat, fov_mask, maxdV, 1e2, 0.0)
        error_data["dV_tot"] = (lld, dV_t)

    # Make the error data set using the lat and longs as coordinates
    errds = xr.Dataset(error_data, {"lat": lats, "long": longs})
    # Merge and return the data sets.
    return xr.merge([dataset, errds])


def pair_list_self(tx_sites):
    """Creates radar network self pair list

    Parameters
    ----------
    tx_sites : list
        List of transmitter sites.
    rx_sites : list
        List of receiver sites.

    Returns
    -------
    pair_list : list
        List of Tx Rx pairs.
    """

    pair_list = []

    for i in range(len(tx_sites)):
        pair_list.append((i, i))

    return pair_list


def pair_list_cross(tx_sites, rx_sites):
    """Creates a radar network cross pair list

    Parameters
    ----------
    tx_sites : list
        List of transmitter sites.
    rx_sites : list
        List of receiver sites.

    Returns
    -------
    pair_list : list
        List of Tx Rx pairs.

    """

    pair_list = []

    for i in range(len(tx_sites)):
        for j in range(len(rx_sites)):
            pair_list.append((i, j))

    return pair_list


def pair_list_mimo(tx_sites, rx_sites):
    """Creates radar network mimo list, i.e. cartesian product.

    Parameters
    ----------
    tx_sites : list
        List of transmitter sites.
    rx_sites : list
        List of receiver sites.

    Returns
    -------
    pair_list : list
        List of Tx Rx pairs.
    """
    txv = list(range(len(tx_sites)))
    rxv = list(range(len(rx_sites)))
    pair_list = list(itertools.product(txv, rxv))

    return pair_list


def annotate_standard(
    alt_m,
    tx_gain,
    rx_elevation_threshold,
    tx_power,
    t_int,
    N_e,
    T_i,
    T_e,
    target_estimation_error,
):
    """
    Create the standard annotation strings for the plots.

    Parameters
    ----------
    alt_m : float
        Altitude in meters.
    tx_gain : list
        Transmitter antenna gain dB.
    rx_elevation_threshold : list
        Elevation threshold for receivers in degrees.
    tx_power : list
        Tx power in W
    t_int : Float
        Pulse length in s.
    N_e : float
        Electron density in m^-3
    T_i : float
        Ion temperature in K.
    T_e : float
        Electron temperature in K
    target_estimation_error : float
        Desired estimation error.

    Returns
    ------
    annotate : string
        Annotation for plots
    """

    annotate1 = (
        "Alt %1.2f km\nGain %1.0f dB\nElevation threshold = %1.2f deg above horizon\n"
        % (alt_m / 1e3, tx_gain[0], rx_elevation_threshold[0])
    )
    annotate2 = "$P_{\mathrm{tx}}=%1.2f$MW\n$T_{\mathrm{int}}=%1.2f$ms\n" % (
        tx_power[0] / 1e6,
        t_int / 1e-3,
    )
    annotate3 = "$\log_{10}(N_e)=%1.2f$ $m^{-3}$ $T_i = %1.2f$ K $T_e=%1.2f$ K\n" % (
        np.log10(N_e),
        T_i,
        T_e,
    )
    annotate4 = "Estimation error $\mathrm{stddev}(N_e)/N_e=%1.2f$" % (
        target_estimation_error
    )
    annotate = annotate1 + annotate2 + annotate3 + annotate4

    return annotate


def map_radar_array(
    tname,
    tx_sites,
    tx_radars,
    rx_sites,
    rx_radars,
    ipp=None,
    n_bauds=1,
    tx_pulse_length=1e-3,
    pair_list=None,
    plasma_parameter_errors=False,
    ionosphere=None,
    t_max=1e5,
    ngrid=100,
    extent=None,
    mtime_estimate_method="std",
    mpclient=None,
    pfunc=print,
):
    """
    Map an radar array for IS radar performance using the provide set of sites, radar types, parameters, ionospheric conditions, and limits. This function uses sets up the neccesary arrays and ionosphere conditions for isr_array_sim is called in this function.

    Parameters
    ----------
    tname : str
        Name of the map being run.
    tx_sites : list
        List of names of the tx sites.
    tx_radars : list
        Names of the tx radars.
    rx_sites : list
        List of names of the rx sites.
    rx_radars : list
        Names of the rx radars.
    ipp : int
        Interpulse period in nanoseconds.
    n_bauds : int
        Number of bauds on the transmit pulse
    tx_pulse_length : int
        Length of pulse of the mode in nanoseconds.
    pair_list: list
        List of transmitter receiver pairs as tuples of list elements, e.g. [(0,0), (0,1)].
    plasma_parameter_errors : bool
        Bool to run plasma parameter error estimates.
    ionosphere : dict
        Dictionary that determines the ionospheric conditions.
    t_max : float
        Maximum amount of integration time that will be used for experiment to reach a specific parameter resolution.
    ngrid : int
        Number of one side of grid points that this will be evalued over.
    extent : dict
        Dictionary containing to determine the latitude and logitude extent. Keys for the location center point are center_lat, center_lon; and keys for sampling size: delta_lat, delta_lon.
    mtime_estimate_method : str
        String to determine measurement time estimation method. Can be std, mracf.
    mpclient : dask.distributed.client
        Dask client to perform multiprocessing operations.
    pfunc : func
        Desired print function to use, use with logger module.

    Returns
    -------
    map_info : xarray.Dataset
        Results of simulation in xarray format.
        Data variables
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
        dNe: float (optional)
            expected plasma density error m^-3
        dTi : float (optional)
            Expected ion temperature error in k.
        dTe : float (optional)
            Expected electron temperature error in k.
        dV : float (optional)
            Expected velocity error in m/s.
    """

    # pfunc the name of the map that is being run
    pfunc("%s" % (tname))

    # tx radar locations
    tx_lats, tx_lons, tx_alts, tx_masks = build_site_lists(tx_sites)
    tx_lat = np.array(tx_lats)
    tx_lon = np.array(tx_lons)
    tx_alt = np.array(tx_alts)

    # rx radar locations
    rx_lats, rx_lons, rx_alts, rx_masks = build_site_lists(rx_sites)
    rx_lat = np.array(rx_lats)
    rx_lon = np.array(rx_lons)
    rx_alt = np.array(rx_alts)

    # terrain masks
    tx_elevation_threshold = tx_masks
    rx_elevation_threshold = rx_masks

    # tx radar parameters
    (
        rdtype,
        tx_boresite,
        steering_mask,
        tx_freq,
        tx_gain,
        rx_gain,
        tx_power,
        tx_duty_cycle,
        rx_tsys_type,
        xtra_tsys,
        notes,
    ) = build_radar_lists(tx_radars)

    # is radar simulator inputs
    tx_gain = np.array(tx_gain)
    tx_mask_limits = np.array(steering_mask)
    tx_power = np.array(tx_power)
    if ipp is None:

        tx_duty_cycle = np.array(tx_duty_cycle)
        t_int = tx_pulse_length / tx_duty_cycle
    else:
        t_int = ipp
        tx_duty_cycle = [float(tx_pulse_length) / ipp] * len(tx_duty_cycle)

    # rx radar parameters
    (
        rdtype,
        rx_boresite,
        steering_mask,
        rx_freq,
        rx_gain,
        rx_gain,
        rx_power,
        rx_duty_cycle,
        rx_tsys_type,
        xtra_tsys,
        notes,
    ) = build_radar_lists(rx_radars)

    rx_gain = np.array(rx_gain)
    rx_mask_limits = np.array(steering_mask)
    rx_tsys_type = rx_tsys_type
    rx_extra_T_sys = np.array(xtra_tsys)
    baud_len_s = 1e-9 * float(tx_pulse_length) / n_bauds
    pfunc("N bauds: " + str(n_bauds) + " baud length: " + str(baud_len_s))
    pfunc("TX Frequency: {0}".format(tx_freq))
    pfunc("TX power: {0} duty cycle: {1}".format(tx_power, tx_duty_cycle))
    # ionospheric parameters for non IRI based map static conditions

    if ionosphere == None:
        alt_m = 300e3
        N_e = 2e11
        T_e = 1000.0
        T_i = 800.0
        iri_time = "fixed parameters"  # '2012-10-22T0:00:00Z'

        ionosphere = {
            "use_iri": False,
            "iri_type": "local",
            "iri_time": iri_time,
            "alt_m": alt_m,
            "N_e": N_e,
            "T_e": T_e,
            "T_i": T_i,
        }
    else:
        if ionosphere["use_iri"]:
            raise ValueError("Dynamic IRI calls not supported yet")
        alt_m = ionosphere["alt_m"]
        N_e = ionosphere["N_e"]
        T_e = ionosphere["T_e"]
        T_i = ionosphere["T_i"]
        iri_time = ionosphere.get("iri_time", "fixed parameters")

    if extent == None:

        # compute map center
        grid_lat0 = np.mean(tx_lat)
        grid_lon0 = np.mean(tx_lon)

        # find lat max offset
        grid_lat0_mindelta = grid_lat0 - np.min(tx_lat)
        grid_lat0_maxdelta = np.max(tx_lat) - grid_lat0
        grid_lat0_delta = np.max([grid_lat0_mindelta, grid_lat0_maxdelta, 10.0]) + 10.0

        # find max lon offset
        grid_lon0_mindelta = grid_lon0 - np.min(tx_lon)
        grid_lon0_maxdelta = np.max(tx_lon) - grid_lon0
        grid_lon0_delta = np.max([grid_lon0_mindelta, grid_lon0_maxdelta, 10.0]) + 10.0

        pfunc(
            "grid info: {0}, {1}, {2}, {3}".format(
                grid_lat0, grid_lon0, grid_lat0_delta, grid_lon0_delta
            )
        )
    else:
        pfunc("set fixed extent")
        grid_lat0 = extent["center_lat"]
        grid_lon0 = extent["center_lon"]
        grid_lat0_delta = extent["delta_lat"]
        grid_lon0_delta = extent["delta_lon"]

        pfunc(
            "grid info: {0}, {1}, {2}, {3}".format(
                grid_lat0, grid_lon0, grid_lat0_delta, grid_lon0_delta
            )
        )

    # note these offsets are arbitrary and can end limiting the evaluation
    # extent to be too narrow for some cases (e.g. high altitudes).
    # This shows up as latitude and longitude cutoffs in the plots.
    #
    grid_lat0_min = np.max([grid_lat0 - grid_lat0_delta, -89.9])
    grid_lat0_max = np.min([grid_lat0 + grid_lat0_delta, 89.9])

    grid_lon0_min = np.max([grid_lon0 - grid_lon0_delta, -179.9])
    grid_lon0_max = np.min([grid_lon0 + grid_lon0_delta, 179.9])

    eval_grid = [grid_lat0_min, grid_lat0_max, grid_lon0_min, grid_lon0_max]
    pfunc("eval grid {0}".format(eval_grid))

    frequency = tx_freq[0]
    target_estimation_error = 0.05
    v_doppler_max = 2500.0

    if pair_list == None:
        pfunc("No pair list provided, evaluating self pairs only")
        pair_list = pair_list_self(tx_lat)
    elif type(pair_list) is str and pair_list == "self":
        pfunc("evaluating self pairs")
        pfunc(tx_lat)
        pair_list = pair_list_self(tx_lat)
    elif type(pair_list) is str and pair_list == "cross":
        pfunc("evaluating TX to RX cross pairs (matched TX to RX in sequence)")
        pair_list = pair_list_cross(tx_lat, rx_lat)
    elif type(pair_list) is str and pair_list == "mimo":
        pfunc("evaluating TX to RX mimo pairs (all TX to all RX)")
        pair_list = pair_list_cross(tx_lat, rx_lat)
    elif type(pair_list) is list:
        pfunc("external pair list provided")

    pfunc("pair list: " + str(pair_list))

    map_info = isr_array_sim(
        tx_lat=tx_lat,
        tx_lon=tx_lon,
        tx_alt=tx_alt,
        rx_lat=rx_lat,
        rx_lon=rx_lon,
        rx_alt=rx_alt,
        tx_el_mask=tx_elevation_threshold,
        rx_el_mask=rx_elevation_threshold,
        tx_mask_limits=tx_mask_limits,
        rx_mask_limits=rx_mask_limits,
        tx_type=rdtype,
        rx_type=rdtype,
        tx_boresite=tx_boresite,
        rx_boresite=rx_boresite,
        tx_gain=tx_gain,
        rx_gain=rx_gain,
        pair_list=pair_list,
        rx_extra_T_sys=rx_extra_T_sys,
        rx_tsys_type=rx_tsys_type,
        tx_frequency=frequency,
        tx_peak_power=tx_power,
        tx_duty_cycle=tx_duty_cycle,
        n_bauds=n_bauds,
        tx_pulse_length=tx_pulse_length,
        eval_grid=eval_grid,
        t_max=t_max,
        target_estimation_error=target_estimation_error,
        plasma_parameter_errors=plasma_parameter_errors,
        v_doppler_max=v_doppler_max,
        n_grid_cells=ngrid,
        max_range=1300e3,
        ionosphere=ionosphere,
        mtime_estimate_method=mtime_estimate_method,
        mpclient=mpclient,
        pfunc=pfunc,
    )

    map_info.attrs["map_title"] = "%s (%s)" % (tname, iri_time)
    map_info.attrs["annotate_txt"] = annotate_standard(
        alt_m,
        tx_gain,
        rx_elevation_threshold,
        tx_power,
        t_int,
        N_e,
        T_i,
        T_e,
        target_estimation_error,
    )
    return map_info
