#!/usr/bin/env python

import xarray as xr
from .isr_performance import is_snr
import numpy as np


ionnamelist = ["O+", "NO+", "N2+", "O2+", "N+", "H+", "He+"]


def get_ion_lists(constdict):

    namelist = []
    fraclist = []
    for iion, ifrac in constdict.items():

        if ifrac > 0:
            namelist.append(iion)
            fraclist.append(ifrac)
    return namelist, fraclist


def prunedict(cur_dict):
    """Used to prune the dictionary down to only neccesary parameters.

    Parameters
    ----------
    cur_dict : dict
        Dictionary with single number values that will be input to the is_snr function and extra material.

    Returns
    -------
    cur_dict : dict
        Same as input but with the extra material removed.
    """
    req_inputs = [
        "tsys_type",
        "monostatic",
        "quick_bandwidth_estimate",
        "calculate_plasma_parameter_errors",
        "peak_power_W",
        "maximum_range_m",
        "n_bauds",
        "pulse_length_ns",
        "duty_cycle",
        "gain_tx_dB",
        "gain_rx_dB",
        "efficiency_tx",
        "efficiency_rx",
        "frequency_Hz",
        "bandwidth_factor",
        "tx_to_target_range_m",
        "target_to_rx_range_m",
        "Ne",
        "Te",
        "Ti",
        "excess_rx_noise_K",
        "estimation_error_stdev",
        "maximum_bulk_doppler",
        "tx_target_rx_angle",
        "bistatic_volume_factor",
        "ionspecies",
        "ionfracs",
        "mtime_estimate_method"
    ]
    inputkeys = list(cur_dict.keys())
    for ikey in inputkeys:
        if not ikey in req_inputs:
            del cur_dict[ikey]

    return cur_dict


def check_dicts(coorddict, paramvalues):
    """Determines input dictionaries for the main function are filled out correctly.


    Parameters
    ----------
    coorddict : dict
        Same as the coorddict input in the simulate data funtion.
    paramvalues : dict
        Same as whats passed to simulate data.

    Returns
    -------
    err_flag : bool
        Will be raised true if dictionaries are not set up correctly.
    outstr : str
        Error explanation.
    """
    att_names = [
        "tsys_type",
        "monostatic",
        "quick_bandwidth_estimate",
        "calculate_plasma_parameter_errors",
    ]
    varnames = [
        "peak_power_W",
        "maximum_range_m",
        "n_bauds",
        "pulse_length_s",
        "duty_cycle",
        "gain_tx_dB",
        "gain_rx_dB",
        "efficiency_tx",
        "efficiency_rx",
        "frequency_Hz",
        "bandwidth_factor",
        "tx_to_target_range_m",
        "target_to_rx_range_m",
        "Ne",
        "Te",
        "Ti",
        "excess_rx_noise_K",
        "estimation_error_stdev",
        "maximum_bulk_doppler",
        "tx_target_rx_angle",
        "bistatic_volume_factor",
        "mtime_estimate_method"
    ]

    allnames = att_names + varnames

    all_keys = list(paramvalues.keys()) + list(coorddict.keys())

    outstr = ""
    ion_missing = True
    err_flag = False

    for ikey in all_keys:
        if ikey in ionnamelist:
            ion_missing = False
            continue
        if ikey in allnames:
            allnames.remove(ikey)
        # else:
        #     outstr = "{} is not a valid input to the simulation.".format(ikey)
        #     err_flag = True
        #     return err_flag, outstr

    if ion_missing:
        err_flag = True
        outstr = "Missing a valid ion species."
    return err_flag, outstr


def get_default(coordvals):
    """Create a default value dictionary given a list of parameters.

    Parameters
    ----------
    coordvals : list
        A list of strings that are parameter names.

    Returns
    -------
    coorddict : dict
        The dictionary holding the default values.
    default_params:
        The dictionary of values not selected.
    """
    default_params = dict(
        peak_power_W=2e6,
        maximum_range_m=800e3,
        n_bauds=1,
        pulse_length_ns=1000000,
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
        gain_tx_dB=42.0,
        gain_rx_dB=42.0,
        frequency_Hz=440e6,
        excess_rx_noise_K=0.0,
        mtime_estimate_method='std'
    )

    default_params["O+"] = 1.0
    coorddict = {}
    for ikey in coordvals:
        coorddict[ikey] = default_params[ikey]
        del default_params[ikey]

    return coorddict, default_params


def is_snr_mp(cur_coords, cursimdict):
    """Wraper function for is_snr that gives that passes the coordinate information to stuff everything back in"""

    outdata = is_snr(**cursimdict)

    return cur_coords, outdata


def rerunsim(ds, i_el):

    paramvalues = ds.attrs
    coordds = ds.coords
    coorddict = coordds.to_dataset().to_dict()["coords"]

    dimdict = ds.dims
    dimnames = list(dimdict.keys())
    dimlist = list(dimdict.values())
    cursimdict = paramvalues.copy()
    cursimdict["quick_bandwidth_estimate"] = True
    cursimdict["calculate_plasma_parameter_errors"] = True

    curcoords = np.unravel_index(i_el, dimlist)
    # go through the coordinates and get all of them.
    for iname, ival in coorddict.items():
        curdims = ds[iname].dims
        # HACK why is this if statement here? Are there coordinates that can be zero length?
        if len(curdims) == 0:
            curdims = iname
        indtuple = [None] * len(curdims)
        for inum, idim in enumerate(curdims):
            dim_ind = dimnames.index(idim)
            indtuple[inum] = curcoords[dim_ind]
        indtuple = tuple(indtuple)
        cursimdict[iname] = ds[iname].values[indtuple]
    # deal with the ion species by getting them into the list format.
    iondict = {i: cursimdict[i] for i in ionnamelist if i in cursimdict.keys()}
    cursimdict["ionspecies"], cursimdict["ionfracs"] = get_ion_lists(iondict)
    cursimdict = prunedict(cursimdict)
    cursimdict["pfunc"] = print

    outdata = is_snr(**cursimdict)

    return cursimdict, outdata


def simulate_data(data_dims, coorddict, paramvalues, mpclient=None, pfunc=print):
    """Creates an xarray data set with results of the simulations

    This function is used to sweep across different parameter values and then output the results to an xarray dataset. The swept parameters will become coordinates in the data set while the static parameters will become attributes. These parameters will be determined using the input dictionaries. Coordinates and dimensions will be based off of coorddict and data_dims dictionaries. All static parameters will be kept in paramvalues dictionary.

    Parameters
    ----------
    data_dims : dict
        Key is dimension name and value is size of the dimension.
    coorddict : dict
        Key is the name of the parameter that the simualtions will vary. Value is the list or array that the varies during the sweep or is a tuple, first object is the dimension name that varys and second is the array of values.
    paramvalues: dict
        This is the dictionary tha tholds the static parameters that do not vary. Key is the parameter name and the value is the parameter value.

    Returns
    -------
    ds : xarray.Dataset
        A data set from all the simulations.
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

    dimnames = list(data_dims.keys())
    dimlist = [data_dims[i] for i in data_dims.keys()]
    outnames = [
        "snr",
        "power_aperture_to_temperature",
        "avg_power_aperture_to_temperature",
        "wavelength_to_debye_length_ratio",
        "echo_bandwidth",
        "measurement_time",
    ]
    if paramvalues["calculate_plasma_parameter_errors"]:
        outnames += ["dNe", "dTi", "dTe", "dV"]

    # Create empty arrays for the data variables.
    data1 = {i: (dimnames, np.zeros(dimlist, dtype=np.float32)) for i in outnames}

    err_flag, err_str = check_dicts(coorddict, paramvalues)

    if err_flag:
        raise ValueError(err_str)

    ds = xr.Dataset(data1, coords=coorddict, attrs=paramvalues)

    nelements = np.prod(dimlist)

    cursimdict_list = []
    curcoords_list = []
    for i_el in range(nelements):
        # unravel everything
        curcoords = np.unravel_index(i_el, dimlist)
        # copy the parameter dictionary because those will be atributes that won't vary
        cursimdict = paramvalues.copy()
        # go through the coordinates and get all of them.
        for iname, ival in coorddict.items():
            curdims = ds[iname].dims
            # HACK why is this if statement here? Are there coordinates that can be zero length?
            if len(curdims) == 0:
                curdims = iname
            indtuple = [None] * len(curdims)
            for inum, idim in enumerate(curdims):
                dim_ind = dimnames.index(idim)
                indtuple[inum] = curcoords[dim_ind]
            indtuple = tuple(indtuple)
            cursimdict[iname] = ds[iname].values[indtuple]
        # deal with the ion species by getting them into the list format.
        iondict = {i: cursimdict[i] for i in ionnamelist if i in cursimdict.keys()}
        cursimdict["ionspecies"], cursimdict["ionfracs"] = get_ion_lists(iondict)
        cursimdict = prunedict(cursimdict)
        cursimdict["pfunc"] = pfunc

        # Run the model for the current input
        if mpclient is None:
            outdata = is_snr(**cursimdict)

            # Get everything in the right data arrays.
            for inum, idata in enumerate(outdata):
                if type(idata) is dict:
                    for dname, ival in idata.items():
                        ds[dname].values[tuple(curcoords)] = ival
                else:
                    dname = outnames[inum]
                    ds[dname].values[tuple(curcoords)] = idata
        else:
            cursimdict_list.append(cursimdict)
            curcoords_list.append(curcoords)

    if not mpclient is None:
        from dask.distributed import progress

        futures = mpclient.map(is_snr_mp, curcoords_list, cursimdict_list)
        progress(futures)
        results = mpclient.gather(futures)
        for ires in results:
            curcoords = ires[0]
            outdata = ires[1]
            for inum, idata in enumerate(outdata):
                if type(idata) is dict:
                    for dname, ival in idata.items():
                        ds[dname].values[tuple(curcoords)] = ival
                else:
                    dname = outnames[inum]
                    ds[dname].values[tuple(curcoords)] = idata

    # return the data
    return ds
