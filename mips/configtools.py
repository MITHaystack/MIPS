#!/usr/bin/env python
"""
Read config files
"""
from pathlib import Path
import yamale
import numpy as np


def read_config_yaml(yamlfile, schematype):
    """Parse config files.

    The function parses the given file and returns a dictionary with the values.

    Note
    ----
    Sections should be named: siminfo and channels

    Parameters
    ----------
    yamlfile : str
        The name of the file to be read including path.

    Returns
    -------
    objs : dictionay
        Dictionary with name given by [Section] each of which contains.
    """

    # dirname = Path(__file__).expanduser().parent
    # schemafile = dirname / "configschema.yaml"
    schemafile = getschemafile(schematype)
    schema = yamale.make_schema(schemafile)
    data = yamale.make_data(yamlfile)
    d1 = yamale.validate(schema, data)

    return data[0][0]


def getschemafile(schematype):
    """Return the schema file name.

    Parameters
    ----------
    schematype : str
        Type of schema file, either mapping or radar.

    Returns
    : str
        Name of the schema file.
    """
    schema_dict = {"mapping": "map_schema.yaml", "radar": "radar_schema.yaml"}
    dirname = Path(__file__).expanduser().parent
    schemadir = dirname.joinpath("schema")
    return str(schemadir.joinpath(schema_dict[schematype]))


def get_radars(files=None):
    """Gets dictionaries of radars and sites with the key as the name and the value a dictionary of parameters.

    Parameters
    ----------
    files : str or list
        A string or list of strings of yaml files. Can also be a directory with yaml files in them.

    Returns
    -------
    radar_dict : dict
        Dictionary with keys as names of ISRs, values are sub dictionaries of parameters.
    site_dict : dict
        Dictionary with keys as names of ISR sites, values are sub dictionaries of parameters.

    """
    if files is None:
        files = str(Path(__file__).expanduser().parent.joinpath("radar_info"))

    if isinstance(files, str) or isinstance(files, Path):
        files = [files]

    paths = []
    for ifile in files:
        fpath = Path(ifile)
        if fpath.is_dir():
            paths += list(fpath.glob("*.yaml"))
        elif fpath.is_file():
            paths.append(fpath)

    radar_dict = {}
    site_dict = {}
    for ipath in paths:
        file_dict = read_config_yaml(ipath, "radar")
        r_list = file_dict["radars"]
        for iradar in r_list:
            r_name = iradar["name"]
            del iradar["name"]
            radar_dict[r_name] = iradar

        s_list = file_dict.get("sites", [])
        for isite in s_list:
            s_name = isite["name"]
            del isite["name"]
            site_dict[s_name] = isite

    return radar_dict, site_dict


def build_site_lists(sites, file_list=None):
    """Builds list of parameters for radar sites.

    Parameters
    ----------
    sites : list
        List of site names.
    file_list : str or list
        A string or list of strings of yaml files. Can also be a directory with yaml files in them.

    Returns
    -------
    lats : list
        Latitudes of the sites in degrees.
    lons : list
        Longitudes of the sites in degrees.
    alts : list
        Altitudes of the sites in meters.
    mask : list
        Elevation masks in degrees.
    """
    _, isr_sites = get_radars(file_list)

    lats = []
    lons = []
    alts = []
    masks = []
    # print sites
    for s in sites:
        # print 'load %s' % (s)
        d = isr_sites.get(s, None)
        if d is None:
            print("unknown site %s in build_site_lists, ignored." % (s))
        else:
            lats.append(d["latitude"])
            lons.append(d["longitude"])
            alts.append(d["altitude"])
            masks.append(d["elevation_mask"])

    return (lats, lons, alts, masks)


def build_radar_lists(radars, file_list=None):
    """Builds list of parameters for radars.

    Parameters
    ----------
    radars : list
        List of site names.
    file_list : str or list
        A string or list of strings of yaml files. Can also be a directory with yaml files in them.

    Returns
    -------
    rtype : list
        Antenna types of the radars.
    boresite : list
        Boresite location of the antennas.
    smask : list
        Steering mask of the antennas
    freq : list
        Center frequncies of the radars in Hz.
    txg : list
        Gain of the transmit antenna in dB.
    rxg : list
        Gain of the receive antenna in dB.
    txpwr : list
        Power of the transmitter in Watts.
    duty : list
        Duty cycle as a ratio, i.e. always <=1.
    tsys_type : list
        Type of tsys for the receiver.
    xtra_tsys : list
        Extra tsys in deg K.
    """
    r_dict, _ = get_radars(file_list)
    rtype = []
    boresite = []
    smask = []
    freq = []
    txg = []
    rxg = []
    txpwr = []
    duty = []
    tsys_type = []
    xtra_tsys = []
    notes = []

    # print radars
    for r in radars:
        # print 'load %s' % (r)
        d = r_dict.get(r, None)
        if d is None:
            print("unknown radar system type %s in build_radar_lists" % (r))
        else:
            rtype.append(d["ant_type"])
            boresite.append(np.array([d["az_rotation"], d["el_tilt"]]))
            smask.append(d["steering_mask"])
            freq.append(d["freq"])
            txg.append(d["tx_gain"])
            rxg.append(d["rx_gain"])
            txpwr.append(d["tx_power"])
            duty.append(d["duty"])
            tsys_type.append(d["tsys_type"])
            xtra_tsys.append(d["xtra_tsys"])
            notes.append(d["notes"])

    return (
        rtype,
        boresite,
        smask,
        freq,
        txg,
        rxg,
        txpwr,
        duty,
        tsys_type,
        xtra_tsys,
        notes,
    )
