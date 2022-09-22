#!/usr/bin/env python
"""
if running via ssh use MPLBACKEND=agg python avstiplotsdump.py
"""

import sys
from pathlib import Path
import time
import argparse
import logging

from mips import read_config_yaml
from mips.isr_mapper import map_radar_array
from mips.isr_plotting import isr_map_plot


def parse_command_line(str_input=None):
    """This will parse through the command line arguments

    Function to go through the command line and if given a list of strings all
    also output a namespace object.

    Parameters
    ----------
    str_input : list
        A list of strings or the input from the command line.

    Returns
    -------
    input_args : Namespace
        An object holding the input arguments wrt the variables.
    """
    scriptpath = Path(sys.argv[0])
    scriptname = scriptpath.name

    formatter = argparse.RawDescriptionHelpFormatter(scriptname)
    width = formatter._width
    title = "Run Mapping program for MIPS[Millstone Is Performance Simulator]"
    shortdesc = (
        "Run script to do performance simulation and put it over a geographic map."
    )
    desc = "\n".join(
        (
            "*" * width,
            "*{0:^{1}}*".format(title, width - 2),
            "*{0:^{1}}*".format("", width - 2),
            "*{0:^{1}}*".format(shortdesc, width - 2),
            "*" * width,
        )
    )
    # desc = "This is the run script for SimVSR."
    # if str_input is None:
    parser = argparse.ArgumentParser(
        description=desc, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    # else:
    #     parser = argparse.ArgumentParser(str_input)

    parser.add_argument(
        "-c",
        "--configfiles",
        dest="configfiles",
        nargs="+",
        required=True,
        help="Names of configuration yaml files.",
    )
    parser.add_argument(
        "-f",
        "--filenames",
        dest="filenames",
        nargs="+",
        default=None,
        help="Files that data is saved to",
        required=False,
    )
    parser.add_argument(
        "-p",
        "--plotfiles",
        dest="plotfiles",
        nargs="+",
        required=False,
        help="Template for the plotted files.",
    )

    parser.add_argument(
        "-l", "--logfile", dest="logfile", help="""Log file name""", default=None
    )
    parser.add_argument(
        "-m",
        "--mp",
        dest="mp",
        help="""Use multiprocessing libraries""",
        action="store_true",
    )

    parser.add_argument(
        "-j",
        "--jobs",
        dest="jobs",
        help="""Number of jobs to run if using a slurm cluster""",
        required=False,
        type=int,
        default=4,
    )
    if str_input is None:
        return parser.parse_args()
    return parser.parse_args(str_input)


def setuplog(logfile=None):
    """Set up the logger object.

    Parameters
    ----------
    logfile : str
        Name of the log file.

    Returns
    -------
    logger : logger
        Logger object.
    """
    logger = logging.getLogger(__file__)
    logger.setLevel(logging.INFO)
    # Create handlers
    c_handler = logging.StreamHandler()
    c_format = logging.Formatter("%(message)s")
    c_handler.setLevel(logging.INFO)
    c_handler.setFormatter(c_format)
    # Add handlers to the logger
    logger.addHandler(c_handler)

    if not logfile is None:
        f_handler = logging.FileHandler(logfile)
        f_handler.setLevel(logging.INFO)
        # Create formatters and add it to handlers
        f_format = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        f_handler.setFormatter(f_format)
        logger.addHandler(f_handler)
    return logger


def runmapping(configfiles, filenames=[], plotfiles=[], logfile=None, mp=False, jobs=4):
    """Main function for this whole thing.

    Parameters
    ----------
    configfiles : list
        List of configuration files.
    filenames : list
        List of output netcdf files.
    plotfiles : list
        List of names for figure files.
    logfile : str
        Name of the log file.
    mp : bool
        Flag to use multiprocessing.
    jobs : int
        Number of jobs if multiprocessing called.
    """
    logger = setuplog(logfile)
    formstr = "%Y-%m-%dT%H:%M:%SZ"
    if mp:
        try:
            from dask.distributed import Client

            client = Client()
            logger.info("Dask avalible client info: " + str(client))
        except:
            logger.info("Dask not avalible, default to single core.")
            mp = False
            client = None
    else:
        client = None

    # Go through the configuration files
    config_type = "mapping"
    sim_list = []
    map_list = []
    fnamesconfig = []
    for iconfig in configfiles:
        file_info = read_config_yaml(iconfig, config_type)
        simdata = file_info["sim"]
        if isinstance(simdata, dict):
            sim_list.append(simdata)
        elif isinstance(simdata, list):
            sim_list += simdata

        mapdata = file_info["map"]
        if isinstance(mapdata, dict):
            map_list.append(mapdata)
        elif isinstance(mapdata, list):
            map_list += mapdata
        if "filenames" in file_info.keys():
            fnamesconfig += file_info["filenames"]

    if not filenames and not fnamesconfig:
        filenames = [None] * len(configfiles)
    elif not filenames and fnamesconfig:
        filenames = fnamesconfig

    if not plotfiles:
        plotfiles = [None] * len(configfiles)

    if len(sim_list) != len(filenames):
        raise ValueError(
            "Config file name list and output file name list are not equal"
        )

    if len(sim_list) != len(plotfiles):
        raise ValueError(
            "Config file name list and figure file name list are not equal"
        )

    for isim, ifilename, imap, iplotname in zip(
        sim_list, filenames, map_list, plotfiles
    ):
        isim["mpclient"] = client
        instr = "Started {0} simulation".format(isim["tname"])

        if logfile is None:
            logprint = print
        else:
            logprint = logger.info
        logprint(instr)

        isim['pfunc'] = logprint
        map_ds = map_radar_array(**isim)
        if not ifilename is None:
            instr1 = "Saving {0} simulation to {1}".format(isim["tname"], ifilename)
            logprint(instr1)
            map_ds.to_netcdf(ifilename, engine="h5netcdf", invalid_netcdf=True)
        if not "map_fname" in imap.keys():
            imap["map_fname"] = plotfiles
        if not imap["map_fname"] is None:
            imap["map_info"] = map_ds
            isr_map_plot(**imap)


if __name__ == "__main__":
    args_commd = parse_command_line()
    arg_dict = {k: v for k, v in args_commd._get_kwargs() if v is not None}
    runmapping(**arg_dict)
