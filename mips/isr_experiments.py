"""
    ISR Experiment Description Library

    These methods allow for the loading of an experiment description in YAML and
    the simulation of these experiments in time and space using MIPS. The
    experiment description allows for sequencing basic operational behaviors for
    each radar system in the experiment. Interleaved modes are supported which
    allow for different simultaneous measurements to be simulated. Basic
    routines are provided to support visualization of the results.

    This is currently under development.

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


class ISExperimentDescription:
    def __init__():
        pass

    def loadExperiment(fname='default_experiment.yaml'):
        pass

    def startExperiment(start_time='2022-01-01T00:00:00Z', end_time='2022-01-01T12:00:00Z'):
        pass

    def resetExperiment():
        pass

    def stepExperiment():
        pass

    def getMeasurementState():
        pass


def simulate_measurement_state(state):
    pass

def output_is_simulation(state, sim_measurements):
    pass

def simulate_experiment(experiment_fname):
    pass
