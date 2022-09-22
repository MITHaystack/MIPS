#!/usr/bin/env python
"""
setup.py
This is the setup file for the MIPS python package

"""

from pathlib import Path
from setuptools import setup, find_packages

# import versioneer

req = [
    "madrigalWeb",
    "numpy",
    "scipy",
    "matplotlib",
    "xarray",
    "yamale",
    "h5py",
    "iri2016",
    "ISRSpectrum>=3.2.2",
    "h5netcdf",
    "netcdf4",
    "astropy",
    "cartopy",
]
scripts = ["examples/mips_performance_sim.py","examples/mips_zenith_comparison.py","examples/mips_misa_mapping.py","examples/mips_eiscat3d_mapping.py","bin/runmapping.py"]
config = {
    "description": "Millstone IS Performance Simulator",
    "author": "Phil Erickson, Frank Lind, Juha Viernien, John Swoboda, Bill Rideout",
    "url": "https://github.com/MITHaystack/MIPS",
    "version": "v2.1.0",  # versioneer.get_version(),
    # "cmdclass": versioneer.get_cmdclass(),
    "install_requires": req,
    "python_requires": ">=3.0",
    "packages": find_packages(),
    "scripts": scripts,
    "name": "mips",
    "package_data": {"mips": ["*.yaml"]},
}

curpath = Path(__file__)
testpath = curpath.joinpath("figures")
try:
    curpath.mkdir(parents=True, exist_ok=True)
except OSError:
    pass
print("created {}".format(testpath))

setup(**config)
