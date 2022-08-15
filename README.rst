.. -*- mode: rst -*-

MIT Incoherent Scatter Performance Simulator (MIPS)
===================================================


MIPS is a tool for simulating the performance of incoherent scatter radar
systems and networks of such radars. MIPS is a physics
based radar performance model that incorporates first and second order effects
and measurement statistics. The model is parameterized and allows
straightforward optimization and design tradeoffs to be evaluated for IS radar
performance metrics (e.g. SNR, measurement speed, estimation errors, etc).

For incoherent scatter radar (ISR) applications studying the near-Earth space
environment, the instrument design parameter space for a modern Geospace
Radar is very large. Particular areas of design freedom include transmitted
waveforms (e.g. required bandwidth, center frequency, duty cycle), array
configuration (e.g. element number and configuration), spatial diversity
(monostatic vs. locally bistatic), receiver sensitivity tradeoffs, and power-aperture
product choices (e.g. high power with few elements versus low power with many
elements).

Important Links
===============

:Official source code repo: https://github.com/MITHaystack/MIPS
:Issue tracker: https://github.com/MITHaystack/MIPS/issues


Citation
========

If you use MIPS in a scientific publication, we would appreciate a citation such as the following (BibTeX_):

P. J. Erickson, J. Vierinen, F. D. Lind and R. Volz, "The MIT Incoherent Scatter Performance Simulator (MIPS)," 2017 United States National Committee of URSI National Radio Science Meeting
(USNC-URSI NRSM), 2017, pp. 1-2, doi: 10.1109/USNC-URSI-NRSM.2017.7878319.

.. _BibTeX: bibtex.bib


Dependencies
============

The main package components are in the source directory and are divided into MIPS
related libraries and examples of using the simulator.

Build
-----

python

  * madrigalWeb
  * numpy
  * scipy
  * matplotlib
  * xarray
  * yamale
  * h5py
  * iri2016
  * ISRSpectrum>=3.2.2
  * h5netcdf
  * netcdf4
  * astropy
  * cartopy

Installation
============

Make sure the above libraries have been installed.

Create the Python distribution file:

python setup.py bdist_egg

Then install using pip:

pip install -e .


Example Usage
=============

A small set of Python examples can be found in the examples directory in the source tree.

MIPS usage can be somewhat complex due to the wide range of radar configurations,
ionospheric conditions, and subtleties related to coding and radar scattering
geometry.

cd examples

   * python mips_performance_sim.py
   * python mips_zenith_comparison.py
   * python mips_misa_mapping.py
   * python mips_esicat3d_mapping.py

It is recommended that you discuss applications with the development team
prior to use in any publications.


Acknowledgments
===============

This work was supported by the National Science Foundation under the Geospace Facilities program.
We are grateful for the support that made this development possible.
