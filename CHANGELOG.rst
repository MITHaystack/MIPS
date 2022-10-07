===============
MIPS Change Log
===============

.. current developments

vv2.2.1
====================

**Added:**

* Sphinx Documentation
* Added pre-commit configuration.

**Changed:**

* Added Dask clients to example scripts.
* Updated figures and example data.



vv2.2.0
====================

**Added:**

* Mapping script runmapping.py which uses yaml files to create maps.
* Yaml based mapping examples.
* Added schema for mapping yaml files.
* Added yaml schema for radar systems.
* Added radar info file.

**Changed:**

* Radar systems are now represented within yaml files.
* Parameterize simulation via baud length in seconds and number of bauds instead of pulse length.
* Mapping can be parameterized by ipp instead of using duty cyle of the radar.
* Takes into account number of bauds for measurement time. Reduces measurement time by a factor of N(N-1) where N is number of bauds.

**Removed:**

* Removed isr_sites.py

**Fixed:**

* Fixed bistatic volume factor measurement.
* Measurement times derived for mapping examples are more reasonable.


