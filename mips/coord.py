"""
    Coordinate transformation functions
"""
import os
import warnings

from datetime import datetime, timedelta
from time import mktime

from numpy import (
    power,
    degrees,
    radians,
    mat,
    cos,
    sin,
    arctan,
    sqrt,
    pi,
    arctan2,
    array,
    transpose,
    dot,
    arccos,
    sign,
)
import math
import numpy as np


def cbrt(x):
    if x >= 0:
        return power(x, 1.0 / 3.0)
    else:
        return -power(abs(x), 1.0 / 3.0)


# Constants defined by the World Geodetic System 1984 (WGS84)
a = 6378.137 * 1e3
b = 6356.7523142 * 1e3
esq = 6.69437999014 * 0.001
e1sq = 6.73949674228 * 0.001
f = 1 / 298.257223563


def geodetic2ecef(lat, lon, alt):
    """
    Convert geodetic coordinates to ECEF.
    @lat, @lon in decimal degrees
    @alt in meters
    """
    lat, lon = radians(lat), radians(lon)
    xi = sqrt(1 - esq * sin(lat) ** 2)
    x = (a / xi + alt) * cos(lat) * cos(lon)
    y = (a / xi + alt) * cos(lat) * sin(lon)
    z = (a / xi * (1 - esq) + alt) * sin(lat)
    return np.array([x, y, z])


def enu2ecef(lat, lon, alt, e, n, u):
    """NED (north/east/down) to ECEF coordinate system conversion."""
    x, y, z = e, n, u
    lat, lon = radians(lat), radians(lon)
    mx = array(
        [
            [-sin(lon), -sin(lat) * cos(lon), cos(lat) * cos(lon)],
            [cos(lon), -sin(lat) * sin(lon), cos(lat) * sin(lon)],
            [0, cos(lat), sin(lat)],
        ]
    )
    enu = array([x, y, z])
    res = dot(mx, enu)
    return res


def ned2ecef(lat, lon, alt, n, e, d):
    """NED (north/east/down) to ECEF coordinate system conversion."""
    x, y, z = e, n, -1.0 * d
    lat, lon = radians(lat), radians(lon)
    mx = array(
        [
            [-sin(lon), -sin(lat) * cos(lon), cos(lat) * cos(lon)],
            [cos(lon), -sin(lat) * sin(lon), cos(lat) * sin(lon)],
            [0, cos(lat), sin(lat)],
        ]
    )
    enu = array([x, y, z])
    res = dot(mx, enu)
    return res


def azel_ecef(lat, lon, alt, az, el):
    """Radar pointing (az,el) to unit vector in ECEF."""
    return ned2ecef(
        lat,
        lon,
        alt,
        cos(-radians(az)) * cos(radians(el)),
        -sin(-radians(az)) * cos(radians(el)),
        -sin(radians(el)),
    )


def ecef2geodetic(x, y, z):
    """Convert ECEF coordinates to geodetic.
    J. Zhu, "Conversion of Earth-centered Earth-fixed coordinates \
    to geodetic coordinates," IEEE Transactions on Aerospace and \
    Electronic Systems, vol. 30, pp. 957-961, 1994."""
    r = sqrt(x * x + y * y)
    Esq = a * a - b * b
    F = 54 * b * b * z * z
    G = r * r + (1 - esq) * z * z - esq * Esq
    C = (esq * esq * F * r * r) / (pow(G, 3))
    S = cbrt(1 + C + sqrt(C * C + 2 * C))
    P = F / (3 * pow((S + 1 / S + 1), 2) * G * G)
    Q = sqrt(1 + 2 * esq * esq * P)
    r_0 = -(P * esq * r) / (1 + Q) + sqrt(
        0.5 * a * a * (1 + 1.0 / Q)
        - P * (1 - esq) * z * z / (Q * (1 + Q))
        - 0.5 * P * r * r
    )
    U = sqrt(pow((r - esq * r_0), 2) + z * z)
    V = sqrt(pow((r - esq * r_0), 2) + (1 - esq) * z * z)
    Z_0 = b * b * z / (a * V)
    h = U * (1 - b * b / (a * V))
    lat = arctan((z + e1sq * Z_0) / r)
    lon = arctan2(y, x)
    return array([degrees(lat), degrees(lon), h])


def geodetic_to_az_el_r(obs_lat, obs_lon, obs_h, target_lat, target_lon, target_h):
    """When given a observer lat,long,h and target lat,long,h, provide azimuth, elevation, and range to target"""
    up = ned2ecef(obs_lat, obs_lon, obs_h, 0.0, 0.0, -1.0)
    north = ned2ecef(obs_lat, obs_lon, obs_h, 1.0, 0.0, 0.0)
    east = ned2ecef(obs_lat, obs_lon, obs_h, 0.0, 1.0, 0.0)
    obs = array(geodetic2ecef(obs_lat, obs_lon, obs_h))
    target = array(geodetic2ecef(target_lat, target_lon, target_h))
    p_vec = target - obs
    az_p = dot(p_vec, north) * north + dot(p_vec, east) * east
    azs = sign(dot(p_vec, east))

    elevation = (
        90.0
        - 180.0
        * arccos(dot(p_vec, up) / (sqrt(dot(p_vec, p_vec)) * sqrt(dot(up, up))))
        / math.pi
    )
    azimuth = (
        azs
        * 180.0
        * arccos(dot(az_p, north) / (sqrt(dot(az_p, az_p)) * sqrt(dot(north, north))))
        / math.pi
    )
    target_range = sqrt(dot(p_vec, p_vec))

    return array([azimuth, elevation, target_range])


def az_el_r2geodetic(obs_lat, obs_lon, obs_h, az, el, r):
    """When given a observer lat,long,h and az,el and r, return lat,long,h of target"""
    x = (
        geodetic2ecef(obs_lat, obs_lon, obs_h)
        + azel_ecef(obs_lat, obs_lon, obs_h, az, el) * r
    )
    llh = ecef2geodetic(x[0], x[1], x[2])
    if llh[1] < 0.0:
        llh[1] = llh[1] + 360.0
    return llh


def antenna_to_cartesian(ranges, azimuths, elevations):
    """
    From ARM-DOE pyart library

    Return Cartesian coordinates from antenna coordinates.
    Parameters
    ----------
    ranges : array
        Distances to the center of the radar gates (bins) in kilometers.
    azimuths : array
        Azimuth angle of the radar in degrees.
    elevations : array
        Elevation angle of the radar in degrees.
    Returns
    -------
    x, y, z : array
        Cartesian coordinates in meters from the radar.
    Notes
    -----
    The calculation for Cartesian coordinate is adapted from equations
    2.28(b) and 2.28(c) of Doviak and Zrnic [1]_ assuming a
    standard atmosphere (4/3 Earth's radius model).
    .. math:
        z = \\sqrt{r^2+R^2+2*r*R*sin(\\theta_e)} - R
        s = R * arcsin(\\frac{r*cos(\\theta_e)}{R+z})
        x = s * sin(\\theta_a)
        y = s * cos(\\theta_a)
    Where r is the distance from the radar to the center of the gate,
    :math:`\\theta_a` is the azimuth angle, :math:`\\theta_e` is the
    elevation angle, s is the arc length, and R is the effective radius
    of the earth, taken to be 4/3 the mean radius of earth (6371 km).
    References
    ----------
    .. [1] Doviak and Zrnic, Doppler Radar and Weather Observations, Second
        Edition, 1993, p. 21.
    """
    theta_e = elevations * np.pi / 180.0    # elevation angle in radians.
    theta_a = azimuths * np.pi / 180.0      # azimuth angle in radians.
    R = 6371.0 * 1000.0 * 4.0 / 3.0     # effective radius of earth in meters.
    r = ranges * 1000.0                 # distances to gates in meters.

    z = (r ** 2 + R ** 2 + 2.0 * r * R * np.sin(theta_e)) ** 0.5 - R
    s = R * np.arcsin(r * np.cos(theta_e) / (R + z))  # arc length in m.
    x = s * np.sin(theta_a)
    y = s * np.cos(theta_a)
    return x, y, z


def geographic_to_cartesian(lon, lat, projparams):
    """
    From ARM-DOE pyart library

    Geographic to Cartesian coordinate transform.
    Transform a set of Geographic coordinate (lat, lon) to a
    Cartesian/Cartographic coordinate (x, y) using pyproj or a build in
    Azimuthal equidistant projection.
    Parameters
    ----------
    lon, lat : array-like
        Geographic coordinates in degrees.
    projparams : dict or str
        Projection parameters passed to pyproj.Proj. If this parameter is a
        dictionary with a 'proj' key equal to 'pyart_aeqd' then a azimuthal
        equidistant projection will be used that is native to Py-ART and
        does not require pyproj to be installed. In this case a non-default
        value of R can be specified by setting the 'R' key to the desired
        value.
    Returns
    -------
    x, y : array-like
        Cartesian coordinates in meters unless projparams defines a value for R
        in different units.
    """
    if isinstance(projparams, dict) and projparams.get('proj') == 'pyart_aeqd':
        # Use Py-ART's Azimuthal equidistance projection
        lon_0 = projparams['lon_0']
        lat_0 = projparams['lat_0']
        if 'R' in projparams:
            R = projparams['R']
            x, y = geographic_to_cartesian_aeqd(lon, lat, lon_0, lat_0, R)
        else:
            x, y = geographic_to_cartesian_aeqd(lon, lat, lon_0, lat_0)
    #else:
        # Use pyproj for the projection
        # check that pyproj is available
        #if not _PYPROJ_AVAILABLE:
        #    raise MissingOptionalDependency(
        #        "PyProj is required to use geographic_to_cartesian "
        #        "with a projection other than pyart_aeqd but it is not "
        #        "installed")
    #    proj = pyproj.Proj(projparams)
    #    x, y = proj(lon, lat, inverse=False)
    return x, y

def geographic_to_cartesian_aeqd(lon, lat, lon_0, lat_0, R=6370997.):
    """
    Azimuthal equidistant geographic to Cartesian coordinate transform.
    Transform a set of geographic coordinates (lat, lon) to
    Cartesian/Cartographic coordinates (x, y) using a azimuthal equidistant
    map projection [1]_.
    .. math:
        x = R * k * \\cos(lat) * \\sin(lon - lon_0)
        y = R * k * [\\cos(lat_0) * \\sin(lat) -
                     \\sin(lat_0) * \\cos(lat) * \\cos(lon - lon_0)]
        k = c / \\sin(c)
        c = \\arccos(\\sin(lat_0) * \\sin(lat) +
                     \\cos(lat_0) * \\cos(lat) * \\cos(lon - lon_0))
    Where x, y are the Cartesian position from the center of projection;
    lat, lon the corresponding latitude and longitude; lat_0, lon_0 are the
    latitude and longitude of the center of the projection; R is the radius of
    the earth (defaults to ~6371 km).
    Parameters
    ----------
    lon, lat : array-like
        Longitude and latitude coordinates in degrees.
    lon_0, lat_0 : float
        Longitude and latitude, in degrees, of the center of the projection.
    R : float, optional
        Earth radius in the same units as x and y. The default value is in
        units of meters.
    Returns
    -------
    x, y : array
        Cartesian coordinates in the same units as R, typically meters.
    References
    ----------
    .. [1] Snyder, J. P. Map Projections--A Working Manual. U. S. Geological
        Survey Professional Paper 1395, 1987, pp. 191-202.
    """
    lon = np.atleast_1d(np.asarray(lon))
    lat = np.atleast_1d(np.asarray(lat))

    lon_rad = np.deg2rad(lon)
    lat_rad = np.deg2rad(lat)

    lat_0_rad = np.deg2rad(lat_0)
    lon_0_rad = np.deg2rad(lon_0)

    lon_diff_rad = lon_rad - lon_0_rad

    # calculate the arccos after ensuring all values in valid domain, [-1, 1]
    arg_arccos = (np.sin(lat_0_rad) * np.sin(lat_rad) +
                  np.cos(lat_0_rad) * np.cos(lat_rad) * np.cos(lon_diff_rad))
    arg_arccos[arg_arccos > 1] = 1
    arg_arccos[arg_arccos < -1] = -1
    c = np.arccos(arg_arccos)
    with warnings.catch_warnings():
        # division by zero may occur here but is properly addressed below so
        # the warnings can be ignored
        warnings.simplefilter("ignore", RuntimeWarning)
        k = c / np.sin(c)
    # fix cases where k is undefined (c is zero), k should be 1
    k[c == 0] = 1

    x = R * k * np.cos(lat_rad) * np.sin(lon_diff_rad)
    y = R * k * (np.cos(lat_0_rad) * np.sin(lat_rad) -
                 np.sin(lat_0_rad) * np.cos(lat_rad) * np.cos(lon_diff_rad))
    return x, y

def cartesian_to_geographic(x, y, projparams):
    """
    Cartesian to Geographic coordinate transform.
    Transform a set of Cartesian/Cartographic coordinates (x, y) to a
    geographic coordinate system (lat, lon) using pyproj or a build in
    Azimuthal equidistant projection.
    Parameters
    ----------
    x, y : array-like
        Cartesian coordinates in meters unless R is defined in different units
        in the projparams parameter.
    projparams : dict or str
        Projection parameters passed to pyproj.Proj. If this parameter is a
        dictionary with a 'proj' key equal to 'pyart_aeqd' then a azimuthal
        equidistant projection will be used that is native to Py-ART and
        does not require pyproj to be installed. In this case a non-default
        value of R can be specified by setting the 'R' key to the desired
        value.
    Returns
    -------
    lon, lat : array
        Longitude and latitude of the Cartesian coordinates in degrees.
    """
    if isinstance(projparams, dict) and projparams.get('proj') == 'pyart_aeqd':
        # Use Py-ART's Azimuthal equidistance projection
        lon_0 = projparams['lon_0']
        lat_0 = projparams['lat_0']
        if 'R' in projparams:
            R = projparams['R']
            lon, lat = cartesian_to_geographic_aeqd(x, y, lon_0, lat_0, R)
        else:
            lon, lat = cartesian_to_geographic_aeqd(x, y, lon_0, lat_0)
    #else:
        # Use pyproj for the projection
        # check that pyproj is available
        #if not _PYPROJ_AVAILABLE:
        #    raise MissingOptionalDependency(
        #        "PyProj is required to use cartesian_to_geographic "
        #        "with a projection other than pyart_aeqd but it is not "
        #        "installed")
    #    proj = pyproj.Proj(projparams)
    #    lon, lat = proj(x, y, inverse=True)
    return lon, lat

def cartesian_to_geographic_aeqd(x, y, lon_0, lat_0, R=6370997.):
    """
    Azimuthal equidistant Cartesian to geographic coordinate transform.
    Transform a set of Cartesian/Cartographic coordinates (x, y) to
    geographic coordinate system (lat, lon) using a azimuthal equidistant
    map projection [1]_.
    .. math:
        lat = \\arcsin(\\cos(c) * \\sin(lat_0) +
                       (y * \\sin(c) * \\cos(lat_0) / \\rho))
        lon = lon_0 + \\arctan2(
            x * \\sin(c),
            \\rho * \\cos(lat_0) * \\cos(c) - y * \\sin(lat_0) * \\sin(c))
        \\rho = \\sqrt(x^2 + y^2)
        c = \\rho / R
    Where x, y are the Cartesian position from the center of projection;
    lat, lon the corresponding latitude and longitude; lat_0, lon_0 are the
    latitude and longitude of the center of the projection; R is the radius of
    the earth (defaults to ~6371 km). lon is adjusted to be between -180 and
    180.
    Parameters
    ----------
    x, y : array-like
        Cartesian coordinates in the same units as R, typically meters.
    lon_0, lat_0 : float
        Longitude and latitude, in degrees, of the center of the projection.
    R : float, optional
        Earth radius in the same units as x and y. The default value is in
        units of meters.
    Returns
    -------
    lon, lat : array
        Longitude and latitude of Cartesian coordinates in degrees.
    References
    ----------
    .. [1] Snyder, J. P. Map Projections--A Working Manual. U. S. Geological
        Survey Professional Paper 1395, 1987, pp. 191-202.
    """
    x = np.atleast_1d(np.asarray(x))
    y = np.atleast_1d(np.asarray(y))

    lat_0_rad = np.deg2rad(lat_0)
    lon_0_rad = np.deg2rad(lon_0)

    rho = np.sqrt(x*x + y*y)
    c = rho / R

    with warnings.catch_warnings():
        # division by zero may occur here but is properly addressed below so
        # the warnings can be ignored
        warnings.simplefilter("ignore", RuntimeWarning)
        lat_rad = np.arcsin(np.cos(c) * np.sin(lat_0_rad) +
                            y * np.sin(c) * np.cos(lat_0_rad) / rho)
    lat_deg = np.rad2deg(lat_rad)
    # fix cases where the distance from the center of the projection is zero
    lat_deg[rho == 0] = lat_0

    x1 = x * np.sin(c)
    x2 = rho*np.cos(lat_0_rad)*np.cos(c) - y*np.sin(lat_0_rad)*np.sin(c)
    lon_rad = lon_0_rad + np.arctan2(x1, x2)
    lon_deg = np.rad2deg(lon_rad)
    # Longitudes should be from -180 to 180 degrees
    lon_deg[lon_deg > 180] -= 360.
    lon_deg[lon_deg < -180] += 360.

    return lon_deg, lat_deg


def test_coord():
    result = geodetic2ecef(69.0, 19.0, 10.0)
    result3 = ned2ecef(69.0, 19.0, 10.0, 10679.6, 1288.2, 49873.3)

    print("North")
    print(geodetic_to_az_el_r(42.61950, 288.50827, 146.0, 43.61950, 288.50827, 100e3))
    print("az_el_r2geodetic")
    print(
        az_el_r2geodetic(
            42.61950, 288.50827, 146.0, 0.00000000e00, 4.12258606e01, 1.50022653e05
        )
    )
    print("East")
    print(geodetic_to_az_el_r(42.61950, 288.50827, 146.0, 42.61950, 289.50827, 100e3))
    print("West")
    print(geodetic_to_az_el_r(42.61950, 288.50827, 146.0, 42.61950, 287.50827, 100e3))
    print("South")
    print(geodetic_to_az_el_r(42.61950, 288.50827, 146.0, 41.61950, 288.50827, 100e3))
    print("Southwest")
    print(geodetic_to_az_el_r(42.61950, 288.50827, 146.0, 41.61950, 287.50827, 100e3))


if __name__ == "__main__":
    test_coord()
