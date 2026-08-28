"""Methods-V7-only EP-flux kernel derived from ``aostools.climate``.

Original code: Martin Jucker, GPLv3, https://github.com/mjucker/aostools.
This vendored subset retains only the all-wave, no-omega, zonal-wind-corrected
path used in this paper.  Unsupported branches raise instead of silently
changing the method.  The notebook records this file's SHA256 in every output.
"""

from __future__ import annotations

import numpy as np

SOURCE_URL = "https://github.com/mjucker/aostools/blob/master/aostools/climate.py"
LICENSE = "GPL-3.0"


def AxRoll(x, ax, invert=False):
    """Move ``ax`` to the front, or restore it after a front-axis operation."""

    position = len(x.shape) + ax if ax < 0 else ax
    return np.rollaxis(x, 0, position + 1) if invert else np.rollaxis(x, position, 0)


def GetAnomaly(x, axis=-1):
    """Remove the mean along one array axis (the zonal anomaly here)."""

    moved = AxRoll(x, axis)
    moved = moved - moved.mean(axis=0)[np.newaxis, ...]
    return AxRoll(moved, axis, invert=True)


def ComputeVertEddy(v, t, p, p0=1.0e3, wave=-1):
    """Return zonal-mean v and all-wave ``v'theta'/d(theta_bar)/dp``."""

    if wave != -1:
        raise ValueError("Methods V7 requires wave=-1 (all waves)")
    kappa = 2.0 / 7.0
    potential_factor = (p0 / p[np.newaxis, :, np.newaxis, np.newaxis]) ** kappa
    dp = np.gradient(p)[np.newaxis, :, np.newaxis]
    theta = t * potential_factor
    v_bar = np.nanmean(v, axis=-1)
    theta_bar = np.nanmean(theta, axis=-1)
    dtheta_dp = np.gradient(theta_bar, axis=1, edge_order=2) / dp
    dtheta_dp[dtheta_dp == 0.0] = np.nan
    # Natural-month calls make this the Methods monthly N2 denominator.
    dtheta_dp = np.nanmean(dtheta_dp, axis=0)[np.newaxis, ...]
    heat_flux = np.nanmean(GetAnomaly(v) * GetAnomaly(theta), axis=-1)
    return v_bar, heat_flux / dtheta_dp


def ComputeEPfluxDiv(lat, pres, u, v, t, w=None, do_ubar=True, wave=-1):
    """Compute the exact Methods-V7 EP-vector and divergence components.

    Inputs have dimensions ``time, pressure, latitude, longitude``; pressure
    is hPa.  Only ``w=None``, ``do_ubar=True``, and ``wave=-1`` are supported.
    Outputs are ``ep1, ep2, div1, div2`` on time, pressure, latitude.
    """

    if w is not None:
        raise ValueError("Methods V7 requires w=None")
    if do_ubar is not True:
        raise ValueError("Methods V7 requires do_ubar=True")
    if wave != -1:
        raise ValueError("Methods V7 requires wave=-1 (all waves)")

    rd, cp = 287.04, 1004.0
    kappa = rd / cp
    omega = 2.0 * np.pi / (24.0 * 3600.0)
    earth_radius = 6.371e6
    latitude_rad = np.asarray(lat, dtype=float) * np.pi / 180.0
    pressure = np.asarray(pres, dtype=float)
    dphi = np.gradient(latitude_rad)[np.newaxis, np.newaxis, :]
    coslat = np.cos(latitude_rad)[np.newaxis, np.newaxis, :]
    sinlat = np.sin(latitude_rad)[np.newaxis, np.newaxis, :]
    inverse_radius = 1.0 / (earth_radius * coslat)
    coriolis = 2.0 * omega * sinlat
    dp = np.gradient(pressure)[np.newaxis, :, np.newaxis]

    u_bar = np.nanmean(u, axis=-1)
    absolute_vorticity = coriolis - (
        inverse_radius
        * np.gradient(u_bar * coslat, axis=2, edge_order=2)
        / dphi
    )
    _, vertical_eddy = ComputeVertEddy(v, t, pressure, 1000.0, wave=-1)
    upvp = np.nanmean(GetAnomaly(u) * GetAnomaly(v), axis=-1)
    shear = np.gradient(u_bar, axis=1, edge_order=2) / dp
    ep1 = -upvp + shear * vertical_eddy
    ep2 = absolute_vorticity * vertical_eddy
    div1 = (
        coslat * np.gradient(ep1, axis=2, edge_order=2) / dphi
        - 2.0 * sinlat * ep1
    ) * inverse_radius
    div2 = np.gradient(ep2, axis=1, edge_order=2) / dp
    return ep1, ep2, div1 * 86400.0, div2 * 86400.0
