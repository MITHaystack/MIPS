#!/usr/bin/env python
"""
Plotting output data
"""
from pathlib import Path
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib.ticker as mticker
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter


def plot_grid(lat0, lon0, coef=0.7, map_type="normal"):
    """Plotting projection helper.

    Parameters
    ----------
    lat0 : float
        Center latitude for projection.
    lon : float
        Center longitude for projection.
    coef : float
        Zoom in coeffiecent. Small means more zoom.
    map_type : string
        Type of projection. Can be normal, north_polar, south_polar or full_globe.

    Returns
    -------
    fig : matplotlib.fig
        Figure object from matplotlib.
    ax : matplotlib.ax
        Axes object from matplotlib.
    proj : ccrs.proj
        Projection object form cartopy
    """

    if map_type == "full_globe":
        print("full globe projection")
        proj = ccrs.LambertCylindrical(central_longitude=lon0)

        # m = Basemap(resolution='l',projection='cyl',lat_0=lat0,lon_0=lon0)
        lat_spacing = 20.0
        lon_spacing = 36.0
    elif map_type == "north_polar":
        # m = Basemap(resolution='l',projection='npstere',boundinglat=60.0,lat_0=lat0,lon_0=lon0)
        proj = ccrs.NorthPolarStereo(central_longitude=lon0)
        lat_spacing = 20.0
        lon_spacing = 36.0
    elif map_type == "south_polar":
        mproj = ccrs.SouthPolarStereo(central_longitude=lon0)
        lat_spacing = 20.0
        lon_spacing = 36.0
    else:
        # normal projection
        proj = ccrs.Orthographic(
            central_longitude=lon0, central_latitude=lat0, globe=None
        )

        # m = Basemap(resolution=resolution,projection='ortho',lat_0=lat0,lon_0=lon0,
        #             llcrnrx=-coef*(m0.urcrnrx - m0.llcrnrx)/2.,
        #             llcrnry=-coef*(m0.urcrnry - m0.llcrnry)/2.,
        #             urcrnrx=coef*(m0.urcrnrx - m0.llcrnrx)/2.,
        #             urcrnry=coef*(m0.urcrnry - m0.llcrnry)/2.)

        lat_spacing = 10.0
        lon_spacing = 10.0
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1, projection=proj)
    old_ext = ax.get_extent(crs=proj)
    new_ext = (
        -coef * 0.5 * (old_ext[1] - old_ext[0]),
        coef * 0.5 * (old_ext[1] - old_ext[0]),
        -coef * 0.5 * (old_ext[3] - old_ext[2]),
        coef * 0.5 * (old_ext[3] - old_ext[2]),
    )

    ax.set_extent(new_ext, crs=proj)
    ax.coastlines()
    gl = ax.gridlines(draw_labels=True, linestyle="--")
    gl.xlocator = mticker.FixedLocator(np.arange(-180, 180, lon_spacing))
    gl.ylocator = mticker.FixedLocator(np.arange(-90.0, 90, lat_spacing))
    gl.xformatter = LongitudeFormatter()
    gl.yformatter = LatitudeFormatter()

    return fig, ax, proj


def isr_map_plot(
    map_info,
    map_parameters,
    dval_max,
    map_zoom,
    range_contours,
    map_fname,
    map_type="normal",
    annotate=True,
    legend=True,
    vmin=0.1,
    vmax=-1,
    extent=None,
):
    """
    Take the information produced by the IS mapping code and produce a map plot with proper annotations.

    Parameters
    ----------
    map_info : xarray.Dataset
        Result of the ISR mapping simulation.
    map_parameters : list
        List of desired map parameters to plot.
    dval_max : list
        List of max values to for the colorbar.
    map_zoom : float
        How much to zoom in, lower number more zoom.
    range_contours : bool
        Make a range contour plot.
    map_fname : string
        Name template.
    map_type : string
        Type of projection. Can be normal, north_polar, south_polar or full_globe.
    annotate : Bool
        Add annotation to maps.
    legend : bool
        Add a legend.
    vmin : float
        Minimum for plot values.
    vmax : float
        Maximum for plot values.
    extent : dict
        Dictionary to show the extent of the plots.

    """
    map_fname = Path(map_fname)
    tx_lat = map_info.attrs["tx_lat"]
    tx_lon = map_info.attrs["tx_lon"]
    rx_lat = map_info.attrs["rx_lat"]
    rx_lon = map_info.attrs["rx_lon"]
    title = map_info.attrs["map_title"]
    if annotate:
        annotation = map_info.attrs["annotate_txt"]

    tx_range_mat = map_info["tx_to_target_range_m"].values
    dataproj = ccrs.PlateCarree()
    for midx, mp in enumerate(map_parameters):

        dval_lim = dval_max[midx]
        # print midx, mp
        if mp == "speed":

            dval_mat = map_info["delta_t_mat_tot"].values
            ptype = "Measurement speed (time to 5% uncertainty)"
            map_final_fname = map_fname
            log_scale = True
        else:

            mfn_split = map_fname.stem
            log_scale = False
            if mp == "dNe":
                dval_mat = map_info["dNe_tot"].values
                ptype = r"Electron density error $\sigma(Ne) (m^{-3})$"
                extstr = "_dNe"
                map_final_fname = mfn_split[0] + "_dNe." + mfn_split[1]
            elif mp == "dTi":
                dval_mat = map_info["dTi_tot"].values
                ptype = r"Ion temperature error $\mathrm{stddev}(Ti) (K)$"
                extstr = "_dTi"
            elif mp == "dTe":
                dval_mat = map_info["dTe_tot"].values
                extstr = "_dTe"
                ptype = r"Electron temperature error $\mathrm{stddev}(Te) (K)$"

            elif mp == "dV":
                dval_mat = map_info["dV_tot"].values
                ptype = r"Velocity error $\mathrm{stddev}(V) (m/s)$"
                extstr = "_dV"

            elif mp == "gamma":
                dval_mat = 90.0 - np.abs(90 - map_info["tx_target_rx_angle"].values[0])
                ptype = "Bistatic Angle in Degrees"
                extstr = "_gamma"
            map_final_fname = map_fname.parent.joinpath(
                map_fname.stem + extstr + map_fname.suffix
            )

        if extent == None:
            fig, ax, proj = plot_grid(
                map_info.attrs["mean_lat"],
                map_info.attrs["mean_lon"],
                coef=map_zoom,
                map_type=map_type,
            )
        else:
            center_lat = extent["center_lat"]
            center_lon = extent["center_lon"]
            fig, ax, proj = plot_grid(
                center_lat, center_lon, coef=map_zoom, map_type=map_type
            )

        longs_g, lats_g = np.meshgrid(map_info["long"].values, map_info["lat"].values)
        dval = ma.masked_invalid(dval_mat)
        cm = plt.get_cmap("viridis").copy()
        cm.set_bad("gray", alpha=0.0)

        if log_scale:

            if vmax < 0:
                # let maximum float
                cs = ax.pcolormesh(
                    longs_g,
                    lats_g,
                    np.log10(dval),
                    vmax=np.log10(dval_lim),
                    vmin=np.log10(vmin),
                    cmap=cm,
                    transform=dataproj,
                )
            else:
                cs = ax.pcolormesh(
                    longs_g,
                    lats_g,
                    np.log10(dval),
                    vmin=np.log10(vmin),
                    vmax=np.log10(vmax),
                    cmap=cm,
                    transform=dataproj,
                )
        else:
            if vmax < 0:
                # let maximum float
                cs = ax.pcolormesh(
                    longs_g,
                    lats_g,
                    dval,
                    vmax=dval_lim,
                    vmin=vmin,
                    cmap=cm,
                    transform=dataproj,
                )
            else:
                cs = ax.pcolormesh(
                    longs_g,
                    lats_g,
                    dval,
                    vmax=vmax,
                    vmin=vmin,
                    cmap=cm,
                    transform=dataproj,
                )

        if mp == "speed":
            cb = plt.colorbar(cs, ticks=[0, 1, 2, 3])
        else:
            cb = plt.colorbar(cs)

        if mp == "speed":
            cb.ax.set_yticklabels(["1", "10", "100", "1000"])
            cb.set_label("Time to 5% uncertainty (seconds)")
        elif mp == "dNe":
            cb.set_label("(%)")
        elif mp == "dTi":
            cb.set_label("(K)")
        elif mp == "dTe":
            cb.set_label("(K)")
        elif mp == "dV":
            cb.set_label("(m/s)")
        elif mp == "gamma":
            cb.set_label("deg")
        else:
            print("Unknown map type %s default label" % (mp))
            cb.set_label("unknown")

        if range_contours:
            for rg in range_contours:
                cs = ax.contour(
                    longs_g,
                    lats_g,
                    tx_range_mat[rg, :, :] / 1e3,
                    [300, 600, 1200, 1800, 2400],
                    linestyles="dashed",
                    linewidths=1.0,
                    colors="black",
                    transform=dataproj,
                )
                ax.clabel(
                    cs,  # Typically best results when labelling line contours.
                    colors=["black"],
                    manual=False,  # Automatic placement vs manual placement.
                    inline=True,  # Cut the line where the label will be placed.
                    fmt="%1.0f km",  # Labes as integers, with some extra space.
                )
        # plt.clabel(cs, fontsize=9, inline=1,  fmt="%1.0f km")

        plt.title(title + "\n" + ptype)
        if annotate:
            plt.annotate(
                annotation,
                xy=(0.05, 0.95),
                xycoords="axes fraction",
                fontsize=6,
                backgroundcolor=(1, 1, 1, 0.5),
                va="top",
            )

        # Plot tx and rx locations
        for itx_lon, itx_lat in zip(tx_lon, tx_lat):
            ax.scatter(
                x=itx_lon,
                y=itx_lat,
                color="red",
                marker="^",
                s=50,
                label="TX",
                transform=dataproj,
            )
        for irx_lon, irx_lat in zip(rx_lon, rx_lat):
            ax.scatter(
                x=irx_lon,
                y=irx_lat,
                color="black",
                marker="v",
                s=50,
                label="RX",
                transform=dataproj,
            )

        if legend:
            plt.legend()

        if map_fname != None:
            print("write figure %s " % (map_final_fname))
            plt.savefig(map_final_fname)
            plt.close(fig)
        else:
            plt.show()


def frequencyplots():
    """Plot paramerters across different center frequencies."""
    lstr = (
        "%.0f km alt\npl=%.0f ms\nne=%.0e m$^{-3}$\n%.0fMW peak @ d=%.0f%%\nTe=%.0f,Ti=%.0f\n$A_{\mathrm{eff}}=%1.0f$ m$^{2}$"
        % (rng / 1e3, tpulse * 1e3, ne, pwr / 1e6, duty * 100, te, ti, Aeff)
    )
    f, ax = pylab.subplots(2, 1, sharex=True)
    for i in range(len(excess_tsys)):
        ax[0].plot(
            frequencies / 1e6,
            snr_sweep[i],
            label="$T_{\mathrm{rx}}=%1.0f$ K" % (excess_tsys[i]),
        )
    ax[0].legend(fontsize=8)
    ax[0].text(1050, 1.0, lstr, fontsize=8, backgroundcolor=(1, 1, 1, 0.5))
    ax[0].set_title("IS Radar Performance: Fixed Antenna Area")
    ax[0].set_ylabel("SNR")
    ax[0].set_ylim(0, 6.0)
    ax[0].set_xlim(0, 1300)
    ax[0].grid(True)

    lstr = "%.0f m^2 Aeff" % Aeff
    for i in range(len(excess_tsys)):
        ax[1].plot(
            frequencies / 1e6,
            mtime_sweep[i],
            label="$T_{\mathrm{rx}}=%1.0f$ K" % (excess_tsys[i]),
        )

    ax[1].set_xlabel("Freq (MHz)")
    ax[1].set_ylabel("Meas time (sec)")
    ax[1].set_ylim(0, 100)
    ax[1].legend(fontsize=8)
    ax[1].grid(True)

    f.savefig("figures/is_sim_fixed_ant_area.png")
