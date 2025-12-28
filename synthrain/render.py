"""
PNG rendering utilities.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def save_field_png_full_color(
    path: str,
    xg: np.ndarray,
    yg: np.ndarray,
    z: np.ndarray,
    title: str = "",
    vmin=None,
    vmax=None,
    links: Optional[pd.DataFrame] = None,
    show_links: bool = False,
    xlabel: str = "x",
    ylabel: str = "y",
):
    fig = plt.figure(figsize=(10, 6))
    ax = plt.gca()
    # Use pcolormesh so irregular grids (e.g. inverse Mercator -> lon/lat) plot correctly
    im = ax.pcolormesh(xg, yg, z, shading="auto", vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, label="mm/h")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if show_links and links is not None and len(links) > 0:
        wet = links["wet"].to_numpy(bool)
        # Plot in lon/lat if available, else fall back to x_center/y_center
        xcol = "lon_center" if "lon_center" in links.columns else "x_center"
        ycol = "lat_center" if "lat_center" in links.columns else "y_center"
        ax.scatter(
            links.loc[~wet, xcol],
            links.loc[~wet, ycol],
            s=12,
            marker="x",
            alpha=0.8,
            label="dry",
        )
        ax.scatter(
            links.loc[wet, xcol],
            links.loc[wet, ycol],
            s=18,
            marker="o",
            alpha=0.8,
            label="wet",
        )
        ax.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_field_png(
    path: str,
    xg: np.ndarray,
    yg: np.ndarray,
    z: np.ndarray,
    title: str = "",
    vmin=None,
    vmax=None,
    links: Optional[pd.DataFrame] = None,
    show_links: bool = False,
    xlabel: str = "x",
    ylabel: str = "y",
    min_rain: float = 0.1,
):

    fig = plt.figure(figsize=(10, 6))
    ax = plt.gca()

    # mask zeros so they render as "nothing"
    z = np.asarray(z, dtype=float)
    # z_masked = np.ma.masked_where(z <= 0.0, z)
    z_masked = np.ma.masked_where(z <= min_rain, z)

    # make masked values transparent
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(alpha=0.0)

    # # axes background itself is transparent too
    # ax.set_facecolor("none")

    # pcolormesh so irregular grids plot correctly
    im = ax.pcolormesh(
        xg, yg, z_masked, shading="auto", vmin=vmin, vmax=vmax, cmap=cmap
    )

    plt.colorbar(im, ax=ax, label="mm/h")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if show_links and links is not None and len(links) > 0:
        wet = links["wet"].to_numpy(bool)
        xcol = "lon_center" if "lon_center" in links.columns else "x_center"
        ycol = "lat_center" if "lat_center" in links.columns else "y_center"
        ax.scatter(
            links.loc[~wet, xcol],
            links.loc[~wet, ycol],
            s=12,
            marker="x",
            alpha=0.8,
            label="dry",
        )
        ax.scatter(
            links.loc[wet, xcol],
            links.loc[wet, ycol],
            s=18,
            marker="o",
            alpha=0.8,
            label="wet",
        )
        ax.legend(loc="upper right")

    fig.tight_layout()

    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_links_png(
    path: str, links: pd.DataFrame, bbox, title: str = "Links (centers)"
):
    x_min, x_max, y_min, y_max = bbox
    fig = plt.figure(figsize=(10, 6))
    ax = plt.gca()
    wet = links["wet"].to_numpy(bool)
    xcol = "lon_center" if "lon_center" in links.columns else "x_center"
    ycol = "lat_center" if "lat_center" in links.columns else "y_center"
    ax.scatter(
        links.loc[~wet, xcol],
        links.loc[~wet, ycol],
        s=12,
        marker="x",
        alpha=0.8,
        label="dry",
    )
    ax.scatter(
        links.loc[wet, xcol],
        links.loc[wet, ycol],
        s=18,
        marker="o",
        alpha=0.8,
        label="wet",
    )
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_title(title)
    ax.set_xlabel("lon" if xcol == "lon_center" else "x")
    ax.set_ylabel("lat" if ycol == "lat_center" else "y")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
