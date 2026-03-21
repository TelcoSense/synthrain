"""Rendering utilities for scenario outputs."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "font.size": 14,
        "axes.titlesize": 18,
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
    }
)

pad_inches: float = 0.0


def _scatter_links(ax, links: pd.DataFrame) -> None:
    wet = links["wet"].to_numpy(bool)
    available = (
        links["available"].to_numpy(bool)
        if "available" in links.columns
        else np.ones(len(links), dtype=bool)
    )
    xcol = "lon_center" if "lon_center" in links.columns else "x_center"
    ycol = "lat_center" if "lat_center" in links.columns else "y_center"

    unavailable = ~available
    dry = available & ~wet
    wet_ok = available & wet

    if np.any(unavailable):
        ax.scatter(
            links.loc[unavailable, xcol],
            links.loc[unavailable, ycol],
            s=20,
            marker="s",
            alpha=0.9,
            color="tab:gray",
            label="unavailable",
        )
    if np.any(dry):
        ax.scatter(
            links.loc[dry, xcol],
            links.loc[dry, ycol],
            s=12,
            marker="x",
            alpha=0.8,
            color="tab:orange",
            label="dry",
        )
    if np.any(wet_ok):
        ax.scatter(
            links.loc[wet_ok, xcol],
            links.loc[wet_ok, ycol],
            s=18,
            marker="o",
            alpha=0.8,
            color="tab:blue",
            label="wet",
        )
    ax.legend(loc="upper right")


def save_field_image(
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
    suffix: str = "png",
):

    fig = plt.figure(figsize=(10, 6))
    ax = plt.gca()

    # values less than min_rain become 0
    z = np.asarray(z, dtype=float)
    z_masked = np.ma.masked_where(z < min_rain, z)

    # make masked values transparent
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(alpha=0.0)

    # pcolormesh so irregular grids plot correctly
    im = ax.pcolormesh(
        xg, yg, z_masked, shading="auto", vmin=vmin, vmax=vmax, cmap=cmap
    )

    plt.colorbar(im, ax=ax, label="mm/h")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if show_links and links is not None and len(links) > 0:
        _scatter_links(ax, links)

    fig.tight_layout()
    fig.savefig(f"{path}.{suffix}", dpi=300, bbox_inches="tight", pad_inches=pad_inches)
    plt.close(fig)


def save_links_image(
    path: str,
    links: pd.DataFrame,
    bbox,
    title: str = "Links (centers)",
    suffix: str = "png",
):
    x_min, x_max, y_min, y_max = bbox
    fig = plt.figure(figsize=(10, 6))
    ax = plt.gca()
    xcol = "lon_center" if "lon_center" in links.columns else "x_center"
    ycol = "lat_center" if "lat_center" in links.columns else "y_center"
    _scatter_links(ax, links)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_title(title)
    ax.set_xlabel("lon" if xcol == "lon_center" else "x")
    ax.set_ylabel("lat" if ycol == "lat_center" else "y")
    fig.tight_layout()
    fig.savefig(f"{path}.{suffix}", dpi=200, bbox_inches="tight", pad_inches=pad_inches)
    plt.close(fig)
