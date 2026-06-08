"""
Get elevation on global 30 arc-second grid from mixed DEM data set
"""
import numpy as np

from gcvalid.prep.maps.elevation import RES_GRID_DEG, grid_from_bounds
import gcvalid.util.constants as u_const
import gcvalid.util.io as u_io


GLOBAL_BOUNDS = (-180, -90, 180, 90)


def main():
    out_path = u_const.ELEVATION_DIR / "global.tif"
    if out_path.exists():
        return

    transform, shape = grid_from_bounds(GLOBAL_BOUNDS)
    data = u_io.read_raster_reproject(
        u_const.DEM_FILE,
        resampling="average",
        transform=transform,
        shape=shape,
        dtype=np.float64,
    )

    print(f"Writing to {out_path}...")
    u_io.write_raster(out_path, data, transform=transform, shape=data.shape, nodata=np.nan)


if __name__ == "__main__":
    main()
