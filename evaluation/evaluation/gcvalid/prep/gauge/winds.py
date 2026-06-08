"""
Associate a wind time series to each gauge record, in order to determine the relevant time period
where a storm surge is expected to take place (or to identify gauge stations where no storm
surge is expected to take place at all)
"""
import argparse
import pathlib
import pickle

from climada.hazard import TropCyclone, Centroids, TCTracks
import numpy as np
import pandas as pd

import gcvalid.util.constants as u_const


def all_gauge_files(source):
    """List all gauge files in alphabetical order"""
    all_files = u_const.GAUGES_DIR.glob(f"{source}/records/*.pickle")
    all_files_str = map(str, all_files)
    return map(pathlib.Path, sorted(all_files_str))


def all_station_locations(source):
    stlocations = {}
    for ifile, gfile in enumerate(all_gauge_files(source)):
        with gfile.open("rb") as fp:
            gdata = pickle.load(fp)
        for gsrc, stations in gdata.items():
            for stdata in stations:
                stname = f"{gsrc}:{stdata['filename']}"
                if stname not in stlocations:
                    stlocations[stname] = stdata['location']
                elif stdata['location'] != stlocations[stname]:
                    print(f"Location mismatch for {stname}!")
    return stlocations


def all_tracks(source):
    tracks = TCTracks.from_netcdf(u_const.TRACKS_DIR / source)
    tracks.equal_timestep(time_step_h=1)
    return tracks


def extract_wind_timeseries(track, windfield, i_centroid):
    npositions = windfield.shape[0]
    windfield = windfield.toarray().reshape(npositions, -1, 2)
    windfield = windfield[:, i_centroid, :]
    return pd.DataFrame({
        "u": windfield[:, 0],
        "v": windfield[:, 1],
        "intensity": np.linalg.norm(windfield, axis=-1),
    }, index=track.time)


def all_windfields(stlocations, tracks):
    stnames = sorted(stlocations.keys())
    stlocs = np.array([stlocations[n] for n in stnames])

    cents = Centroids.from_lat_lon(stlocs[:, 0], stlocs[:, 1])
    cents.set_dist_coast(precomputed=True)
    haz = TropCyclone.from_tracks(
        tracks, cents, store_windfields=True,
        max_latitude=90, max_dist_eye_km=1000)

    return haz.event_name, stnames, haz.windfields


def main():
    parser = argparse.ArgumentParser(description='Wind time series for each gauge station.')
    parser.add_argument('source', type=str, metavar="SOURCE", choices=['dfo', 'gfd', 'rapid'],
                        help='The flood map source.')
    source = parser.parse_args().source

    tracks = all_tracks(source)
    trnames, stnames, windfields = all_windfields(
        all_station_locations(source),
        tracks,
    )

    for ifile, gfile in enumerate(all_gauge_files(source)):
        map_id = gfile.stem
        ibtracs_id = map_id.split("-")[0]
        with gfile.open("rb") as fp:
            gdata = pickle.load(fp)
        for gsrc, stations in gdata.items():
            for stdata in stations:
                stname = f"{gsrc}:{stdata['filename']}"
                i_station = stnames.index(stname)
                i_track = trnames.index(ibtracs_id)
                df = extract_wind_timeseries(
                    tracks.data[i_track],
                    windfields[i_track],
                    i_station,
                )

                # discard overly accurate data
                df.loc[df.intensity < 1, ["u", "v", "intensity"]] = 0

                if (df.intensity.values == 0).all():
                    df = None
                stdata['wind'] = df
        print(f"Writing to {gfile} ...")
        with gfile.open("wb") as fp:
            gdata = pickle.dump(gdata, fp)


if __name__ == "__main__":
    main()
