"""
Set up GeoClaw job parameters for all flood maps
"""
import argparse
import pickle

import pandas as pd

import gcvalid.util.constants as u_const


def get_jobs_by_ibtracs_id(source, meta):
    gauges_by_ibtracs_id = {}
    for gfile in (u_const.GAUGES_DIR / source / "records").glob("*.pickle"):
        map_id = gfile.stem
        ibtracs_id = map_id[:-2]
        bounds = [m[1] for m in meta if m[0] == map_id][0]
        if ibtracs_id not in gauges_by_ibtracs_id:
            gauges_by_ibtracs_id[ibtracs_id] = [[], []]

        padded_bounds = (bounds[0] - 0.1, bounds[1] - 0.1,
                         bounds[2] + 0.1, bounds[3] + 0.1)
        gauges_by_ibtracs_id[ibtracs_id][0].append(padded_bounds)

        with gfile.open("rb") as fp:
            gdata = pickle.load(fp)
        for gsrc, stations in gdata.items():
            gauges_by_ibtracs_id[ibtracs_id][1].extend([
                stdata['gc_location'] for stdata in stations
                if stdata['discarded'] == False or gsrc == "codec"
            ])

    result = {}
    for ibtracs_id, (bounds, latlon) in gauges_by_ibtracs_id.items():
        s = (
            f"{ibtracs_id} --bounds " +
            " ".join("B" + ",".join(f"{c:.5f}" for c in b)
                     for b in bounds)
        )
        if len(latlon) > 0:
            # truncate coordinates to 5 digits precision, and remove duplicates
            latlon_s = [tuple(round(c, 5) for c in l) for l in latlon]
            latlon_u = sorted(set(latlon_s))
            s += (
                " --gauges " +
                " ".join("P" + ",".join(f"{c:.5f}" for c in l)
                         for l in latlon_u)
            )
        result[ibtracs_id] = s
    return result


def main():
    parser = argparse.ArgumentParser(description='Set up GeoClaw job parameters.')
    parser.add_argument('source', type=str, metavar="SOURCE", choices=['dfo', 'gfd', 'rapid'],
                        help='The flood map source.')
    source = parser.parse_args().source

    meta = pd.read_hdf(u_const.FLOODMAPS_DIR / source / "meta.hdf5").sort_values(by="date")
    meta = [[row['map_id'], [row['xmin'], row['ymin'], row['xmax'], row['ymax']]]
            for idx, row in meta.iterrows()]

    jobs_by_ibtracs_id = get_jobs_by_ibtracs_id(source, meta)

    for zos in ["0", "aviso", "mercator"]:
        for tides in ["no", "min", "mean", "max"]:
            path = u_const.GEOCLAW_DIR / source / "jobs" / f"{source}-zos_{zos}-fes_{tides}.txt"
            print(f"Writing to {path} ...")
            with path.open("w") as fp:
                fp.writelines([
                    f"{jobs_by_ibtracs_id[key]} --tides {tides} --zos {zos} --suffix _{source}\n"
                    for key in sorted(jobs_by_ibtracs_id.keys())
                ])


if __name__ == "__main__":
    main()
