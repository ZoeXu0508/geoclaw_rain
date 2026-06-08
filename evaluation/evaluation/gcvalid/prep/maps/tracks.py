"""
Generate NetCDF track files for flood maps
"""
import argparse

from climada.hazard import TCTracks
import pandas as pd

import gcvalid.util.constants as u_const


IBTRACS_NAMES_FILE = u_const.TRACKS_DIR / "raw" / "names.csv"
IBTRACS_NAMES = pd.read_csv(IBTRACS_NAMES_FILE, index_col="ibtracs_id")


def get_ibtracs_name(ibtracs_id):
    if ibtracs_id in IBTRACS_NAMES.index:
        category, name = IBTRACS_NAMES.loc[ibtracs_id, ["category", "name"]]
    else:
        html = requests.get(f"http://ibtracs.unca.edu/index.php?name=v04r00-{ibtracs_id}").text
        time.sleep(0.5)
        m = re.search(r"</a>\n *([^<]*) ([A-Z:-]+) *\([^[\)]*\)</h1>", html, flags=re.DOTALL)
        if m is None:
            print(f'Unofficial IBTrACS ID: {ibtracs_id}')
            return "", ""
        category = m.group(1)
        name = m.group(2)
        name = ":".join(n[0].upper() + n[1:].lower() for n in name.split(":"))
        name = "-".join(n[0].upper() + n[1:].lower() for n in name.split("-"))
        IBTRACS_NAMES.loc[ibtracs_id, ["category", "name"]] = [category, name]
        IBTRACS_NAMES.to_csv(IBTRACS_NAMES_FILE)
    return category, name


def main():
    parser = argparse.ArgumentParser(description='Generate NetCDF track files for flood maps.')
    parser.add_argument('source', type=str, metavar="SOURCE", choices=['dfo', 'gfd', 'rapid'],
                        help='The flood map source.')
    source = parser.parse_args().source

    meta = pd.read_hdf(u_const.FLOODMAPS_DIR / source / "meta.hdf5")

    tracks = TCTracks.from_ibtracs_netcdf(
        storm_id=meta['ibtracs_id'].to_list(), estimate_missing=True)

    for t in tracks.data:
        category, name = get_ibtracs_name(t.sid)
        t.attrs['ibtracs_category'] = category
        t.attrs['ibtracs_name'] = name

    out_path = u_const.TRACKS_DIR / source
    print(f"Writing to {out_path}...")
    tracks.write_netcdf(out_path)


if __name__ == "__main__":
    main()
