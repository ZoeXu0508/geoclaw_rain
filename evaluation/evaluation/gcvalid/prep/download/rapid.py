"""
Download flood maps from the RAPID repository
"""
import multiprocessing
import pathlib
import re
import sys

import numpy as np
import pandas as pd
import requests

import gcvalid.util.constants as u_const


RAPID_DATA_DIR = u_const.FLOODMAPS_DIR / "rapid" / "raw"

RAPID_INDEX_DIR = u_const.INDEX_DIR / "rapid"

RAPID_LIST_URL = "https://rapid-nrt-flood-maps.s3.us-west-2.amazonaws.com/"

RAPID_BASE_URL = "https://rapid-nrt-flood-maps.s3.amazonaws.com/"


def link_dfo_tcs():
    dfo_df = pd.read_csv(u_const.INDEX_DIR / "dfo" / "archive_link_ibtracs.csv")
    dfo_df = dfo_df[~dfo_df['ibtracs_id'].isna()]
    dfo_df['ID'] = dfo_df['ID'].astype(int)
    dfo_df['MainCause'] = dfo_df['MainCause'].str.lower()

    rapid_df = pd.read_excel(RAPID_INDEX_DIR / "List_EventsLinktoDFOandFloodMaps.xlsx")
    rapid_df = rapid_df.rename(columns={rapid_df.columns[1]: "dfo_id"})
    rapid_df = rapid_df[~rapid_df["dfo_id"].isna()]
    rapid_df = rapid_df[~rapid_df["List of Flood Maps"].isna()]
    rapid_df["dfo_id"] = rapid_df["dfo_id"].astype(int)

    mask = (dfo_df['ID'] >= rapid_df["dfo_id"].min())
    rapid_df = rapid_df[np.isin(rapid_df["dfo_id"], dfo_df[mask]['ID'].values)]
    match_idx = [(dfo_df['ID'].values == i).nonzero()[0][0] for i in rapid_df["dfo_id"].values]

    out_df = rapid_df[['EventID', 'dfo_id']].rename(columns={"EventID": "rapid_id"})
    out_df['ibtracs_id'] = dfo_df['ibtracs_id'].values[match_idx]
    out_df.to_csv(RAPID_INDEX_DIR / "linked_ids.csv", index=False)

    for idx, row in rapid_df.iterrows():
        path = RAPID_DATA_DIR / row['EventID']
        path.mkdir(parents=True, exist_ok=True)
        path = path / "List_FloodMaps.txt"
        if path.exists():
            continue
        print(f"Fetching URLs for RAPID ID {row['EventID']}")
        url = RAPID_BASE_URL + row['List of Flood Maps']
        url = url.replace("Archive_Flood_EventsFloodEvents", "Archive_Flood_Events/FloodEvents")
        path.write_text(requests.get(url).text)


def download_file(bucket_path, outpath):
    r = requests.get(RAPID_BASE_URL + str(bucket_path), stream=True)
    total_length = int(r.headers.get('content-length'))
    print_fname = outpath.name[:11] + "..." + outpath.name[-25:]
    with outpath.open('wb') as fp:
        chunk_size = 4096
        for ichunk, chunk in enumerate(r.iter_content(chunk_size=chunk_size)):
            fp.write(chunk)
            perc = int(100 * ichunk * chunk_size / total_length)
            if ichunk % 40 == 0:
                sys.stdout.write(f"\rDownloading {print_fname} ... {perc: 3d}%")
                sys.stdout.flush()
        sys.stdout.write(f"\rDownloading {print_fname} ... 100%\n")


def download_maps():
    paths = []
    for fname in RAPID_DATA_DIR.glob("*/List_FloodMaps.txt"):
        for bucket_path in fname.read_text().split("\n"):
            if bucket_path[:5] != "RAPID":
                continue
            out_dir = fname.parent / (
                pathlib.Path(bucket_path).relative_to("RAPID_Archive_Flood_Maps")
            )
            out_dir.mkdir(parents=True, exist_ok=True)

            path = out_dir / "ListBucketResult.xml"
            if not path.exists():
                path.write_text(requests.get(RAPID_LIST_URL, params={
                    "list-type": "2",
                    "delimiter": "/",
                    "prefix": bucket_path,
                }).text)
            list_bucket_result = path.read_text()

            for m in re.finditer(r">([^<]+\.tif)", list_bucket_result):
                bucket_path = pathlib.Path(m.group(1))
                outpath = out_dir / bucket_path.name
                if not outpath.exists():
                    paths.append((bucket_path, outpath))

    if len(paths) == 0:
        print("All of the RAPID maps have been downloaded already.")
        return

    print(f"Downloading {len(paths)} files...")
    pool = multiprocessing.Pool(5)
    pool.starmap(download_file, paths)


def main():
    link_dfo_tcs()
    download_maps()

if __name__ == "__main__":
    main()
