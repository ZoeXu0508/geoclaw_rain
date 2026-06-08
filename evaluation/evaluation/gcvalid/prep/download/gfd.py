"""
Download TC-related flood maps from the Global Flood Database, following the intructions in the
official GitHub repository: https://github.com/cloudtostreet/MODIS_GlobalFloodDatabase
"""
import subprocess

import pandas as pd

import gcvalid.util.constants as u_const


GFD_DATA_DIR = u_const.FLOODMAPS_DIR / "gfd"

GFD_INDEX_DIR = u_const.INDEX_DIR / "gfd"

GFD_GCS_URL = "gs://gfd_v1_4"


def get_file_list():
    path_file_list = GFD_INDEX_DIR / "file_list.txt"
    if not path_file_list.exists():
        p = subprocess.run(["gsutil", "-m", "ls", GFD_GCS_URL], capture_output=True)
        path_file_list.write_text(p.stdout.decode())
    return [l.strip() for l in path_file_list.read_text().split("\n") if l.strip() != ""]


def main():
    dfo_df = pd.read_csv(u_const.INDEX_DIR / "dfo" / "archive_link_ibtracs.csv")
    dfo_df.loc[dfo_df['ibtracs_id'].isna(), 'ibtracs_id'] = ""
    dfo_df['ibtracs_id'] = dfo_df['ibtracs_id'].astype(str)
    dfo_df = dfo_df[~dfo_df['ID'].isna()]
    dfo_df['ID'] = dfo_df['ID'].astype(int)
    file_list = get_file_list()
    linked_ids = []
    for gcs_name in file_list:
        fname = gcs_name.split("/")[-1]
        dfo_id = int(fname.split("_")[1])
        match = dfo_df[dfo_df['ID'] == dfo_id]
        assert match.shape[0] == 1
        match = match.iloc[0, :]
        if match["ibtracs_id"] == "":
            continue
        linked_ids.append((dfo_id, match["ibtracs_id"]))
        out_path = GFD_DATA_DIR / "raw" / fname
        if not out_path.exists():
            print(f"Downloading {gcs_name} ...")
            subprocess.run(["gsutil", "-m", "cp", gcs_name, str(out_path)])

    (
        pd.DataFrame(linked_ids, columns=["dfo_id", "ibtracs_id"])
        .sort_values(by=['dfo_id'])
        .to_csv(GFD_INDEX_DIR / "linked_ids.csv", index=None)
    )


if __name__ == "__main__":
    main()
