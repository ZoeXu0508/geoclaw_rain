
import pathlib
import re
import shutil

LOG_DIR = pathlib.Path("./log")

OUTPUT_DIR = pathlib.Path("./output")

STDOUT_RUNPATH = "/p/tmp/tovogt/.climada/data/geoclaw/runs/[0-9]{4}-[0-9NS-]+/"

NO_RUN_REASONS = [
    "No centroids within reach of this storm track.",
    "This storm doesn't affect any coastal areas.",
]

def copy_meta(run_dir, target_dir):
    job_name = target_dir.parent.name
    for rundata in ["claw", "geoclaw", "regions"]:
        for f_rundata in run_dir.glob(f"*/{rundata}.data"):
            f_target = target_dir / f"{job_name}-{f_rundata.parent.name}-{rundata}.data"
            shutil.copy(f_rundata, f_target)


def store_run_dir(run_dirs, run_dir, key):
    if run_dir not in run_dirs:
        run_dirs[run_dir] = []
    run_dirs[run_dir].append(key)


def extract_meta(job_id, log_output, target_dir, run_dirs):
    match = re.search(STDOUT_RUNPATH, log_output)
    if match is None:
        print(f"No rundata found for job {job_id}!")
        return
    run_dir = pathlib.Path(match.group(0))
    if not run_dir.exists():
        print("Path to run directory does not exist", job_id, run_dir)

    for p in target_dir.iterdir():
        p.unlink()

    copy_meta(run_dir, target_dir)
    store_run_dir(run_dirs, run_dir, job_id)


def main():
    # keep track of which run directories we encounter
    run_dirs = {}

    # extract from log files
    for log_file in LOG_DIR.glob("*.out"):
        job_id = log_file.stem.split("-")[-1]
        log_output = log_file.read_text()
        match = re.search(r"- INFO - Writing (output/.+)\.hdf5", log_output)
        if match is None:
            print(f"No output for {job_id}, the job is probably still running...")
            continue
        target_dir = pathlib.Path(match.group(1)).parent / "meta"
        target_dir.mkdir(exist_ok=True)
        if any(r in log_output for r in NO_RUN_REASONS):
            continue
        extract_meta(job_id, log_output, target_dir, run_dirs)

    # extract from resume files
    for resume_file in OUTPUT_DIR.glob("*/*-resume.txt"):
        target_dir = resume_file.parent / "meta"
        if target_dir.exists():
            continue
        target_dir.mkdir(exist_ok=True)
        run_dir = pathlib.Path(resume_file.read_text().strip())
        if run_dir.exists():
            copy_meta(run_dir, target_dir)
            store_run_dir(run_dirs, run_dir, target_dir.parent.name)

    # print suspicious cases
    for run_dir, ids in run_dirs.items():
        if len(ids) > 1:
            print("Duplicate run directory", run_dir, ",".join(ids))


if __name__ == "__main__":
    main()
