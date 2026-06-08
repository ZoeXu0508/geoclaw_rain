
import pathlib
import re
import shutil
import subprocess
import sys

import numpy as np

STDERR_IGNORE = [
    "*** Parallel Studio XE 2019 without Python support ***",
    "*** Parallel Studio XE 2017 Update 1 available (module load intel/2017.1)  ***",
    "*** (Note new module location!)  ***",
    "*** This sets up compilers, MPI library, VTune (Amplifier), Inspector, Advisor, Debugger ***",
]

STDERR_PYERROR = "Traceback (most recent call last):"

STDERR_GCERROR = "RuntimeError: GeoClaw run failed (see output above)."

STDOUT_GCERROR = "Reading GeoClaw output failed (see output above)."

STDERR_WHOAMI = "whoami: cannot find name for user ID 4080: Connection refused"

STDERR_TIME_LIMIT = ("slurmstepd: error: \*\*\* JOB [0-9]+ ON [a-z0-9-]+ "
                     "CANCELLED AT [0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2} "
                     "DUE TO TIME LIMIT \*\*\*")

STDERR_MANUAL = ("slurmstepd: error: \*\*\* JOB [0-9]+ ON [a-z0-9-]+ "
                 "CANCELLED AT [0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2} "
                 "\*\*\*")

LOG_PATH = pathlib.Path("./log")

JOBS_PATH = pathlib.Path("./jobs")

STDOUT_RUNPATH = "/p/tmp/tovogt/.climada/data/geoclaw/runs/[0-9]{4}-[0-9NS-]+/"


def extract_time_from_logline(line):
    trunc = line[:23].replace(",", ".")
    return np.datetime64(trunc)


def runtime_from_logfile(logfile):
    lines = [l for l in logfile.read_text().split("\n") if l[:2] == "20"]
    first_dt = extract_time_from_logline(lines[0])
    last_dt = extract_time_from_logline(lines[-1])
    return (last_dt - first_dt) / np.timedelta64(1, 's')


def human_readable_timedelta(timedelta_sec):
    if timedelta_sec < 60:
        time_str = f"{timedelta_sec:.0f}sec"
    elif timedelta_sec < 60 * 60:
        time_str = f"{timedelta_sec / 60:.0f}min"
    else:
        time_str = f"{timedelta_sec / 3600:.0f}h{(timedelta_sec / 60) % 60:.0f}min"
    return time_str


def job_failed(log_prefix, job_id):
    path = LOG_PATH / f"{log_prefix}{job_id}.err"
    err = [l for l in path.read_text().split("\n") if l not in STDERR_IGNORE and l != ""]
    filtered_err = []
    i = 0
    while i < len(err):
        l = err[i]
        if "ShapelyDeprecationWarning" in l:
            # the following line is a code snippet that triggered the warning
            i += 2
        elif "UserWarning: rename" in l:
            i += 2
        elif "RuntimeWarning: invalid value encountered in cast" in l:
            i += 2
        else:
            filtered_err.append(l)
            i += 1
    err = filtered_err
    if len(err) > 0:
        return err
    path = LOG_PATH / f"{log_prefix}{job_id}.out"
    out = [l for l in path.read_text().split("\n") if STDOUT_GCERROR in l]
    if len(out) > 0:
        return out
    return False


def jobid_from_logname(log_prefix, path):
    return path.stem[len(log_prefix):]


def jobarrayid_from_logname(log_prefix, path):
    return jobid_from_logname(log_prefix, path).split("_")[0]


def get_jobfile_from_args(args):
    for job_file in JOBS_PATH.glob("*.txt"):
        jobs = job_file.read_text().split("\n")
        if all(a in jobs for a in args):
            return str(job_file)
    return None


def get_jobfile_from_arrayid(log_prefix, array_id):
    stem_glob = f"{log_prefix}{array_id}_*"
    job_file = None
    for path in LOG_PATH.glob(f"{stem_glob}.jobfile"):
        job_file = path.read_text().strip()
        break
    if job_file is not None:
        return job_file
    args_files = list(LOG_PATH.glob(f"{stem_glob}.args"))
    job_file = get_jobfile_from_args([path.read_text().strip() for path in args_files])
    if job_file is not None:
        for args_f in args_files:
            path = args_f.parent / f"{args_f.stem}.jobfile"
            path.write_text(job_file)
    return job_file


def jobarrays_by_files(log_prefix):
    array_ids = sorted({
        jobarrayid_from_logname(log_prefix, path)
        for path in LOG_PATH.glob(f"{log_prefix}*.args")
    })
    jobfiles = {}
    for array_id in array_ids:
        job_file = get_jobfile_from_arrayid(log_prefix, array_id)
        if job_file is None:
            print(f"No job file for job array {array_id}!")
            continue
        if job_file not in jobfiles:
            jobfiles[job_file] = []
        jobfiles[job_file].append(array_id)
    return jobfiles


def is_timeout_error(errors):
    return any(re.match(STDERR_TIME_LIMIT, l) for l in errors)


def is_manual_cancel(errors):
    return any(re.match(STDERR_MANUAL, l) for l in errors)


def is_whoami_error(errors):
    return any(re.match(STDERR_WHOAMI, l) for l in errors)


def is_gc_ld_error(gc_error):
    if gc_error is None:
        return False
    return "collect2: error: ld returned 1 exit status" in gc_error


def is_gc_freelist_error(gc_error):
    if gc_error is None:
        return False
    for line in gc_error[::-1]:
        if line.startswith("  free list full with"):
            return True
    return False


def is_gc_segfault_error(gc_error):
    if gc_error is None:
        return False
    for line in gc_error[::-1]:
        lline = line.lower()
        if ("segmentation fault" in lline
            or "program aborted. backtrace:" in lline
            or lline.startswith("forrtl: severe")):
                return True
    return False


def is_gc_permission_error(gc_error):
    if gc_error is None:
        return False
    for line in gc_error[::-1]:
        if "PermissionError: [Errno 13] Permission denied:" in line and "xgeoclaw" in line:
            return True
    return False


def is_gc_solution_error(gc_error):
    if gc_error is None:
        return False
    return " **** Too many dt reductions ****" in gc_error


def is_gc_dt_error(gc_error):
    if gc_error is None:
        return False
    for line in gc_error[::-1]:
        if " SOLUTION ERROR --- ABORTING CALCULATION" in line:
            return True
    return False


def get_gc_error(log_prefix, job_id):
    output = (LOG_PATH / f"{log_prefix}{job_id}.out").read_text()
    gc_error = None
    for line in output.split("\n"):
        if "Output of 'make .output' in GeoClaw work directory:" in line:
            gc_error = []
            continue
        if gc_error is not None:
            gc_error.append(line)
    return gc_error


def print_geoclaw_stdout(log_prefix, job_id):
    gc_error = get_gc_error(log_prefix, job_id)
    if gc_error is None:
        return
    if is_gc_permission_error(gc_error):
        print("Permission error when trying to execute `xgeoclaw` binary")
    elif is_gc_solution_error(gc_error):
        print("Solution error during calculation")
    elif is_gc_dt_error(gc_error):
        print("Too many dt reductions")
    elif is_gc_freelist_error(gc_error):
        print("Free list error during calculation")
    elif is_gc_segfault_error(gc_error):
        print("Segmentation fault")
    else:
        print("Unknown error type")
        for e in gc_error:
            if "floating-point exceptions" in e:
                print(e)
        if any("not available from grid" in e for e in gc_error):
            print("Time wanted ... not available from grid ...")
        if any("*** WARNING *** Courant number" in e for e in gc_error):
            print("Courant number ... is larger than input cfl_max ...")
    logfile = LOG_PATH / f"{log_prefix}{job_id}.out"
    print(f"Runtime: {human_readable_timedelta(runtime_from_logfile(logfile))}")


def rm_runfiles(log_prefix, job_id, do_exec=False):
    output = (LOG_PATH / f"{log_prefix}{job_id}.out").read_text()
    match = re.search(STDOUT_RUNPATH, output)
    if match is None:
        print(f"No rundata found for job {job_id}!")
        return
    runpath = pathlib.Path(match.group(0))
    print(f"rm -rf {runpath}")
    if do_exec:
        shutil.rmtree(runpath)


def rm_logfiles(log_prefix, job_id, do_exec=False):
    glob_name = f"{log_prefix}{job_id}.*"
    print(f"rm {LOG_PATH / glob_name}")
    for path in LOG_PATH.glob(glob_name):
        if do_exec:
            path.unlink()


def restart_jobs(job_file, job_ids, medium=False, do_exec=False):
    if len(job_ids) == 0:
        return
    restart_pos = sorted(set([s.split("_")[-1] for s in job_ids]))
    bash_cmd = (
        f"for jobs in {job_file}; do "
        f"sbatch{' --qos=medium' if medium else ''} --array={','.join(restart_pos)} "
        "slurm/tc_run_geoclaw.sh $jobs; done"
    )
    print(bash_cmd)
    if do_exec:
        subprocess.call(bash_cmd, shell=True)


def main(log_prefix, do_exec, rm, ignore):
    print(f"Check jobs with prefix: {log_prefix}")
    print(f"Ignoring: {', '.join(ignore)}")
    if do_exec:
        print("Running in execution mode!")
    else:
        print("Running in dry mode, use --exec to execute the commands.")

    failed_jobs = {}
    for path in LOG_PATH.glob(f"{log_prefix}*.err"):
        job_id = jobid_from_logname(log_prefix, path)
        if job_id in ignore:
            continue
        errors = job_failed(log_prefix, job_id)
        if errors:
            failed_jobs[job_id] = errors

    jobfiles = jobarrays_by_files(log_prefix)
    print("Jobarrays:")
    for f, arrs in jobfiles.items():
        print(f"{f[5:]} ({', '.join(arrs)})")

    rm_jobids = []
    for job_id in rm:
        # translate array ids into job ids
        if any(job_id in arrs for arrs in jobfiles.values()):
            for path in LOG_PATH.glob(f"{log_prefix}{job_id}_*.out"):
                rm_jobids.append(jobid_from_logname(log_prefix, path))
        else:
            rm_jobids.append(job_id)
    for job_id in rm_jobids:
        rm_runfiles(log_prefix, job_id, do_exec=do_exec)
        rm_logfiles(log_prefix, job_id, do_exec=do_exec)
    if len(rm) > 0:
        return

    if len(failed_jobs.keys()) == 0:
        print("No failed jobs!")
        return

    restart_ids_def = {jobfile: [] for jobfile in jobfiles.keys()}
    restart_ids_medium = {jobfile: [] for jobfile in jobfiles.keys()}
    for jobfile, array_ids in jobfiles.items():
        del_failed_jobs = []
        for job_id, errors in failed_jobs.items():
            job_arrid, job_arrpos = job_id.split("_")
            if job_arrid not in array_ids:
                continue
            if is_timeout_error(errors):
                # jobs with timeout: restart on 'medium' qos
                restart_ids_medium[jobfile].append(job_id)
            elif is_whoami_error(errors):
                # jobs with whoami error: restart on 'medium' qos
                # this is typically for the Idai runs (don't know why though...)
                restart_ids_medium[jobfile].append(job_id)
            else:
                gc_error = get_gc_error(log_prefix, job_id)
                if gc_error is None:
                    continue
                if is_gc_ld_error(gc_error):
                    restart_ids_medium[jobfile].append(job_id)
                else:
                    continue
            del_failed_jobs.append(job_id)
        for job_id in del_failed_jobs:
            del failed_jobs[job_id]

    for jobfile in jobfiles.keys():
        all_jobids = restart_ids_def[jobfile] + restart_ids_medium[jobfile]
        if len(all_jobids) == 0:
            continue
        restart_jobs(jobfile, restart_ids_def[jobfile], medium=False, do_exec=do_exec)
        restart_jobs(jobfile, restart_ids_medium[jobfile], medium=True, do_exec=do_exec)
        for job_id in all_jobids:
            rm_runfiles(log_prefix, job_id, do_exec=do_exec)
            rm_logfiles(log_prefix, job_id, do_exec=do_exec)

    # print info about remaining failed jobs
    for job_id, errors in failed_jobs.items():
        job_arrid, job_arrpos = job_id.split("_")
        job_file = [pathlib.Path(f).name for f, a in jobfiles.items() if job_arrid in a][0]
        print("")
        if STDERR_GCERROR in errors or STDOUT_GCERROR in errors[0]:
            print(f"{job_id} ({job_file}) failed due to GeoClaw error:")
            print_geoclaw_stdout(log_prefix, job_id)
        elif STDERR_PYERROR in errors:
            print(f"{job_id} ({job_file}) failed due to Python error:")
            for e in errors:
                print(e)
        elif is_manual_cancel(errors):
            print(f"{job_id} ({job_file}) has been cancelled manually.")
            rm_runfiles(log_prefix, job_id, do_exec=do_exec)
            rm_logfiles(log_prefix, job_id, do_exec=do_exec)
        else:
            print(f"{job_id} ({job_file}) failed due to unknown error:")
            for e in errors:
                print(e)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Check log dir for failed jobs (and restart)')
    parser.add_argument('--exec', action='store_true', default=False,
                        help='Really execute the restart and cleanup commands.')
    parser.add_argument('--prefix', type=str, default="tc_run_geoclaw-",
                        help='Prefix of log files to check.')
    parser.add_argument('--rm', type=str, nargs="+", default=[], metavar="ID",
                        help='Remove log and run files for the specified job or array IDs.')
    parser.add_argument('--ignore', type=str, nargs="+", default=[], metavar="ID",
                        help='Ignore these job ids in the whole procedure.')
    args = parser.parse_args()

    main(args.prefix, args.exec, args.rm, args.ignore)
