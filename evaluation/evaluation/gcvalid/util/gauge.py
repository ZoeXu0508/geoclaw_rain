
import pickle
import re

import climada.util.coordinates as u_coord
import numba
import numpy as np
import pandas as pd
import pyfes

import gcvalid.util.constants as u_const


HARMONICS_START_DATE = np.datetime64("1899-12-30T00:00")
"""The start date for the harmonics analysis of sea level used by JRC (e.g. for the WSL data)"""


HARMONICS_FREQ_UNIT = np.timedelta64(1, "D")
"""The temporal unit of the frequencies given in HARMONICS_FREQ_D (below)."""


HARMONICS_FREQ_D = np.array([
    0.128766334283776,
    0.128857194376164,
    0.129381266016223,
    0.147068369304709,
    0.168413975812495,
    0.168569435980517,
    0.170357083547568,
    0.170516143636361,
    0.172508349331654,
    0.174695758496972,
    0.199890544415426,
    0.205453337753506,
    0.249658233357344,
    0.249999992,
    0.253952177489379,
    0.254305803118864,
    0.25631444055176,
    0.256674704014076,
    0.258762532032447,
    0.261215591287424,
    0.333029389398687,
    0.340714167095135,
    0.341351021629882,
    0.34501669723487,
    0.349429284557433,
    0.489771751709733,
    0.491088802767104,
    0.498634767923513,
    0.499316511591742,
    0.499999984,
    0.507984165946608,
    0.509240592196617,
    0.51606260547855,
    0.516792827371988,
    0.517525060850908,
    0.518259333570738,
    0.526083541864433,
    0.527431160919067,
    0.536323226339296,
    0.537723941372983,
    0.546969491030712,
    0.548426385474333,
    0.899093169594225,
    0.92941980084125,
    0.934174122926896,
    0.962436511046921,
    0.966956557148237,
    0.991853148349387,
    0.994554139787846,
    0.997269547781583,
    0.999999872000017,
    1.00274540461034,
    1.00550583624144,
    1.02954467892393,
    1.03471863450781,
    1.04061468326798,
    1.06950552105274,
    1.07580588726596,
    1.11346057230323,
    1.11951481625018,
    1.16034950581132,
    1.16692589225208,
    1.21136109404707,
    13.6607901226149,
    14.7652926794033,
    27.5545491899403,
    31.8119339543532,
    182.621183765123,
    365.259977441544,
])
"""Periods (in days) of the 69 harmonics components used by JRC (e.g. for the WSL data)

Most of the components are listed in Table 1 of the following publication:

    Annunziato, A., Probst, P. (2016): Continuous Harmonics Analysis of Sea Level Measurements.
    JRC Technical Reports, EUR 28308 EN. Ispra (Italy): Publications Office of the European Union.
    https://data.europa.eu/doi/10.2788/4295
"""


def harmonics_fun(harmonics, start_date=None, freq_unit=None):
    """For the specified harmonics coefficients, set up a function that predicts tides

    Make sure that you use consistent temporal offset and scaling when fitting the harmonics and
    when predicting tides.

    Note that this function is agnostic of the vertical datum and units. The predicted tides will
    be relative to the same vertical datum and in the same units that were used when fitting the
    harmonics.

    Parameters
    ----------
    harmonics : ndarray of floats, shape (nharmonics, 3)
        Harmonics constituents. Each row is a tuple consisting of a frequency, a cosine coefficient
        and a sine coefficient. The first row (indexed 0) is expected to contain the constant
        constituent as the cosine coefficient.
    start_date : datetime64, optional
        The temporal offset used in the harmonics fitting. Default: HARMONICS_START_DATE
    freq_unit : timedelta64, optional
        The temporal unit of the frequencies used in the harmonics fitting.
        Default: HARMONICS_FREQ_UNIT

    Returns
    -------
    tide_fun : function
        A function that computes tides for a given datetime64-array of times.
    """
    if start_date is None:
        start_date = HARMONICS_START_DATE

    if freq_unit is None:
        freq_unit = HARMONICS_FREQ_UNIT

    harmonics0 = harmonics[0, 1]
    harmonics = harmonics[1:, :].copy()
    harmonics[:, 0] = 2 * np.pi / harmonics[:, 0]
    def tmp_fun(times, h0=harmonics0, h=harmonics, t0=start_date, t_unit=freq_unit):
        args = h[:, 0:1] * ((times - t0) / t_unit)[None]
        return h0 + (h[:, 1:2] * np.cos(args) + h[:, 2:3] * np.sin(args)).sum(axis=0)
    return tmp_fun


def fit_harmonics(data, start_date=None, frequencies=None, freq_unit=None):
    """Fit harmonics for tide gauge series `data`

    Note that the vertical datum and units will be taken as is. Tides that are predicted based on
    the fitted harmonics constituents will have the same vertical datum and units.

    Parameters
    ----------
    data : Series of floats with datetime index
        The tide time series for which to estimate harmonics constituents.
    start_date : datetime64, optional
        The temporal offset.
    frequencies : ndarray of floats, optional
        The frequencies (in `freq_unit`) of the harmonics components to which the data is fitted.
        Default: HARMONICS_FREQ_D
    freq_unit : timedelta64, optional
        The temporal unit of the frequencies. Default: HARMONICS_FREQ_UNIT

    Returns
    -------
    harmonics : ndarray of floats, shape (nharmonics, 3)
        Harmonics constituents. Each row is a tuple consisting of a frequency, a cosine coefficient
        and a sine coefficient. The first row (indexed 0) contains the constant constituent as the
        cosine coefficient.
    """
    if freq_unit is None:
        freq_unit = HARMONICS_FREQ_UNIT

    if frequencies is None:
        frequencies = HARMONICS_FREQ_D

    data = data.dropna()
    nharmonics = frequencies.size
    times = (data.index.values - start_date) / freq_unit
    heights = data.values

    Cmat, rhs = _fit_harmonics_lse(times, frequencies, heights)
    result = np.linalg.solve(Cmat, rhs)

    harmonics = np.zeros((nharmonics + 1, 3))
    harmonics[1:, 0] = frequencies
    harmonics[0, 1]= result[0]
    harmonics[1:, 1]= result[1:nharmonics + 1]
    harmonics[1:, 2]= result[nharmonics + 1:]
    return harmonics


@numba.njit
def _fit_harmonics_lse(times, frequencies, heights):
    """Set up a linear system of equations for harmonics fit.

    Works with huge numbers of measurements `heights` (even in cases where numpy-only operations
    would run out of memory).

    For details, see Section 3.1 in the following publication:

        Annunziato, A., Probst, P. (2016): Continuous Harmonics Analysis of Sea Level Measurements.
        JRC Technical Reports, EUR 28308 EN. Ispra (Italy): Publications Office of the European
        Union. https://data.europa.eu/doi/10.2788/4295

    Parameters
    ----------
    times : ndarray of floats, shape (ntimes,)
        The times at which the heights are given.
    frequencies : ndarray of floats, shape (nfrequencies,)
        The frequencies of the harmonics components to consider.
    heights : ndarray of floats, shape (ntimes,)
        The water levels at the specified times.

    Returns
    -------
    C : ndarray of floats, shape (2 * nfrequencies + 1, 2 * nfrequencies + 1)
        There is a cosine and a sine component for each frequency plus a constant component.
    rhs : ndarray of floats, shape (2 * nfrequencies + 1,)
        There is a cosine and a sine component for each frequency plus a constant component.
    """
    N = times.size
    n = frequencies.size
    m = 2 * n + 1

    C = np.zeros((m, m))
    rhs = np.zeros((m,))

    C[0, 0] = N
    rhs[0] = heights.sum()
    for i in range(n):
        arg_i = 2 * np.pi  * times[:] / frequencies[i]
        cos_i = np.cos(arg_i)
        sin_i = np.sin(arg_i)

        # right-hand side of equation
        rhs[1 + i] = (heights * cos_i).sum()
        rhs[1 + n + i] = (heights * sin_i).sum()

        C[0, 1 + i] = cos_i.sum()
        C[0, 1 + n + i] = sin_i.sum()

        # symmetric matrix entries:
        C[1 + i, 0] = C[0, 1 + i]
        C[1 + n + i, 0] = C[0, 1 + n + i]

        for j in range(n):
            arg_j = 2 * np.pi * times[:] / frequencies[j]
            cos_j = np.cos(arg_j)
            sin_j = np.sin(arg_j)

            C[1 + i, 1 + n + j] = (cos_i * sin_j).sum()

            # symmetric matrix entries:
            C[1 + n + j, 1 + i] = C[1 + i, 1 + n + j]

            if j >= i:
                C[1 + i, 1 + j] = (cos_i * cos_j).sum()
                C[1 + n + i, 1 + n + j] = (sin_i * sin_j).sum()

                # symmetric matrix entries:
                C[1 + j, 1 + i] = C[1 + i, 1 + j]
                C[1 + n + j, 1 + n + i] = C[1 + n + i, 1 + n + j]

    return C, rhs


class FESReader:
    """Convenience wrapper around pyfes handlers and their calculation routines."""

    def __init__(self):
        """New instance of FESReader

        Sets up pyfes handlers for ocean and load (radial) tides with data from FES2014.
        """
        fes_path = u_const.GAUGES_DIR / "fes2014"
        ocean_tide_path = fes_path / "ocean_tide_extrapolated.ini"
        load_tide_path = fes_path / "load_tide.ini"
        self.short_tide = pyfes.Handler("ocean", "io", str(ocean_tide_path))
        self.radial_tide = pyfes.Handler("radial", "io", str(load_tide_path))


    def calculate(self, lons, lats, dates):
        """Calculate astronomical tides at given locations and dates

        The three input parameters are broadcasted to the same shape. For example, if lons and lats
        have shape (1, npoints) and dates has shape (ntimes, 1), then tides will be calculated for
        every combination of location and date, so that the result will have the
        shape (ntimes, npoints). If all three parameters are arrays of the same shape, then the
        result will have the same shape as the input.

        Parameters
        ----------
        lons : ndarray of various shapes
            Longitudinal coordinates of locations at which to calculate tides.
        lats : ndarray of various shapes
            Latitudinal coordinates of locations at which to calculate tides.
        dates : ndarray of various shapes and dtype datetime64[us]
            Dates at which to calculate tides.

        Returns
        -------
        tides_mm : ndarray of broadcast shape
            The astronomical tides (in mm) according to FES2014. Note that the vertical datum is
            not specified in the FES source code. It is most likely to be mean sea level, but if
            you don't want to rely on it, you should explicitly subtract annual means. The shape of
            the returned array corresponds to the broadcasted shape of all three input parameters.
        """
        lons, lats, dates = np.broadcast_arrays(lons, lats, dates)
        tide, lp, _ = self.short_tide.calculate(lons.ravel(), lats.ravel(), dates.ravel())
        load, _, _ = self.radial_tide.calculate(lons.ravel(), lats.ravel(), dates.ravel())
        # sum up and convert cm to mm
        tides_mm = (tide + lp + load) * 10
        return tides_mm.reshape(lons.shape)


    def calculate_period(self, lons, lats, period, t_res_h=1):
        """Calculate astronomical tides at given locations within the given period

        Parameters
        ----------
        lons : ndarray of shape (npoints,)
            Longitudinal coordinates of locations at which to calculate tides.
        lats : ndarray of shape (npoints,)
            Latitudinal coordinates of locations at which to calculate tides.
        period : pair of datetime64
            Start and end date defining the period for which to calculate tides. The start and end
            date will be included.
        t_res_h : float, optional
            Temporal resolution (in hours) at which to calculate tides. Default: 1

        Returns
        -------
        tides_mm : ndarray of shape (ntimes, npoints)
            The astronomical tides (in mm) according to FES2014. Note that the vertical datum is
            not specified in the FES source code. It is most likely to be mean sea level, but if
            you don't want to rely on it, you should explicitly subtract annual means.
        """
        t_res = np.timedelta64(t_res_h, 'h')
        dates = np.arange(period[0], period[1] + t_res, t_res).astype('datetime64[us]')
        return self.calculate(lons[None, :], lats[None, :], dates[:, None])


def compute_fes_tides(lons, lats, dates, reader=None, ref_annual_msl=False):
    """Compute astronomical tides according to the FES2014 model

    The three input parameters are broadcasted to the same shape. For example, if lons and lats
    have shape (1, npoints) and dates has shape (ntimes, 1), then tides will be calculated for
    every combination of location and date, so that the result will have the shape (ntimes,
    npoints). If all three parameters are arrays of the same shape, then the result will have the
    same shape as the input.

    Parameters
    ----------
    lons : ndarray of various shapes
        Longitudinal coordinates of locations at which to calculate tides.
    lats : ndarray of various shapes
        Latitudinal coordinates of locations at which to calculate tides.
    dates : ndarray of various shapes and dtype datetime64[us]
        Dates at which to calculate tides.
    reader : FESReader, optional
        If not given, a new instance of FESReader is created.
    ref_annual_msl : boolean, optional
        If given, reference the results to annual mean sea level.

    Returns
    -------
    tides_mm : ndarray of broadcast shape
        The astronomical tides (in mm) according to FES2014. Note that the vertical datum is not
        specified in the FES source code. It is most likely to be mean sea level, but if you don't
        want to rely on it, you should explicitly subtract annual means. The shape of the returned
        array corresponds to the broadcasted shape of all three input parameters.
    """
    if reader is None:
        reader = FESReader()
    tides_mm = reader.calculate(lons, lats, dates)
    if not ref_annual_msl:
        return tides_mm

    lons, lats, dates = np.broadcast_arrays(lons, lats, dates)
    years = dates.astype('datetime64[Y]').astype(int) + 1970
    y_uniq, y_inverse = np.unique(years, return_inverse=True)
    annual_msl = np.array([
        reader.calculate_period(lons, lats, (
            np.datetime64(f"{y}-01-01T00"),
            np.datetime64(f"{y}-12-31T11"),
        )).mean()
        for y in y_uniq
    ])
    tides_mm -= annual_msl[y_inverse]
    return tides_mm


def _set_referenced(stdata):
    if stdata['gsrc'] in ["gtsm", "codec"]:
        stdata['tide_levels'] = stdata['waterlevel']
        stdata['tide_levels_full'] = stdata['waterlevel'].copy()
    else:
        # tides from harmonics coefficients
        times = stdata['waterlevel'].index.values
        suffixes = {
            # the 90-day harmonics are the default ones, the full harmonics are just for checking
            "": "_full",
            "_90d": "",
        }
        for harmonics_suffix, tl_suffix in suffixes.items():
            tide_fun = harmonics_fun(stdata[f'harmonics{harmonics_suffix}'])
            stdata[f'tide_levels{tl_suffix}'] = pd.Series(tide_fun(times), index=times)
        stdata['combined'] = stdata['waterlevel']

    # the "waterlevel" property is ambiguous, remove it
    del stdata['waterlevel']

    if 'annual_msl' not in stdata:
        print("No annual_msl for", stdata['gsrc'], stdata['filename'])
        stdata['annual_msl'] = 0

    # Use the surge anomaly for the analyses because we aren't able to model tidal variations.
    stdata['anomaly'] = stdata['combined'] - stdata['tide_levels']

    # Shift the anomaly in height so that the maximum anomaly is as high as the maximum observed
    # waterlevel since we have to replicate this maximum in order to get the flood plains right.
    # Note that the max anomaly might be attained at a different time than the max waterlevel.
    max_tide_rel = stdata['combined'].max() - stdata['annual_msl']
    stdata['referenced'] = stdata['anomaly'] + (max_tide_rel - stdata['anomaly'].max())

    mm_conv_factor = u_const.GAUGE_SOURCE_MM_CONVERSION[stdata['gsrc']]
    mm_conv_vars = [
        'tide_levels', 'tide_levels_full', 'anomaly', 'referenced', 'combined', 'annual_msl'
    ]
    for v in mm_conv_vars:
        stdata[v] *= mm_conv_factor


def _gc_gauges_for_station(gcdata, stdata):
    # for each zos, return the closest GC station
    st_locations = np.stack([stdata['gc_location'], stdata['location']], axis=0)
    zos_names = set(gd['zos'] for gd in gcdata)
    filtered = []
    for mname in zos_names:
        sub_gcdata = [gd for gd in gcdata if gd['zos'] == mname]
        gc_locations = np.stack([gc['location'] for gc in sub_gcdata], axis=0)
        dists = u_coord.dist_approx(
            gc_locations[None, :, 0], gc_locations[None, :, 1],
            st_locations[None, :, 0], st_locations[None, :, 1],
            method="geosphere")[0, :, :].min(axis=1)
        argmin_dists = np.argmin(dists)
        min_dist = dists[argmin_dists]
        if min_dist >= 30:
            print(
                f"Warning: Far GC gauge ({min_dist:.1f} km) for "
                f"{stdata['map_id']} {stdata['gsrc']} {ſtdata['filename']} {mname}"
            )
        filtered.append(sub_gcdata[argmin_dists])
    return filtered


def _gc_to_referenced(gd, stdata, referenced, resample):
    altimetry_src = gd['zos'].split("-")[0]
    st_annual_msl = stdata[f'annual_msl_{altimetry_src}']

    gd['referenced'] = []
    gd['annual_msl'] = st_annual_msl

    if len(gd["time"]) == 0:
        return

    sub_ref = st_annual_msl if referenced else 0
    max_lvl = max(max(l) for l in gd["amr_level"])
    for i_run, t in enumerate(gd["time"]):
        l = gd['amr_level'][i_run]
        h = gd['height_above_ground'][i_run]
        sl = gd['height_above_geoid'][i_run]

        # truncate at least the first 5 time steps from the data
        # that's often required for geoclaw to get started properly
        idx0 = 5

        # only consider two highest available refinement levels
        idx0 = l.size if not any(lvl in l[idx0:] for lvl in [max_lvl - 1, max_lvl]) else idx0
        idx0 = max(idx0, l.size if idx0 >= l.size else idx0 + np.argmin(l[idx0:] < max_lvl - 1))

        # truncate height_above_ground==0 from the beginning of the time series
        idx0 = max(idx0, h.size if idx0 >= h.size else idx0 + np.argmin(h[idx0:] == 0))

        gd['referenced'].append(
            pd.Series((sl[idx0:]  - sub_ref) * 1000, index=t[idx0:]).dropna()
            # resample to temporal averages
            .resample(resample).mean().dropna()
        )


def _assign_gc_gauges(source, map_id, gdata, referenced, zos, resample):
    ibtracs_id = map_id.split("-")[0]
    gcdata_all = []
    gcdata_dir = u_const.GEOCLAW_DIR / source / "results"
    fname = f"{ibtracs_id}_{source}-zos_{zos}-gauge_data.pickle"
    for path in gcdata_dir.glob(fname):
        m = re.match(r".*_.*-zos_(.*-fes.*)-gauge_data", path.stem)
        mod_zos = m.group(1)
        with path.open("rb") as fp:
            tmp = pickle.load(fp)[0]
            for gd in tmp:
                gd['zos'] = mod_zos
            gcdata_all.extend(tmp)

    if all(len(gd['time']) == 0 for gd in gcdata_all):
        print("No GC data for", map_id)
        gcdata_all = None

    for stdata in gdata:
        if gcdata_all is None:
            stdata['geoclaw'] = None
            continue

        gcdata_st = _gc_gauges_for_station(gcdata_all, stdata)
        for gd in gcdata_st:
            _gc_to_referenced(gd, stdata, referenced, resample)

        stdata['geoclaw'] = gcdata_st


def load_gaugedata(source, map_id, by_gsrc=True, referenced=True, geoclaw_zos=None,
                   geoclaw_resample="1h", include_discarded=False, filter_gsrc=None):
    """Prepare gauge data for presentation and further processing

    1. Exclude records that were marked as "discarded" during preprocessing. This means:
       * There are several records in the same location, and others have more complete data.
       * There is a station, but it doesn't have data in the event period.
       * There is a station, but it doesn't have enough data to compute an annual mean.
    2. Include only records from the specified sources.
    3. For each station, load the annual MSL according to altimetry products.
    4. Reference all records to the local annual MSL.
    5. Generate surge series, by removing the tidal signal.
    6. Convert all values to mm.
    7. Assign GeoClaw outputs to each station, and reference to altimetry's MSL.

    Parameters
    ----------
    source, map_id : str
        The source ("gfd", "rapid", "dfo") and ID of the floodmap to consider.
    by_gsrc : bool, optional
        If True, return a dictionary with the gauge sources as keys. Otherwise, a list of all
        records will be returned. In any case, the gauge source information will be included for
        each individual record in the "gsrc" attribute. Default: True
    referenced : bool, optional
        If False, omit the referencing step, but return the raw recorded data instead. This will
        also omit the conversion to mm. Default: True
    geoclaw_zos : str or None, optional
        If given, GeoClaw outputs are assigned to each station. This can be a glob such
        as "aviso-fes_*" which will include one GeoClaw output for each of the zos parameters
        matching this glob. In general, only the single GeoClaw output that best overlaps with the
        period around the wind maximum is included for each zos parameter. Default: None
    geoclaw_resample : str, optional
        If `geoclaw_zos` is given, this specifies the temporal resolution of the GeoClaw records
        since the original output resolution can be extremely high. Default: "1h"
    include_discarded : bool, optional
        If True, include records that have been discarded during preprocessing. Default: False
    filter_gsrc : list or None, optional
        If given, only records from the specified gauge sources will be included.

    Returns
    -------
    dict or list
    """
    path = u_const.GAUGES_DIR / source / "records" / f"{map_id}.pickle"
    with path.open("rb") as fp:
        gdata_by_gsrc = {
            gsrc: [stdata for stdata in stations if include_discarded or not stdata['discarded']]
            for gsrc, stations in pickle.load(fp).items()
        }
        if filter_gsrc is not None:
            gdata_by_gsrc = {
                gsrc: gdata_by_gsrc.get(gsrc, [])
                for gsrc in filter_gsrc
            }

    map_year = int(map_id[:4])
    df_annual_msl = {}
    if geoclaw_zos is not None:
        altimetry_src_glob = geoclaw_zos.split("-")[0]
        for p in u_const.GAUGES_DIR.glob(f"annual_msl_{altimetry_src_glob}.hdf5"):
            altimetry_src = p.stem[11:]
            df = pd.read_hdf(p)
            df_annual_msl[altimetry_src] = df.set_index('years').loc[map_year, :]

    all_gdata = []
    for gsrc, stations in gdata_by_gsrc.items():
        for stdata in stations:
            stdata['map_id'] = map_id
            stdata['gsrc'] = gsrc
            for alitmetry_src, df in df_annual_msl.items():
                stdata[f'annual_msl_{alitmetry_src}'] = df[stdata['filename']]
            if referenced:
                _set_referenced(stdata)
            all_gdata.append(stdata)

    if geoclaw_zos is not None:
        _assign_gc_gauges(source, map_id, all_gdata, referenced, geoclaw_zos, geoclaw_resample)

    return gdata_by_gsrc if by_gsrc else all_gdata


def load_gauge_locations(source, gsrc):
    """Load all tide gauge locations for the specified source

    Parameters
    ----------
    source : str
        The source ("gfd", "rapid", "dfo") of the floodmap to consider.
    gsrc : str
        The gauge source (e.g. "gesla3") to consider.

    Returns
    -------
    DataFrame
    """
    gauges_dir = u_const.COMPARE_DIR / source / "gauges"
    gaugedata = []
    for path in gauges_dir.glob("*.pickle"):
        with path.open("rb") as fp:
            gaugedata.extend(
                [stdata for stdata in pickle.load(fp)
                 if stdata['affected'] and stdata['valid']
                 and stdata['gsrc'] == gsrc]
            )
    return pd.DataFrame({
        "source": source,
        "gsrc": gsrc,
        "name": [stdata['filename'] for stdata in gaugedata],
        "map_id": [stdata['map_id'] for stdata in gaugedata],
        "lat": [stdata['location'][0] for stdata in gaugedata],
        "lon": [stdata['location'][1] for stdata in gaugedata],
    })
