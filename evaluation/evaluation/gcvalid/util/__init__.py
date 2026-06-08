
import numpy as np

import gcvalid.util.constants as u_const


def dt64_to_dmy(date):
    """Convert datetime64 values to separate values for day month and year

    Parameters
    ----------
    date : scalar or array-like of type datetime64
        The value or array of values to convert.

    Returns
    -------
    day, month, year : integers with same shape as input
        The day, month and year corresponding to the input data.
    """
    year = date.astype('datetime64[Y]').astype(int) + 1970
    month = date.astype('datetime64[M]').astype(int) % 12 + 1
    day = (date - date.astype('datetime64[M]') + 1).astype(int)
    return day, month, year


def saffir_simpson_category(winds_ms):
    """Assign a (numerical) Saffir-Simpson hurricane category to each wind speed value

    Parameters
    ----------
    winds_ms : ndarray of type float
        The wind speeds in meters per second as a one-dimensional array.

    Returns
    -------
    categories : ndarray of type int
        The Saffir-Simpson hurricane category corresponding to each of the specified wind speeds
        with values ranging from -1 (Tropical Depression) and 0 (Tropical Storm) to 5 (Category 5).
    """
    return 5 - np.argmin(
        winds_ms[:, None] < np.array([0] + u_const.SAFFIR_SIMPSON_THRESHS)[None, ::-1],
        axis=1,
    )


def rectangles_disjoint(r1, r2):
    """Check whether two rectangles are disjoint

    Parameters
    ----------
    r1, r2 : tuples (xmin, ymin, xmax, ymax)
        The rectangles as tuples in "bounds" order.

    Returns
    -------
    bool
    """
    return (
        (r1[0] > r2[2] or r1[2] < r2[0])
        or (r1[1] > r2[3] or r1[3] < r2[1])
    )


def rectangular_components(rectangles):
    """Set of disjoint rectangles that cover the specified rectangles

    Parameters
    ----------
    rectangles : list of tuples (xmin, ymin, xmax, ymax)
        The rectangles to cover as tuples in "bounds" order.

    Returns
    -------
    rectangles : list of tuples (xmin, ymin, xmax, ymax)
        A disjoint list of rectangles (as tuples in "bounds" order) that cover the input
    """
    i = 0
    while i < len(rectangles):
        rect = rectangles.pop(i)

        # check for overlaps
        disj = [rectangles_disjoint(rect, r) for r in rectangles]

        if all(disj):
            rectangles.insert(i, rect)
            i += 1
            continue

        # determine bounds of connected component
        comp = [r for r, d in zip(rectangles, disj) if not d]
        comp.append(rect)
        comp = np.array(comp)
        comp = (
            comp[:, 0].min(),
            comp[:, 1].min(),
            comp[:, 2].max(),
            comp[:, 3].max(),
        )

        # rebuild list of rectangles
        rectangles = [comp] + [
            r for r, d in zip(rectangles, disj) if d
        ]

        # restart checking for overlaps
        i = 0

    return rectangles
