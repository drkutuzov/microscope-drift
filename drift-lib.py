import numpy as np
from scipy.fft import dst

#------------------Median-estimator for locating large fluorescent beads------------------
def index_median(y):
    """
    Estimate the index of the median position of a 1D intensity distribution.

    Parameters
    ----------
    y : xarray.DataArray
        One-dimensional intensity distribution with coordinate `"x"`.

    Returns
    -------
    int
        Index whose normalized cumulative intensity is closest to 0.5.
    """
    cs = y.cumsum("x")

    cs_norm = (cs - cs.isel(x=0)) / (cs.isel(x=-1) - cs.isel(x=0))

    return int(abs(cs_norm - 0.5).argmin("x"))


def median(y):
    """
    Estimate the center of a 1D intensity distribution using a median estimator
    with subpixel precision.

    Parameters
    ----------
    y : xarray.DataArray
        One-dimensional intensity distribution with coordinate `"x"`.
        The `"x"` coordinate is assumed to be uniformly spaced.

    Returns
    -------
    float
        Estimated median position in `"x"` units.
    """
    ind_med = index_median(y)

    dx = float(y.x.isel(x=1) - y.x.isel(x=0))

    sum_left = float(y.isel(x=slice(None, ind_med)).sum())
    sum_right = float(y.isel(x=slice(ind_med + 1, None)).sum())

    alpha = 0.5 + (sum_right - sum_left) / (2 * float(y.isel(x=ind_med)))

    return float(y.x.isel(x=ind_med - 1) + dx * alpha)

###--------------------------------------------------------------------------
###--------------Spectral analysis: Detrending and Periodograms--------------
def detrend(darray):
    """
    Remove linear drift from a time series.

    Parameters
    ----------
    darray : xarray.DataArray
        1D time series with coordinate `"t"`.

    Returns
    -------
    xarray.DataArray
        Detrended signal.
    """
    p = darray[0] + (darray[-1] - darray[0]) / (darray.t[-1] - darray.t[0]) * darray.t

    return darray - p


def custom_dst(x_dtr, dt):
    """
    Compute sine-transform coefficients of a detrended signal.

    Parameters
    ----------
    x_dtr : array-like, shape (N+1,)
        Detrended time series.
    dt : float
        Time step.

    Returns
    -------
    np.ndarray
        Coefficients b_k (k = 0,...,N), with b[0] = b[N] = 0.
    """
    x_dtr = np.asarray(x_dtr, dtype=float)
    N = len(x_dtr) - 1

    if N < 1:
        raise ValueError("x_dtr must contain at least two samples.")

    y = dst(x_dtr[1:N], type=1, norm=None)

    b = np.zeros(N + 1)
    b[1:N] = dt * y

    return b


def periodogram(x_dtr, dt):
    """
    Compute periodogram of a detrended signal.

    Parameters
    ----------
    x_dtr : array-like, shape (N+1,)
        Detrended time series.
    dt : float
        Time step.

    Returns
    -------
    np.ndarray
        Periodogram values.
    """
    x_dtr = np.asarray(x_dtr, dtype=float)
    tmsr = dt * (len(x_dtr) - 1)

    b = custom_dst(x_dtr, dt)
    return b**2 / (2 * tmsr)

###-------------------------------------------------------
###--------------MLE-fitting of Periodograms--------------

def spectrum(k, D, b, tmsr):
    """
    Theoretical power spectrum.

    Parameters
    ----------
    k : array-like
        Frequency indices.
    D : float
        Diffusion coefficient.
    b : float
        White-noise offset.
    tmsr : float
        Total measurement time.

    Returns
    -------
    np.ndarray
        Spectrum values P_e(k).
    """
    df = 1 / (2 * tmsr)
    return b + D / 2 / (np.pi * k * df) ** 2


def nll(p, Po, k, tmsr):
    """
    Negative log-likelihood up to an additive constant.

    Parameters
    ----------
    p : array-like
        Fit parameters [D, b].
    Po : array-like
        Observed periodogram values.
    k : array-like
        Frequency indices.
    tmsr : float
        Total measurement time.

    Returns
    -------
    float
        Negative log-likelihood.
    """
    Po = np.asarray(Po, dtype=float)
    D, b = p
    Pe = spectrum(k, D, b, tmsr)

    return np.sum(0.5 * np.log(Pe) + Po / (2.0 * Pe))


def nll_grad(p, Po, k, tmsr):
    """
    Gradient of the negative log-likelihood.

    Parameters
    ----------
    p : array-like
        Fit parameters [D, b].
    Po : array-like
        Observed periodogram values.
    k : array-like
        Frequency indices.
    tmsr : float
        Total measurement time.

    Returns
    -------
    np.ndarray
        Gradient [d/dD, d/db].
    """
    Po = np.asarray(Po, dtype=float)
    k = np.asarray(k, dtype=float)

    D, b = p
    Pe = spectrum(k, D, b, tmsr)

    alpha = 2 * tmsr**2 / (np.pi**2 * k**2)
    w = (Pe - Po) / (2 * Pe**2)

    dD = np.sum(w * alpha)
    db = np.sum(w)

    return np.array([dD, db])


def fisher_covariance(k, Dhat, bhat, tmsr):
    """
    Asymptotic covariance matrix from Fisher information.

    Parameters
    ----------
    k : array-like
        Frequency indices.
    Dhat : float
        Fitted diffusion coefficient.
    bhat : float
        Fitted noise offset.
    tmsr : float
        Total measurement time.

    Returns
    -------
    np.ndarray
        2x2 covariance matrix for [D, b].
    """
    k = np.asarray(k, dtype=float)

    alpha = 2 * tmsr**2 / (np.pi**2 * k**2)
    Pe = spectrum(k, Dhat, bhat, tmsr)

    S0 = np.sum(1.0 / Pe**2)
    S1 = np.sum(alpha / Pe**2)
    S2 = np.sum(alpha**2 / Pe**2)

    det = S0 * S2 - S1**2

    cov = (2.0 / det) * np.array([
        [S0, -S1],
        [-S1, S2]
    ])

    return cov


def block_average_variable(x, block_sizes, t=None):
    """
    Block-average a 1D array using variable block sizes.

    Parameters
    ----------
    x : array-like
        Input data.
    block_sizes : array-like
        Lengths of consecutive blocks.
    t : array-like, optional
        Coordinates for block centers. If None, indices are used.

    Returns
    -------
    means : np.ndarray
        Mean value in each block.
    sems : np.ndarray
        Standard error of the mean in each block.
    centers : np.ndarray
        Mean coordinate of each block.
    """
    x = np.asarray(x, dtype=float)

    if t is None:
        t = np.arange(len(x))
    else:
        t = np.asarray(t)

    means = []
    sems = []
    centers = []

    idx = 0

    for size in block_sizes:
        if idx + size > len(x):
            break

        xb = x[idx:idx + size]
        tb = t[idx:idx + size]

        means.append(np.mean(xb))

        if size > 1:
            sems.append(np.std(xb, ddof=1) / np.sqrt(size))
        else:
            sems.append(np.nan)

        centers.append(np.mean(tb))
        idx += size

    return np.array(means), np.array(sems), np.array(centers)