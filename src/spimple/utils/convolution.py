from ducc0.fft import c2r, good_size, r2c
import numpy as np

iFs = np.fft.ifftshift
Fs = np.fft.fftshift


def Gaussian2D(xin, yin, GaussPar=(1.0, 1.0, 0.0), normalise=True, nsigma=5):
    """
    xin         - grid of x coordinates
    yin         - grid of y coordinates
    GaussPar    - (emaj, emin, pa) with emaj/emin in units x and pa in radians.
    normalise   - normalise kernel to have volume 1
    nsigma      - compute kernel out to this many sigmas

    Note - the rotation matrix is defined as

    [[np.sin(PA), -np.cos(PA)],
     [np.cos(PA), np.sin(PA)]]

    instead of

    [[np.cos(PA), -np.sin(PA)],
     [np.sin(PA), np.cos(PA)]]

    with t = pi/2 - pa

    for compatibility with fits
    """
    Smaj, Smin, PA = GaussPar
    A = np.array([[1.0 / Smaj**2, 0], [0, 1.0 / Smin**2]])
    R = np.array([[np.sin(PA), -np.cos(PA)], [np.cos(PA), np.sin(PA)]])
    A = np.dot(np.dot(R.T, A), R)
    sOut = xin.shape
    # only compute the result out to 5 * emaj
    extent = (nsigma * Smaj) ** 2
    xflat = xin.squeeze()
    yflat = yin.squeeze()
    idx, idy = np.where(xflat**2 + yflat**2 <= extent)
    x = np.array([xflat[idx, idy].ravel(), yflat[idx, idy].ravel()])
    R = np.einsum("nb,bc,cn->n", x.T, A, x)
    # need to adjust for the fact that GaussPar corresponds to FWHM
    fwhm_conv = 2 * np.sqrt(2 * np.log(2))
    tmp = np.exp(-fwhm_conv * R)
    gausskern = np.zeros(xflat.shape, dtype=np.float64)
    gausskern[idx, idy] = tmp

    if normalise:
        gausskern /= np.sum(gausskern)
    return np.ascontiguousarray(gausskern.reshape(sOut), dtype=np.float64)


def get_padding_info(nx, ny, pfrac):
    npad_x = int(pfrac * nx)
    nfft = good_size(nx + npad_x, True)
    npad_xl = (nfft - nx) // 2
    npad_xr = nfft - nx - npad_xl

    npad_y = int(pfrac * ny)
    nfft = good_size(ny + npad_y, True)
    npad_yl = (nfft - ny) // 2
    npad_yr = nfft - ny - npad_yl
    padding = ((0, 0), (npad_xl, npad_xr), (npad_yl, npad_yr))
    unpad_x = slice(npad_xl, -npad_xr)
    unpad_y = slice(npad_yl, -npad_yr)
    return padding, unpad_x, unpad_y


def convolve2gaussres(image, xx, yy, gaussparf, nthreads, gausspari=None, pfrac=0.5, norm_kernel=False):
    """
    Convolves the image to a specified resolution.

    Parameters
    ----------
    Image - (nband, nx, ny) array to convolve
    xx/yy - coordinates on the grid in the same units as gaussparf.
    gaussparf - tuple containing Gaussian parameters of desired resolution
                (emaj, emin, pa).
    gausspari - initial resolution . By default it is assumed that the image
                is a clean component image with no associated resolution.
                If beampari is specified, it must be a tuple containing gausspars
                for each imaging band in the same format.
    nthreads - number of threads to use for the FFT's.
    pfrac - padding used for the FFT based convolution. Will pad by pfrac/2 on
            both sides of image
    """
    nband, nx, ny = image.shape
    padding, unpad_x, unpad_y = get_padding_info(nx, ny, pfrac)
    ax = (1, 2)  # axes over which to perform fft
    lastsize = ny + np.sum(padding[-1])

    gausskern = Gaussian2D(xx, yy, gaussparf, normalise=norm_kernel)
    gausskern = np.pad(gausskern[None], padding, mode="constant")
    gausskernhat = r2c(iFs(gausskern, axes=ax), axes=ax, forward=True, nthreads=nthreads, inorm=0)

    image = np.pad(image, padding, mode="constant").astype(np.float64)
    imhat = r2c(iFs(image, axes=ax), axes=ax, forward=True, nthreads=nthreads, inorm=0)

    # convolve to desired resolution
    if gausspari in [None, ()]:
        imhat *= gausskernhat
    else:
        for i in range(nband):
            thiskern = Gaussian2D(xx, yy, gausspari[i], normalise=norm_kernel).astype(np.float64)
            thiskern = np.pad(thiskern[None], padding, mode="constant")
            thiskernhat = r2c(
                iFs(thiskern, axes=ax),
                axes=ax,
                forward=True,
                nthreads=nthreads,
                inorm=0,
            )

            if not np.all(np.isnan(thiskernhat)):
                convkernhat = np.where(np.abs(thiskernhat) > 1e-10, gausskernhat / thiskernhat, 0.0)
            else:
                print("Nan values have been encountered. Subverting RuntimeWarning")
                convkernhat = np.zeros_like(thiskernhat).astype("complex")

            imhat[i] *= convkernhat[0]

    image = Fs(
        c2r(imhat, axes=ax, forward=False, lastsize=lastsize, inorm=2, nthreads=nthreads),
        axes=ax,
    )[:, unpad_x, unpad_y]

    return image, gausskern[:, unpad_x, unpad_y]
