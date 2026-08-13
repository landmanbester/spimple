import numpy as np
from ducc0.fft import c2r, good_size, r2c

iFs = np.fft.ifftshift  # noqa: N816 - domain idiom, paired with Fs
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
    # the docstring's contract: nsigma standard deviations of the major axis
    sigma_maj = Smaj / (2 * np.sqrt(2 * np.log(2)))
    extent = (nsigma * sigma_maj) ** 2
    xflat = xin.squeeze()
    yflat = yin.squeeze()
    idx, idy = np.where(xflat**2 + yflat**2 <= extent)
    x = np.array([xflat[idx, idy].ravel(), yflat[idx, idy].ravel()])
    R = np.einsum("nb,bc,cn->n", x.T, A, x)
    # GaussPar is FWHM: a Gaussian with FWHM S needs exp(-4 ln2 x^2 / S^2),
    # i.e. the coefficient is 0.5 * fwhm_conv**2, not fwhm_conv. The previous
    # form made every kernel 8.51% too wide.
    fwhm_conv = 2 * np.sqrt(2 * np.log(2))
    tmp = np.exp(-0.5 * fwhm_conv**2 * R)
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


def convolve2gaussres(image, xx, yy, gaussparf, nthreads, gausspari=None, pfrac=0.5, norm_kernel=False, yx_order=False):
    """
    Convolves the image to a specified resolution.

    Parameters
    ----------
    image - (nband, nx, ny) array to convolve, or (nband, ny, nx) with yx_order.
    xx/yy - coordinates on the grid in the same units as gaussparf. ALWAYS built
            x-major, from nx then ny, whatever yx_order is.
    gaussparf - Gaussian parameters of the desired resolution (emaj, emin, pa),
                either shared by every plane as a (3,) tuple or per plane as a
                (nband, 3) array.
    gausspari - initial resolution. By default it is assumed that the image
                is a clean component image with no associated resolution.
                If specified, it must contain gausspars for each imaging band
                in the same format.
    nthreads - number of threads to use for the FFT's.
    pfrac - padding used for the FFT based convolution. Will pad by pfrac/2 on
            both sides of image
    norm_kernel - normalise the Gaussian kernel to have volume 1.
    yx_order - set True for (Y, X)-ordered DataTree arrays. The convolution is
               defined x-major; this transposes in and out so the position angle
               keeps its meaning.
    """
    if yx_order:
        image = image.transpose(0, 2, 1)
    nband, nx, ny = image.shape

    gaussparf = np.asarray(gaussparf, dtype=np.float64)
    per_plane = gaussparf.ndim > 1
    if per_plane and gaussparf.shape[0] != nband:
        raise ValueError(f"gaussparf must be of length {nband}, got {gaussparf.shape[0]}")
    if gausspari is not None and np.ndim(gausspari) > 1 and np.shape(gausspari)[0] != nband:
        raise ValueError(f"gausspari must be of length {nband}, got {np.shape(gausspari)[0]}")

    padding, unpad_x, unpad_y = get_padding_info(nx, ny, pfrac)
    ax = (1, 2)  # axes over which to perform fft
    lastsize = ny + np.sum(padding[-1])

    if per_plane:
        gausskern = np.stack([Gaussian2D(xx, yy, tuple(gaussparf[i]), normalise=norm_kernel) for i in range(nband)])
    else:
        gausskern = Gaussian2D(xx, yy, tuple(gaussparf), normalise=norm_kernel)[None]
    gausskernhat = r2c(
        iFs(np.pad(gausskern, padding, mode="constant"), axes=ax),
        axes=ax,
        forward=True,
        nthreads=nthreads,
        inorm=0,
    )

    image = np.pad(image, padding, mode="constant").astype(np.float64)
    imhat = r2c(iFs(image, axes=ax), axes=ax, forward=True, nthreads=nthreads, inorm=0)

    # convolve to desired resolution
    if gausspari in [None, ()]:
        imhat *= gausskernhat
    else:
        for i in range(nband):
            thiskern = Gaussian2D(xx, yy, tuple(gausspari[i]), normalise=norm_kernel).astype(np.float64)
            thiskern = np.pad(thiskern[None], padding, mode="constant")
            thiskernhat = r2c(
                iFs(thiskern, axes=ax),
                axes=ax,
                forward=True,
                nthreads=nthreads,
                inorm=0,
            )

            target = gausskernhat[i] if per_plane else gausskernhat[0]
            if not np.all(np.isnan(thiskernhat)):
                convkernhat = np.where(np.abs(thiskernhat[0]) > 1e-10, target / thiskernhat[0], 0.0)
            else:
                print("Nan values have been encountered. Subverting RuntimeWarning")
                convkernhat = np.zeros_like(thiskernhat[0]).astype("complex")

            imhat[i] *= convkernhat

    image = Fs(
        c2r(imhat, axes=ax, forward=False, lastsize=lastsize, inorm=2, nthreads=nthreads),
        axes=ax,
    )[:, unpad_x, unpad_y]

    if yx_order:
        return image.transpose(0, 2, 1), gausskern.transpose(0, 2, 1)
    return image, gausskern
