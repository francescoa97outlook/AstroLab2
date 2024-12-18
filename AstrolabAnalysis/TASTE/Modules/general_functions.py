import numpy as np
from astropy.io import fits


def read_fits(path_f):
    """
    Reads a FITS file and extracts the header, header comments, and data.

    Args:
        path_f (str): Path to the FITS file.

    Returns:
        tuple: A tuple containing:
            - header (astropy.io.fits.Header): FITS header.
            - header_comments (astropy.io.fits.CardList): Comments in the header.
            - data (np.ndarray): FITS data array.
    """
    hdul = fits.open(path_f)
    header = hdul[0].header
    header_comments = hdul[0].header.comments
    data = hdul[0].data
    hdul.close()
    return header, header_comments, data


def get_header_info(header, header_comments, first=None):
    """
    Extracts key header information from a FITS file and optionally prints the information.

    Args:
        header (astropy.io.fits.Header): FITS header.
        header_comments (astropy.io.fits.CardList): Comments in the header.
        first (str, optional): Optional label to indicate the source of the header information.

    Returns:
        tuple: A tuple containing:
            - JD (float): Julian Date.
            - AIRMASS (float): Airmass of the observation.
            - GAIN (float): Gain value.
            - GAIN_comments (str): Comments about the gain.
            - RDNOISE (float): Readout noise value.
            - RDNOISE_comments (str): Comments about the readout noise.
            - NAXIS1 (int): Number of pixels along the x-axis.
            - NAXIS2 (int): Number of pixels along the y-axis.
            - EXPTIME (float): Exposure time in seconds.
            - SITELAT (float): Latitude of the observation site.
            - SITELONG (float): Longitude of the observation site.
            - OBJCTRA (str): Right Ascension of the target.
            - OBJCTDEC (str): Declination of the target.
    """
    JD = header["JD"]
    AIRMASS = header["AIRMASS"]
    GAIN = header["GAIN"]
    GAIN_comments = header_comments["GAIN"]
    RDNOISE = header["RDNOISE"]
    RDNOISE_comments = header_comments["RDNOISE"]
    NAXIS1 = header["NAXIS1"]
    NAXIS2 = header["NAXIS2"]
    EXPTIME = header['EXPTIME']
    SITELAT = header['SITELAT']
    SITELONG = header['SITELONG']
    OBJCTRA = header['OBJCTRA']
    OBJCTDEC = header['OBJCTDEC']
    if first:
        print(
            "JD first " + first + " " + str(JD) + "\n" +
            "AIRMASS first " + first + " " + str(AIRMASS) + "\n" +
            "GAIN first " + first + " " + str(GAIN) + ", " + GAIN_comments + "\n" +
            "RDNOISE first " + first + " " + str(RDNOISE) + ", " + RDNOISE_comments + "\n" +
            "NAXIS1 first " + first + " " + str(NAXIS1) + "\n" +
            "NAXIS2 first " + first + " " + str(NAXIS2) + "\n" +
            "EXPTIME first " + first + " " + str(EXPTIME) + "\n" +
            "SITELAT first " + first + " " + str(SITELAT) + "\n" +
            "SITELONG first " + first + " " + str(SITELONG) + "\n" +
            "OBJCTRA first " + first + " " + str(OBJCTRA) + "\n" +
            "OBJCTDEC first " + first + " " + str(OBJCTDEC) + "\n"
        )
    return (
        JD, AIRMASS, GAIN, GAIN_comments, RDNOISE, RDNOISE_comments, NAXIS1, NAXIS2, EXPTIME,
        SITELAT, SITELONG, OBJCTRA, OBJCTDEC
    )


def make_circle_around_star2(x_pos, y_pos, radius, thickness=0.5, label='', color='w', alpha=1., axf=None):
    """
    Draws a circular region around a star with a specified radius and thickness.

    Args:
        x_pos (float): X-coordinate of the circle's center.
        y_pos (float): Y-coordinate of the circle's center.
        radius (float): Radius of the circle.
        thickness (float, optional): Thickness of the circle. Defaults to 0.5.
        label (str, optional): Label for the circle. Defaults to an empty string.
        color (str, optional): Color of the circle. Defaults to 'w' (white).
        alpha (float, optional): Transparency level of the circle. Defaults to 1.0.
        axf (matplotlib.axes._axes.Axes, optional): Matplotlib axes on which to draw. Defaults to None.
    """
    n, radii = 50, [radius, radius + thickness]
    theta = np.linspace(0, 2 * np.pi, n, endpoint=True)
    xs = np.outer(radii, np.cos(theta))
    ys = np.outer(radii, np.sin(theta))
    # in order to have a closed area, the circles
    # should be traversed in opposite directions
    xs[1, :] = xs[1, ::-1]
    ys[1, :] = ys[1, ::-1]
    axf.fill(
        np.ravel(xs) + x_pos, np.ravel(ys) + y_pos, edgecolor=None, facecolor=color,
        alpha=alpha, label=label
    )


def make_annulus_around_star(x_pos, y_pos, inner_radius, outer_radius, label='', color='y', ax=None):
    """
    Draws an annular region around a star with specified inner and outer radii.

    Args:
        x_pos (float): X-coordinate of the annulus's center.
        y_pos (float): Y-coordinate of the annulus's center.
        inner_radius (float): Inner radius of the annulus.
        outer_radius (float): Outer radius of the annulus.
        label (str, optional): Label for the annulus. Defaults to an empty string.
        color (str, optional): Color of the annulus. Defaults to 'y' (yellow).
        ax (matplotlib.axes._axes.Axes, optional): Matplotlib axes on which to draw. Defaults to None.
    """
    n, radii = 50, [inner_radius, outer_radius]
    theta = np.linspace(0, 2 * np.pi, n, endpoint=True)
    xs = np.outer(radii, np.cos(theta))
    ys = np.outer(radii, np.sin(theta))
    # in order to have a closed area, the circles
    # should be traversed in opposite directions
    xs[1, :] = xs[1, ::-1]
    ys[1, :] = ys[1, ::-1]
    ax.fill(np.ravel(xs) + x_pos, np.ravel(ys) + y_pos, edgecolor=None, facecolor=color, alpha=0.75, label=label)