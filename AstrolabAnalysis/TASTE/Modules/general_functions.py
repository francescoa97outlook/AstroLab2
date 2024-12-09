import numpy as np
from astropy.io import fits


def read_fits(path_f):
    hdul = fits.open(path_f)
    header = hdul[0].header
    header_comments = hdul[0].header.comments
    data = hdul[0].data
    hdul.close()
    return header, header_comments, data


def get_header_info(header, header_comments, first=None):
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
    n, radii = 50, [inner_radius, outer_radius]
    theta = np.linspace(0, 2 * np.pi, n, endpoint=True)
    xs = np.outer(radii, np.cos(theta))
    ys = np.outer(radii, np.sin(theta))
    # in order to have a closed area, the circles
    # should be traversed in opposite directions
    xs[1, :] = xs[1, ::-1]
    ys[1, :] = ys[1, ::-1]
    ax.fill(np.ravel(xs) + x_pos, np.ravel(ys) + y_pos, edgecolor=None, facecolor=color, alpha=0.75, label=label)