"""
envos/obs/convolve.py — convolve, Convolver

Moved from envos/obs.py (P2-B split, move-only, no behaviour changes).
"""
import numpy as np
from scipy import signal
import astropy.convolution as aconv

from ..log import logger


def convolve(
    image,
    beam_maj_au=None,
    beam_min_au=None,
    vreso_kms=None,
    beam_pa_deg=0,
    mode="scipy",
):
    convolver = Convolver(
        (image.dx_au, image.dy_au, image.dv_kms),
        beam_maj_au,
        beam_min_au,
        vreso_kms,
        beam_pa_deg,
        mode,
    )
    return convolver(image.Ipv)


class Convolver:
    """
    shape of 3d-image is (ix_max, jy_max, kv_max)
    Kernel

    """

    def __init__(
        self,
        grid_size,
        beam_maj_au=None,
        beam_min_au=None,
        vreso_kms=None,
        beam_pa_deg=0,
        mode="scipy",
    ):
        # relation : standard deviation = 1/(2 sqrt(ln(2))) * FWHM of Gaussian
        # theta_deg : cclw is positive
        self.mode = mode
        if mode == "null":
            return
        if beam_maj_au is None or beam_min_au is None:
            raise ValueError(
                "beam size is required for convolution: "
                "set beam_maj_au and beam_min_au, or use mode='null'"
            )
        sigma_over_FWHM = 2 * np.sqrt(2 * np.log(2))
        conv_size = [beam_maj_au + 1e-100, beam_min_au + 1e-100]
        if vreso_kms is not None:
            conv_size += [vreso_kms + 1e-100]
        stddev = np.array(conv_size) / np.array(grid_size) / sigma_over_FWHM
        beampa = np.radians(beam_pa_deg)
        self.Kernel_xy2d = aconv.Gaussian2DKernel(
            x_stddev=stddev[0], y_stddev=stddev[1], theta=beampa
        )._array
        if len(conv_size) == 3 and (conv_size[2] is not None):
            Kernel_v1d = aconv.Gaussian1DKernel(stddev[2])._array
            self.Kernel_3d = np.multiply(
                # self.Kernel_xy2d[np.newaxis, :, :],
                self.Kernel_xy2d[:, :, np.newaxis],
                Kernel_v1d[np.newaxis, np.newaxis, :],
            )

    def __call__(self, image):
        if self.mode == "null":
            return image

        if len(image.shape) == 2 or image.shape[2] == 1:
            Kernel = self.Kernel_xy2d
            logger.info("Convolving image with 2d-Kernael")
        elif len(image.shape) == 3:
            Kernel = self.Kernel_3d
            logger.info("Convolving image with 3d-Kernael")
        else:
            raise Exception("Unknown data.image shape: ", image.shape)

        logger.debug("Kernel shape is %s", Kernel.shape)
        logger.debug("Image shape is %s", image.shape)

        if self.mode == "normal":
            return aconv.convolve(image, Kernel)
        elif self.mode == "fft":
            return aconv.convolve_fft(image, Kernel, allow_huge=True)
            # from scipy.fftpack import fft, ifft
            # return aconv.convolve_fft(image, Kernel, allow_huge=True, nan_treatment='interpolate', normalize_kernel=True, fftn=fft, ifftn=ifft)
        elif self.mode == "scipy":
            return signal.convolve(image, Kernel, mode="same", method="auto")
        else:
            raise Exception("Unknown convolve mode: ", self.mode)


#    if pointsource_test:
#        I = np.zeros_like(I)
#        I[I.shape[0]//2, I.shape[1]//2, I.shape[2]//2] = 1
