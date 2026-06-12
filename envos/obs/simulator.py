"""
envos/obs/simulator.py — ObsSimulator, gen_radmc_cmd, format_array, read_radmcdata

Moved from envos/obs.py (P2-B split, move-only, no behaviour changes).
"""
import os
import re
import shutil
import glob
import numpy as np
import multiprocessing
from logging import INFO
import contextlib

from astropy import units as u
import radmc3dPy.image as rmci
import radmc3dPy.analyze as rmca

from .. import nconst as nc
from .. import tools
from ..log import logger
from ..radmc3d import RadmcController


def gen_radmc_cmd(
    mode="image",
    dpc=0,
    incl=0,
    phi=0,
    posang=0,
    npixx=32,
    npixy=32,
    zoomau=[],
    lam=None,
    iline=None,
    vc_kms=None,
    vhw_kms=None,
    nlam=None,
    option="",
):
    position = f"dpc {dpc} incl {incl} phi {phi} posang {posang}"
    camera = f"npixx {npixx} npixy {npixy} "
    camera += f"zoomau {zoomau[0]} {zoomau[1]} {zoomau[2]} {zoomau[3]}"
    if lam is not None:
        freq = f"lambda {lam}"
    elif iline is not None:
        freq = f"iline {iline} widthkms {vhw_kms:g} linenlam {nlam:d}"
        freq += f" vkms {vc_kms:g}" if vc_kms is not None else ""
    cmd = " ".join(["radmc3d", f"{mode}", position, camera, freq, option])
    return cmd


class ObsSimulator:
    """
    This class returns observation data
    Basically, 1 object per 1 observation
    """

    def __init__(self, config=None, radmcdir=None, dpc=None, n_thread=1):
        # P2-A-2: radmc_dir comes from the explicit argument or the config
        # (resolved in init_from_config); no global gpath fallback.
        self.radmc_dir = radmcdir
        self.dpc = dpc
        self.n_thread = n_thread
        self.conv = False
        self.incl = None
        self.phi = None
        self.posang = None

        if config is not None:
            self.init_from_config(config)

    def init_from_config(self, config):
        self.config = config
        if self.radmc_dir is None:
            self.radmc_dir = config.radmc_path
        from ..plot_tools import plot_funcs as _pfun
        _pfun.set_fig_dir(config.fig_path)
        self.dpc = config.dpc
        self.n_thread = config.n_thread
        self.incl = config.incl
        self.phi = config.phi
        self.posang = config.posang
        self.iline = config.iline
        self.molname = config.molname
        self.lineobs_option = config.lineobs_option

        self.set_resolution(
            sizex_au=config.sizex_au or config.size_au,
            sizey_au=config.sizey_au or config.size_au,
            pixsize_au=config.pixsize_au,
            #    npix=config.npix,
            #    npixx=config.npixx,
            #    npixy=config.npixy,
            vfw_kms=config.vfw_kms,
            dv_kms=config.dv_kms,
        )

        self.set_convolver(
            beam_maj_au=config.beam_maj_au,
            beam_min_au=config.beam_min_au,
            vreso_kms=config.vreso_kms,
            beam_pa_deg=config.beam_pa_deg,
            convmode=config.convmode,
        )

    def set_model(self, model, conf=None):
        logger.info("Setting model for observation")
        if conf is None:
            conf = self.config
        radmc = RadmcController(config=conf)
        radmc.clean_radmc_dir()
        radmc.set_model(model)
        radmc.set_temperature(model.Tgas)
        radmc.set_lineobs_inpfiles()

    def set_resolution(
        self,
        sizex_au,
        sizey_au=None,
        pixsize_au=None,
        # npix=None,
        # npixx=None,
        # npixy=None,
        vfw_kms=None,
        dv_kms=None,
    ):
        logger.info("Setting resolution of ObsSimulator")

        if (sizey_au is not None) and (sizex_au != sizey_au):
            logger.warning(
                "Use rectangle viewing mode. "
                "Currently (2020/12), a bug in the mode was fixed. "
                "Please check if your version is latest. "
                "If you do not want to use this, "
                "please set sizex_au = sizey_au."
            )

        self.zoomau_x = [-sizex_au / 2, sizex_au / 2]
        _sizey_au = sizey_au or sizex_au
        self.zoomau_y = [-_sizey_au / 2, _sizey_au / 2]
        Lx = self.zoomau_x[1] - self.zoomau_x[0]
        Ly = self.zoomau_y[1] - self.zoomau_y[0]

        if pixsize_au is not None:
            self.pixsize_au = pixsize_au
            npixx_float = Lx / pixsize_au
            self.npixx = int(round(npixx_float))
            npixy_float = Ly / pixsize_au
            self.npixy = int(round(npixy_float))
            """comment: // or int do not work to convert into int."""
        else:
            logger.error("No pixsize_au.")
            raise Exception
        #    self.npixx = npixx or npix
        #    self.npixy = npixy or npix
        #    self.pixsize_au = Lx / self.npixx

        self.dx_au = sizex_au / self.npixx
        self.dy_au = _sizey_au / self.npixy

        if self.dx_au != self.dy_au:
            raise Exception("dx is not equal to dy. Check your setting again.")

        if sizex_au == 0:
            self.zoomau_x = [-self.pixsize_au / 2, self.pixsize_au / 2]
        if sizey_au == 0:
            self.zoomau_y = [-self.pixsize_au / 2, self.pixsize_au / 2]

        self.vfw_kms = vfw_kms
        self.dv_kms = dv_kms

    def set_convolver(
        self,
        beam_maj_au,
        beam_min_au=None,
        vreso_kms=None,
        beam_pa_deg=0,
        convmode="fft",
    ):
        logger.info("Setting convolution function of ObsSimulator")

        self.conv = True
        self.convolve_config = {
            "beam_maj_au": beam_maj_au,
            "beam_min_au": beam_min_au,
            "beam_pa_deg": beam_pa_deg,
            "vreso_kms": vreso_kms,
        }
        from .convolve import Convolver
        self.convolver = Convolver(
            (self.dx_au, self.dy_au, self.dv_kms),
            **self.convolve_config,
            mode=convmode,
        )

    def observe_cont(
        self, lam_mic=None, freq=None, incl=None, phi=None, posang=None, star=False
    ):
        if lam_mic is None and freq is not None:
            lam_mic = nc.c / freq * 1e4

        incl = incl if incl is not None else self.incl
        phi = phi if phi is not None else self.phi
        posang = posang if posang is not None else self.posang

        logger.info(f"Observing continum with wavelength of {lam_mic} micron")
        zoomau = np.concatenate([self.zoomau_x, self.zoomau_y])

        cmd = gen_radmc_cmd(
            mode="image",
            dpc=self.dpc,
            incl=incl,
            phi=phi,
            posang=posang,
            npixx=self.npixx,
            npixy=self.npixy,
            lam=lam_mic,
            zoomau=zoomau,
            option="noscat" + ("" if star else " nostar"),
        )

        tools.shell(cmd, cwd=self.radmc_dir, error_keyword="ERROR", log_prefix="    ")

        self.data_cont = rmci.readImage(fname=f"{self.radmc_dir}/image.out")
        self.data_cont.dpc = self.dpc
        self.data_cont.freq0 = nc.c / (lam_mic / 1e4)
        odat = read_radmcdata(self.data_cont)

        if self.conv:
            odat.data = self.convolver(odat.data)
            odat.set_obs_resolution(**self.convolve_config)

        return odat

    def observe_line(
        self, iline=None, molname=None, incl=None, phi=None, posang=None, obsdust=False
    ):
        iline = iline if iline is not None else self.iline
        molname = molname if molname is not None else self.molname
        incl = incl if incl is not None else self.incl
        phi = phi if phi is not None else self.phi
        posang = posang if posang is not None else self.posang

        logger.info(f"Observing line with {molname}")
        self.nlam = int(round(self.vfw_kms / self.dv_kms))  # + 1
        mol_path = f"{self.radmc_dir}/molecule_{molname}.inp"
        self.mol = rmca.readMol(fname=mol_path)
        logger.info(
            f"Total cell number is {self.npixx}x{self.npixy}x{self.nlam}"
            + f" = {self.npixx*self.npixy*self.nlam}"
        )

        common_cmd = {
            "mode": "image",
            "dpc": self.dpc,
            "incl": incl,
            "phi": phi,
            "posang": posang,
            "npixx": self.npixx,
            "npixy": self.npixy,
            "zoomau": [*self.zoomau_x, *self.zoomau_y],
            "iline": iline,
            "option": "noscat nostar "
            + self.lineobs_option
            + " ",  # + (" doppcatch " if ,
        }
        if not obsdust:
            common_cmd["option"] += "nodust "

        v_calc_points = np.linspace(-self.vfw_kms / 2, self.vfw_kms / 2, self.nlam)

        if self.n_thread >= 2:
            logger.info("Use OpenMP dividing processes into velocity directions.")
            logger.info("Number of threads is {self.n_thread}.")
            # Load-balance for multiple processing
            n_points = self._divide_nlam_by_threads(self.nlam, self.n_thread)
            vrange_list = self._calc_vrange_list(v_calc_points, n_points)
            v_center = [0.5 * (vmax + vmin) for vmin, vmax in vrange_list]
            v_hwidth = [0.5 * (vmax - vmin) for vmin, vmax in vrange_list]

            logger.info(f"All calc points: {format_array(v_calc_points)}")
            logger.info("Calc points in each thread:")
            _zipped = zip(v_center, v_hwidth, n_points)
            for i, (vc, vhw, ncp) in enumerate(_zipped):
                vax_nthread = np.linspace(vc - vhw, vc + vhw, ncp)
                logger.info(f" -- {i}th thread: {format_array(vax_nthread)}")

            def cmdfunc(i):
                return gen_radmc_cmd(
                    vc_kms=v_center[i],
                    vhw_kms=v_hwidth[i],
                    nlam=n_points[i],
                    **common_cmd,
                )

            args = [(i, cmdfunc(i)) for i in range(self.n_thread)]

            logger.info(
                "Follwing messages come from RADMC-3D running at representative #1 core"
            )
            logger.info("***** RADMC-3D message start *****")
            with multiprocessing.Pool(self.n_thread) as pool:
                results = pool.starmap(self._subcalc, args)
            logger.info("***** RADMC-3D message end *****")

            self._check_multiple_returns(results)
            self.data = self._combine_multiple_returns(results)

        else:
            logger.info("Not use OpenMP.")
            cmd = gen_radmc_cmd(vhw_kms=self.vfw_kms / 2, nlam=self.nlam, **common_cmd)
            logger.info("***** RADMC-3D message start *****")
            tools.shell(cmd, cwd=self.radmc_dir, error_keyword="ERROR")
            logger.info("***** RADMC-3D message end *****")
            self.data = rmci.readImage(fname=f"{self.radmc_dir}/image.out")

        if np.max(self.data.image) == 0:
            print(vars(self.data))
            logger.warning("Zero image !")
            raise Exception

        self.data.dpc = self.dpc
        self.data.freq0 = self.mol.freq[iline - 1]
        odat = read_radmcdata(self.data)
        odat.set_sobs_info(
            iline=iline, molname=molname, incl=incl, phi=phi, posang=posang
        )

        if self.conv:
            odat.set_I(self.convolver(odat.Ippv))
            odat.set_obs_resolution(**self.convolve_config)

        return odat

    @staticmethod
    def _divide_nlam_by_threads(nlam, nthr):
        divided_nlam_list = [nlam // nthr] * nthr
        remainder = nlam % nthr
        for i in range(remainder):
            i_distribute = (nthr - 1 - i // 2) if i % 2 else i // 2
            divided_nlam_list[i_distribute] += 1
        return divided_nlam_list

    @staticmethod
    def _calc_vrange_list(whole_vrange, divided_nlam_list):
        vrange_list = []
        sum_nlam = 0
        for nlam in divided_nlam_list:
            i_start = sum_nlam
            i_end = sum_nlam + nlam - 1
            vrange = (whole_vrange[i_start], whole_vrange[i_end])
            vrange_list.append(vrange)
            sum_nlam += nlam
        return np.array(vrange_list)

    def _subcalc(self, p, cmd):
        dn = f"proc{p:d}"
        dpath_sub = f"{self.radmc_dir}/{dn}"
        os.makedirs(dpath_sub, exist_ok=True)
        for f in glob.glob(f"{self.radmc_dir}/*"):
            if re.search(r".*\.(inp|dat)$", f):
                shutil.copy2(f, f"{dpath_sub}/")

        log = logger.isEnabledFor(INFO) and p == 0
        tools.shell(
            cmd,
            cwd=dpath_sub,
            log=log,
            error_keyword="ERROR",
            log_prefix="    ",
        )
        with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
            fname = f"{dpath_sub}/image.out"
            return rmci.readImage(fname=fname)

    def _check_multiple_returns(self, return_list):
        for i, r in enumerate(return_list):
            logger.debug(f"The {i}th return")
            for k, v in r.__dict__.items():
                if isinstance(v, (np.ndarray)):
                    vrange = f"[{np.min(v)}, {np.max(v)}]"
                    logger.debug(f"{k}: shape {v.shape}, range {vrange}")
                else:
                    logger.debug(f"{k}: {v}")

    def _combine_multiple_returns(self, return_list):
        data = return_list[0]
        for ret in return_list[1:]:
            data.image = np.append(data.image, ret.image, axis=-1)
            data.imageJyppix = np.append(data.imageJyppix, ret.imageJyppix, axis=-1)
            data.freq = np.append(data.freq, ret.freq, axis=-1)
            data.wav = np.append(data.wav, ret.wav, axis=-1)
            data.nfreq += ret.nfreq
            data.nwav += ret.nwav
        return data

    def output_fits(self, filepath):
        fp_fitsdata = filepath
        if os.path.exists(fp_fitsdata):
            logger.info(f"remove old fits file: {fp_fitsdata}")
            os.remove(fp_fitsdata)
        self.data.writeFits(fname=fp_fitsdata, dpc=self.dpc)
        logger.info(f"Saved fits file: {fp_fitsdata}")


def format_array(array):
    if len(array) >= 2:
        delta = abs(array[1] - array[0])
    else:
        delta = 0
    msg = f"[{min(array):.2f}:{max(array):.2f}] "
    msg += f"with delta = {delta:.4g} and N = {len(array)}"
    return msg


def read_radmcdata(data):
    from .data import Cube, Image
    if len(data.image.shape) == 2:
        Ipp = data.imageJyppix / data.dpc**2
        dtype = "image"
    elif len(data.image.shape) == 3 and data.image.shape[2] == 1:
        Ipp = data.imageJyppix[:, :, 0] / data.dpc**2
        dtype = "image"
    elif len(data.image.shape) == 3:
        Ippv = data.imageJyppix / data.dpc**2  # .transpose(2, 1, 0)
        dtype = "cube"
    Iunit = u.Jy / u.pix  # "Jy/pixel"

    dpc = data.dpc  # or 100
    # Nx = data.nx
    # Ny = data.ny
    # Nv = data.nfreq
    # Nf = data.nfreq
    xau = data.x / nc.au
    yau = data.y / nc.au
    freq = data.freq
    freq0 = data.freq0
    vkms = tools.freq_to_vkms_array(freq, freq0)
    # dx = data.sizepix_x / nc.au
    # dy = data.sizepix_y / nc.au
    # dv = vkms[1] - vkms[0] if len(vkms) > 1 else 0
    # Check
    # if (data.nx != len(xau)) or (data.ny != len(yau)) or (data.nfreq != len(vkms) or \
    #   (dx != (xau[1]-xau[0])) or (dy != (yau[1]-yau[0])) or (dv != (vkms[1] - vkms[0])):
    #    sys.exit("Something in this radmcdata will be wrong.")

    if dtype == "cube":
        obj = Cube(Ippv, xau=xau, yau=yau, vkms=vkms, Iunit=Iunit, dpc=dpc, freq0=freq0)

    elif dtype == "image":
        obj = Image(Ipp, xau=xau, yau=yau, Iunit=Iunit, dpc=dpc, freq0=freq0)

    return obj
