import numpy as np
from scipy import interpolate, integrate
from . import log

def calc_column_density(model, direction, double_interp=True):
    if direction == "r":
        return integrate.cumulative_trapezoid(model.rhogas, model.rr, axis=0, initial=0)
    elif direction == "theta":
        return (
            integrate.cumulative_trapezoid(model.rhogas, model.tt, axis=1, initial=0)
            * model.rr
        )
    elif direction == "z":
        if double_interp:
            # x = np.linspace(0, np.max(model.rc_ax), 10000)
            x = np.linspace(0, np.max(model.rc_ax) ** (1 / 1.5), 3000) ** (1.5)
            # rho ~ r^-1.5
            # rho/r^1.5 ~ const

            z = x
            xx, zz = np.meshgrid(x, z, indexing="ij")

            newgrid = np.stack(
                [np.sqrt(xx**2 + zz**2), np.arctan2(xx, zz)], axis=-1
            )
            rho_xz = interpolate.interpn(
                (model.rc_ax, model.tc_ax),
                model.rhogas[..., 0],
                newgrid,
                bounds_error=False,
                fill_value=None,
            )

            colz = -integrate.cumulative_trapezoid(
                rho_xz[:, ::-1], zz[:, ::-1], axis=1, initial=0
            )[:, ::-1]

            col = interpolate.interpn(
                (x, z),
                colz,
                np.stack([model.R, model.z], axis=-1),
                bounds_error=False,
                fill_value=None,
            )

            return col

        else:
            "Samller grid noise in the interpolation"
            log.logger.info(
                "Computing column density in z-direction, which could take time..."
            )
            rho_interp = interpolate.RegularGridInterpolator(
                (model.rc_ax, model.tc_ax),
                model.rhogas[..., 0],
                bounds_error=False,
                fill_value=None,
            )

            zlim = np.max(model.z) * 1.1
        return v_column_density_z(model.R, model.z, model, rho_interp, zlim).astype(
            float
        )
    else:
        raise ValueError(f"Wrong input direction: {direction}")


def test_column_density(model):
    import matplotlib.pyplot as plt

    test_colr = 0
    test_colt = 0
    test_colz = 1

    # 1. density integration along r-direction
    if test_colr:
        colr = integrate.cumulative_trapezoid(model.rhogas, model.rr, axis=0, initial=0)
        plt.pcolormesh(
            model.R[..., 0], model.z[..., 0], colr[..., 0], shading="nearest"
        )
        plt.show()

    # 2. density integration along theta-direction
    if test_colt:
        colt = (
            integrate.cumulative_trapezoid(model.rhogas, model.tt, axis=1, initial=0)
            * model.rr
        )
        plt.pcolormesh(
            model.R[..., 0], model.z[..., 0], colt[..., 0], shading="nearest"
        )
        plt.show()

    # 3. density integration along z-direction
    if test_colz:
        rho_interp = interpolate.RegularGridInterpolator(
            (model.rc_ax, model.tc_ax),
            model.rhogas[..., 0],
            bounds_error=False,
            fill_value=None,
        )

        zlim = np.max(model.z) * 1.1
        colz = v_column_density_z(model.R, model.z, model, rho_interp, zlim).astype(
            float
        )
        print(colz.shape)
        plt.pcolormesh(
            model.R[..., 0], model.z[..., 0], colz[..., 0], shading="nearest"
        )
        plt.show()


def column_density_z(R, z, model, interp_func, zlim):
    _R = R
    # C-55 fix: guard against step=0 when z=0
    step = max(0.1 * z, zlim * 1e-4)
    _z_ax = np.arange(z, zlim, step)
    r = np.sqrt(_R**2 + _z_ax**2)
    t = np.arctan2(_R, _z_ax)
    points = np.stack([r, t], axis=-1)
    rho = interp_func(points)
    colz = integrate.simpson(rho, _z_ax)
    return colz


v_column_density_z = np.frompyfunc(column_density_z, 5, 1)

if __name__ == "__main__":
    import envos
    config = envos.Config(
        run_dir="./run",
        n_thread=10,
        nphot=1e5,
        rau_in=10,
        rau_out=1000,
        dr_to_r=0.01,
        aspect_ratio=1,
        inenv="UCM",
        CR_au=100,
        Ms_Msun=0.3,
        T=10,
        cavangle_deg=45,
        f_dg=0.01,
        opac="MRN20",
        Lstar_Lsun=1.0,
    )
    mg = envos.ModelGenerator(config)
    mg.calc_kinematic_structure()
    mg.calc_thermal_structure()
    model = mg.get_model()

    test_column_density(model)
