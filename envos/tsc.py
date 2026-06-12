import os
import numpy as np
import pandas as pd
from scipy import integrate, interpolate, optimize
from dataclasses import dataclass

from .log import logger
from . import nconst as nc

FILENAME = "tscsol.pkl"


@dataclass
class TscData:
    x: np.ndarray
    alpha_0: np.ndarray
    alpha_M: np.ndarray
    alpha_Q: np.ndarray
    V_0: np.ndarray
    V_M: np.ndarray
    V_Q: np.ndarray
    W_Q: np.ndarray
    m_0: np.ndarray
    da0dy: np.ndarray
    dV0dy: np.ndarray
    Delta_Q: float

    def variables(self):
        return (
            self.alpha_0,
            self.alpha_M,
            self.alpha_Q,
            self.V_0,
            self.V_M,
            self.V_Q,
            self.W_Q,
            self.m_0,
            self.da0dy,
            self.dV0dy,
        )

    def make_meshgrid_data(self, theta):
        xx, tt = np.meshgrid(self.x, theta, indexing="ij")
        vvlist = [np.meshgrid(v, theta, indexing="ij")[0] for v in self.variables()]
        tscdata = TscData(xx, *vvlist, self.Delta_Q)
        tscdata.xx = xx
        tscdata.tt = tt
        return tscdata

    def make_interpolated_data(self, x):
        vlist = [make_function(self.x, v, fill_value=0)(x) for v in self.variables()]
        return TscData(x, *vlist, self.Delta_Q)


class TscSolver:
    def __init__(self, tau=0.001, n=500, eps=1e-8, plot=False, fig_dir="."):
        self.tau = tau
        self.plot = plot
        self.fig_dir = fig_dir
        self.eps = eps
        self.xout = 1.0 + np.geomspace(1 / self.tau - 1, eps, n)
        self.xin = 1 / (1.0 + np.geomspace(eps, self.tau ** (-2) - 1.0, n))
        self.x = np.hstack((self.xout, self.xin))

    def solve(self, search_K=False):
        self.solve_0thorder()
        self.solve_Qualdrapolar(search_K=search_K)
        self.solve_Monopolar()
        self.print_result_constants()

    def solve_Spherically_Symmetric_Solution(self):
        # Solve Spherically symmetric solutions
        # unpertubed rotating equilibria
        # see Eq (11) in TSC with Psi = Psi(ksi)
        def f(x, y):  # y = [ dPhi, Phi]
            dPhi = y[1]
            ddPhi = (
                -2 * y[1] / x - 2 * np.exp(y[0]) / x**2 + 2 * (1 + x**2) / x**2
            )
            return dPhi, ddPhi

        logger.debug("Start solve_Spherically_Symmetric_Solution")
        ksi = np.geomspace(0.01, 100, 200)
        sol = integrate.solve_ivp(
            f,
            (ksi[0], ksi[-1]),
            (0.5 * ksi[0], 0.25 * ksi[0] ** 2),
            t_eval=ksi,
            method="BDF",
        )
        rho_eq = np.exp(sol.y[0]) * ksi ** (-2)
        M_eq = integrate.cumulative_trapezoid(np.exp(sol.y[0]), ksi, initial=0) + ksi[0]
        ksi_eq = ksi
        return ksi_eq, rho_eq, M_eq

    def solve_0thorder(self):
        logger.debug("Start solve_0thorder")

        def func0th(y, v):
            V_0, al_0 = v
            dV_0 = (
                (al_0 * (y - V_0) - 2 / y) * (y - V_0) / ((y - V_0) ** 2 - 1)
            )  # if x < 1 else 0
            dal_0 = (
                al_0 * (al_0 - 2 / y * (y - V_0)) * (y - V_0) / ((y - V_0) ** 2 - 1)
            )  # if x < 1 else -4/x**3
            return np.array([dV_0, dal_0])

        def jacfunc(y, v):
            V0, al0 = v
            s = y - V0
            f1y1 = -(2 / y) * (s**2 - al0 * y * s + 1) / (y**2 - 1) ** 2
            f1y2 = s**2 / (s**2 - 1)
            f2y1 = 2 * al0 * s**2 * (al0 - 2 + 2 * V0 / y) / (s**2 - 1) ** 2 + (
                4 * al0 - 4 * al0 * V0 / y - al0**2
            ) / (s**2 - 1)
            f2y2 = s * (2 * al0 - 2 + 2 * V0 / y) / (s**2 - 1)
            return np.array([[f1y1, f1y2], [f2y1, f2y2]])

        eps = 2 * self.eps
        sol = integrate.solve_ivp(
            func0th,
            (self.xin[0], self.xin[-1]),
            [-eps, 2 + 2 * eps],
            t_eval=self.xin,
            jac=jacfunc,
            method="BDF",
            vectorized=True,
            rtol=self.eps,
            atol=1e-10,
        )

        if not sol.success:
            logger.error(sol)
            raise Exception("Failed to solve TSC 0th-order equations.")

        if len(self.xin) != len(sol.t):
            raise Exception(
                f"x finally used in the solution is not equal to the input x. \n input x: {self.xin.shape}, used x: {sol.t.shape}"
            )

        self.V_0 = np.hstack((np.full_like(self.xout, -1e-50), sol.y[0]))
        self.al_0 = np.hstack((2 / self.xout**2, sol.y[1]))

        self.m_0 = self.x**2 * self.al_0 * (self.x - self.V_0)
        self.f_al0 = make_function(self.x, self.al_0)
        self.f_V0 = make_function(self.x, self.V_0)
        self.da0dy = self.f_dal0dy(self.x)
        self.dV0dy = self.f_dV0dy(self.x)
        logger.info(f"m0 is {self.m_0[-1]}")

    def f_m0(self, y):
        return y**2 * self.f_al0(y) * (y - self.f_V0(y))

    def f_dal0dy(self, y):
        al_0 = self.f_al0(y)
        V_0 = self.f_V0(y)
        return np.where(
            (y - V_0) ** 2 != 1,
            al_0 * (al_0 - 2 / y * (y - V_0)) * (y - V_0) / ((y - V_0) ** 2 - 1),
            -2 / y**2,
        )

    def f_dV0dy(self, y):
        al_0 = self.f_al0(y)
        V_0 = self.f_V0(y)
        return np.where(
            (y - V_0) ** 2 != 1,
            (al_0 * (y - V_0) - 2 / y) * (y - V_0) / ((y - V_0) ** 2 - 1),
            1,
        )

    def solve_Qualdrapolar(self, search_K=True):
        def f_QuadraPolar(x, vals):  ## Eq 63
            al, V, W, Q, P = vals
            al_0 = self.f_al0(x)  # if x < 1 else 2/x**2
            V_0 = self.f_V0(x)  # if x < 1 else 0
            dal0dx = self.f_dal0dy(x)  # if x < 1 else -4/x**3
            dV0dx = self.f_dV0dy(x)  # if x < 1 else 0
            Psi = -(x**2) / 3 - Q / x**3 - x**2 * P
            dPsidx = -2 / 3 * x + 3 * Q / x**4 - 2 * x * P
            m = x**2 * al_0 * (x - V_0)
            A = (
                al / x**2 * (2 * x * V_0 + x**2 * dV0dx)
                + V / x**2 * (x**2 * dal0dx + 2 * x * al_0)
                - 6 * al_0 * W / x
            )
            B = (
                -al / al_0**2 * dal0dx
                + (2 + dV0dx) * V
                + (dPsidx + 2 / 3 * (m / 2) ** 4 / x**3)
            )
            dal = 1 / ((x - V_0) ** 2 - 1) * ((x - V_0) * A + al_0 * B)
            dV = 1 / ((x - V_0) ** 2 - 1) * ((x - V_0) * B + 1 / al_0 * A)
            dW = (
                1
                / x
                / (x - V_0)
                * ((2 * x + V_0) * W + al / al_0 + (Psi + (m / 2) ** 4 / (3 * x**2)))
            )
            dQ = 0.2 * x**4 * al
            dP = -0.2 * al / x
            return np.array([dal, dV, dW, dQ, dP])

        def f_QuadraPolar_out(x, vals):
            al, V, W, Q, P = vals
            fac = 1 / (x**2 - 1)
            dal = -12 / x**2 * W
            dal += 2 / x**2 * (x * al + 2 * V + 3 * Q / x**4 - 2 * x * P)
            dal *= fac
            dV = x * (x * al + 2 * V + 3 * Q / x**4 - 2 * x * P)
            dV += -6 / x * W
            dV *= fac
            dW = 2 / x * W + 0.5 * al - Q / x**5 - P
            dQ = 0.2 * x**4 * al
            dP = -0.2 * al / x
            return dal, dV, dW, dQ, dP

        def fjac_QuadraPolar_out(x, vals):
            fac = 1 / (x**2 - 1)
            df1 = fac * np.array([2 / x, 4 / x**2, -12 / x**2, 6 / x**6, -4 / x])
            df2 = fac * np.array([x**2, 2 * x, -6 / x, 3 / x**3, -2 / x**2])
            df3 = np.array([0.5, 0, 2 / x, -1 / x**5, -1])
            df4 = np.array([0.2 * x**4, 0, 0, 0, 0])
            df5 = np.array([-0.2 / x, 0, 0, 0, 0])
            return np.array([df1, df2, df3, df4, df5])

        def sol_Q(K):
            opt = {
                "method": {
                    0: "RK45",
                    1: "BDF",
                    2: "Radau",
                    3: "DOP853",
                    4: "LSODA",
                }[1],
                "rtol": self.eps,
                "atol": 1e-10,
            }
            x0 = self.x[0]
            y0 = (
                -2 / 7 * K * x0 ** (-7),
                -K / 2 * x0 ** (-4),
                K / 6 * x0 ** (-4),
                K * (1 + x0 ** (-2) / 35),
                -2 / 245 * K * x0 ** (-7),
            )
            sol_o = integrate.solve_ivp(
                f_QuadraPolar_out,
                (self.xout[0], self.xout[-1]),
                y0,
                t_eval=self.xout,
                jac=fjac_QuadraPolar_out,
                **opt,
            )
            # sol_o = integrate.solve_ivp(f_QuadraPolar_out, (self.xout[0], self.xout[-1]), y0, t_eval=self.xout, **opt)
            Delta_Q = -sol_o.y[1][-1] / 4
            alin = sol_o.y[0][-1] + 2 * Delta_Q
            Vin = sol_o.y[1][-1] + 2 * Delta_Q
            Win = sol_o.y[2][-1]
            Qin = sol_o.y[3][-1]
            Pin = sol_o.y[4][-1]
            cond_outerEW = (
                sol_o.y[0][-1]
                + 2 * sol_o.y[1][-1]
                - 6 * sol_o.y[2][-1]
                + 3 * sol_o.y[3][-1]
                - 2 * sol_o.y[4][-1]
            )
            sol_i = integrate.solve_ivp(
                f_QuadraPolar,
                (self.xin[0], self.xin[-1]),
                (alin, Vin, Win, Qin, Pin),
                t_eval=self.xin,
                **opt,
            )
            soly = np.hstack((sol_o.y, sol_i.y))
            solx = np.hstack((sol_o.t, sol_i.t))
            return solx, soly, Delta_Q

        j = 0

        def loop_K(K):
            nonlocal j
            j += 1
            solx, soly, Delta_Q = sol_Q(K)
            if self.plot:
                import matplotlib.pyplot as plt

                alpha = soly[0] - Delta_Q * solx * self.f_dal0dy(solx)
                plt.plot(np.log10(solx), np.log10((-alpha).clip(1e-100)))
                plt.xlim(-3, 1)
                plt.ylim(-5, 7)
                logger.info(f"{j}: K={K}")
                plt.savefig(f"{self.fig_dir}/alpha_{j}.pdf")
                logger.info("saved figure")
                plt.clf()
            return soly[3][-1] - 0.2 * soly[0][-1] * solx[-1] ** 5

        K0 = -0.001489605859125321
        if search_K:
            K = optimize.newton(loop_K, x0=K0, rtol=1e-8)
        else:
            K = K0
        solx, soly, Delta_Q = sol_Q(K)
        self.K = K
        self.Delta_Q = Delta_Q
        self.al_Q = soly[0]
        self.V_Q = soly[1]
        self.W_Q = soly[2]

    def solve_Monopolar(self):
        def f_MonoPolar(x, vals):
            al, V, M = vals
            al_0 = self.f_al0(x) if x < 1 else 2 / x**2
            V_0 = self.f_V0(x) if x < 1 else 0
            dal0dx = self.f_dal0dy(x) if x < 1 else -4 / x**3
            dV0dx = self.f_dV0dy(x) if x < 1 else 0
            dPhidx = x / 6 + M / x**2
            m = x**2 * al_0 * (x - V_0)
            A = al / x**2 * (2 * x * V_0 + x**2 * dV0dx) + V / x**2 * (
                2 * x * al_0 + x**2 * dal0dx
            )
            B = (
                -al / al_0**2 * dal0dx
                + (2 + dV0dx) * V
                + (dPhidx - 2 / 3 * (m / 2) ** 4 / x**3)
            )
            dal = ((x - V_0) * A + al_0 * B) / ((x - V_0) ** 2 - 1)
            dV = ((x - V_0) * B + A / al_0) / ((x - V_0) ** 2 - 1)
            dM = (al - 0.5) * x**2
            return dal, dV, dM

        y0 = (1 / 2, 0, 0)
        solM = integrate.solve_ivp(
            f_MonoPolar,
            (self.x[0], self.x[-1]),
            y0,
            t_eval=self.x,
            method="BDF",
            rtol=1e-8,
        )
        self.al_M = solM.y[0]
        self.V_M = solM.y[1]
        self.Ms = -integrate.simpson(
            self.x[self.x < 1] ** 2 * (self.al_M[self.x < 1] - 1 / 2),
            x=self.x[self.x < 1],
        )

    # def save_table(self, filename="tscsol.dat", path=None):
    #    soldict = asdict(self.get_solution())
    #    path = path or os.path.join(rc.storage_dir, filename)
    #    vals = np.array(list(soldict.values())).T
    #    np.savetxt(path, vals, header=" ".join(soldict.keys())  )
    def print_result_constants(self):
        logger.info(f"result: K={self.K}, ΔQ={self.Delta_Q}, m*={self.Ms}")

    def save_table(self, filename=FILENAME, storage_dir=None, path=None):
        if path is None:
            if storage_dir is None:
                raise ValueError("save_table requires storage_dir or path")
            path = os.path.join(storage_dir, filename)
        pd.to_pickle(self.get_solution(), path)

    def get_solution(self):
        sol = TscData(
            self.x,
            self.al_0,
            self.al_M,
            self.al_Q,
            self.V_0,
            self.V_M,
            self.V_Q,
            self.W_Q,
            self.m_0,
            self.da0dy,
            self.dV0dy,
            self.Delta_Q,
        )
        return sol


# def read_table(filename="tscsol.dat", path=None):
#    path = path or os.path.join(rc.storage_dir, filename)
#    vals = np.loadtxt(path)
#    return TscData(*vals.T)


def calc_rho(theta, tau, Omega, DeltaQ, al0, alM, alQ, da0dy):
    P2 = 1 - 3 / 2 * np.sin(theta) ** 2
    # _rho = al0 + tau**2 * (alM + alQ*P2 - DeltaQ * P2 * da0dy)
    _rho = al0 + tau**2 * (alM + alQ * P2)
    # _rho = al0 #+ tau**2 * (alM + alQ*P2 - DeltaQ * P2 * da0dy)
    return Omega**2 / (4 * nc.pi * nc.G * tau**2) * _rho


def calc_velocity(x, theta, tau, cs, DeltaQ, V0, VM, VQ, WQ, m0, dV0dy):
    P2 = 1 - 3 / 2 * np.sin(theta) ** 2
    _vr = V0 + tau**2 * (VM + VQ * P2)
    _vth = tau**2 * WQ * (-3 * np.sin(theta) * np.cos(theta))
    _vph = tau / (4 * x) * m0**2 * np.sin(theta)
    return cs * _vr, cs * _vth, cs * _vph


def make_function(x, y, extrapolate=False, fill_value=None):
    fill_value = "extrapolate" if extrapolate else fill_value
    if (not extrapolate) or (fill_value is not None):
        bounds_error = False
    f = interpolate.interp1d(x, y, fill_value=fill_value, bounds_error=bounds_error)
    return f



"""
Wrapper

"""


def get_tsc(r, theta, t, cs, Omega, mode="read", filename=FILENAME, storage_dir=None):
    if mode == "read":
        _sol = read_table(filename=filename, storage_dir=storage_dir)
        if _sol is None:
            logger.info(
                "Failed to load the table of TSC solution (probably just due to missing the file), and so solve TSC equations. After solving, the solution will be saved in the storage directory. From the next time, the table will be loaded to save computational costs."
            )
            mode = "solve"

    if mode == "solve":
        tscs = TscSolver()
        tscs.solve()
        if storage_dir is not None:
            tscs.save_table(filename=filename, storage_dir=storage_dir)
        _sol = tscs.get_solution()

    x = r / cs / t
    tau = Omega * t
    sol_interp = _sol.make_interpolated_data(x)
    mgdata = sol_interp.make_meshgrid_data(theta)
    rho = calc_rho(
        mgdata.tt,
        tau,
        Omega,
        mgdata.Delta_Q,
        mgdata.alpha_0,
        mgdata.alpha_M,
        mgdata.alpha_Q,
        mgdata.da0dy,
    )
    vr, vt, vp = calc_velocity(
        mgdata.xx,
        mgdata.tt,
        tau,
        cs,
        mgdata.Delta_Q,
        mgdata.V_0,
        mgdata.V_M,
        mgdata.V_Q,
        mgdata.W_Q,
        mgdata.m_0,
        mgdata.dV0dy,
    )

    # return {"vars":(rho, vr, vt, vp), "Delta":mgdata.Delta_Q}
    return {"rho": rho, "vr": vr, "vt": vt, "vp": vp, "Delta": mgdata.Delta_Q}


def save_table(filename=FILENAME, storage_dir=None, tau=0.01, search_K=False, **kwargs):
    fig_dir = kwargs.pop("fig_dir", ".")
    tscs = TscSolver(tau=tau, fig_dir=fig_dir, **kwargs)
    tscs.solve(search_K=search_K)
    tscs.save_table(filename=filename, storage_dir=storage_dir)


def read_table(filename=FILENAME, storage_dir=None, path=None):
    try:
        if path is None:
            if storage_dir is None:
                return None
            path = os.path.join(storage_dir, filename)
        return pd.read_pickle(path)
    except Exception:
        return None


def _main(storage_dir=".", fig_dir="."):
    tsc.save_table(
        storage_dir=storage_dir, n=300, tau=0.001, eps=1e-8, search_K=True, plot=True,
        fig_dir=fig_dir,
    )
    sol = tsc.read_table(storage_dir=storage_dir)

    import matplotlib.pyplot as plt

    plt.xscale("log")
    plt.yscale("log")
    plt.xlim(1e-3, 1e1)
    plt.ylim(1e-5, 1e7)
    l = plt.plot(
        sol.x,
        np.array([sol.alpha_0, sol.alpha_M, sol.alpha_Q, -sol.alpha_Q]).T,
    )
    plt.savefig(f"{fig_dir}/tscfig3.pdf")
    [_l.remove() for _l in l]
    plt.xlim(1e-3, 1e0)
    plt.ylim(1e-4, 1e3)
    l = plt.plot(sol.x, np.array([-sol.V_0, sol.V_M]).T)
    plt.savefig(f"{fig_dir}/tscfig4a.pdf")
    [_l.remove() for _l in l]
    plt.xlim(1e-3, 1e1)
    plt.ylim(1e-8, 1e3)
    l = plt.plot(sol.x, np.array([-sol.V_Q, sol.V_Q, -sol.W_Q]).T)
    plt.savefig(f"{fig_dir}/tscfig4b.pdf")


if __name__ == "__main__":
    from envos import tsc

    _main()
