from py_wake import np
from abc import ABC, abstractmethod
from numpy import newaxis as na
from py_wake.utils.gradients import cabs


class SuperpositionModel(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, value_jxxx):
        """Calculate the sum of jxxx

        This method must be overridden by subclass

        Parameters
        ----------

        deficit_jxxx : array_like
            deficit caused by source turbines(j) on xxx (xxx optionally includes
            destination turbine/site, wind directions, wind speeds

        Returns
        -------
        sum_xxx : array_like
            sum for xxx (see above)
        """

    def superpose_deficit(self, deficit_jxxx, **kwargs):
        return self(deficit_jxxx, **kwargs)


class AddedTurbulenceSuperpositionModel():
    def calc_effective_TI(self, TI_xxx, add_turb_jxxx):
        """Calculate effective turbulence intensity

        Parameters
        ----------
        TI_xxx : array_like
            Local turbulence intensity. xxx optionally includes destination turbine/site, wind directions, wind speeds
        add_turb_jxxx : array_like
            added turbulence caused by source turbines(j) on xxx (see above)

        Returns
        -------
        TI_eff_xxx : array_like
            Effective turbulence intensity xxx (see TI_xxx)
        """
        return TI_xxx + self(add_turb_jxxx)


class SquaredSum(SuperpositionModel, AddedTurbulenceSuperpositionModel):
    def __call__(self, value_jxxx):
        assert not np.any(value_jxxx < 0), "SquaredSum only works for deficit - not speedups"
        return np.sqrt(np.sum(np.power(value_jxxx, 2), 0))


class LinearSum(SuperpositionModel, AddedTurbulenceSuperpositionModel):
    def __call__(self, value_jxxx):
        return np.sum(value_jxxx, 0)


class MaxSum(SuperpositionModel, AddedTurbulenceSuperpositionModel):
    def __call__(self, value_jxxx):
        return np.max(value_jxxx, 0)


class SqrMaxSum(AddedTurbulenceSuperpositionModel):
    def calc_effective_TI(self, TI_xxx, add_turb_jxxx):
        return np.sqrt(TI_xxx**2 + np.max(add_turb_jxxx, 0)**2)


class WeightedSum(SuperpositionModel):
    """
    Implemented according to the paper by:
    Haohua Zong and Fernando Porté-Agel
    A momentum-conserving wake superposition method for wind farm power prediction
    J. Fluid Mech. (2020), vol. 889, A8; doi:10.1017/jfm.2020.77
    """

    def __init__(self, delta=0.01, max_err=1e-3, max_iter=5):
        # minimum deficit (as fraction of free-stream) to invoke weighted summation
        self.delta = delta
        # convergence limit for computing global convection velocity
        self.max_err = max_err
        # maximum number of iterations used in computing weights
        self.max_iter = max_iter

    def __call__(self, centerline_deficit_jxxx, WS_xxx,
                 convection_velocity_jxxx,
                 sigma_sqr_jxxx, cw_jxxx, hcw_jxxx, dh_jxxx):

        Ws = WS_xxx + np.zeros(centerline_deficit_jxxx.shape[1:])

        usc = centerline_deficit_jxxx
        uc = convection_velocity_jxxx
        sigma_sqr = sigma_sqr_jxxx
        cw = cw_jxxx
        hcw = hcw_jxxx * np.ones_like(usc)
        dh = dh_jxxx * np.ones_like(usc)
        Us = np.zeros_like(Ws)
        # Determine non-centreline deficit ratio
        # Local deficit
        us = usc * np.exp(-1 / (2 * sigma_sqr) * cw**2)

        # Set lower deficit limit below which deficits are linearly added
        us_lim = max(self.delta, 1e-30) * Ws[na]
        # Get indices
        Il = us >= us_lim

        # Only start weighting computation if at least two deficits need to be combined
        # Computations are performed where velocities exceed the specified limit (us_lim).
        # The more complex indexing leads to more complex code.
        if Il.any():
            # Get indices where deficits need to be combined
            Ilx = Il.any(axis=0)
            # Total cross-wind integrated deficit
            us_int = np.zeros_like(us)
            us_int[Il] = usc[Il] * 2 * np.pi * sigma_sqr[Il]
            # initialize combined quanatities
            Uc = Ws.copy()
            Uc_star = Ws.copy()
            Us = np.zeros_like(Ws)
            Us_int = np.zeros_like(Ws)

            # Initialize
            count = 0
            Uc_star = 10 * Uc
            if np.iscomplexobj(Ws) or np.iscomplexobj(uc):
                Uc_star = Uc_star.astype(np.complex128)
            tmp1, tmp2 = np.zeros_like(us), np.zeros_like(us)
            tmpUS, tmpUSint = np.zeros_like(us), np.zeros_like(us)
            sum1, sum2, ucn = np.zeros_like(Us), np.zeros_like(Us), np.ones_like(uc)
            # sum1 precomputed part
            sum1_pre = usc[Il]**2 * np.pi * sigma_sqr[Il]
            n_wt = us.shape[0]
            # Iterate until combined convection velocity converges
            while (np.max(cabs((Uc[Ilx] - Uc_star[Ilx]) / Uc_star[Ilx])) > self.max_err) and (count < self.max_iter):
                # Initialize combined convection velocity
                if count == 0:
                    # Take maximum across all turbines to initilize global convection velocity
                    Uc = np.max(np.where(Il, uc, 0), 0)
                else:
                    Uc = Uc_star.copy()
                # Initialize and avoid division by zero
                ucn[:] = 1.
                Inz = Uc != 0
                ucn[:, Inz] = uc[:, Inz] / Uc[Inz]

                # Combined local deficit
                # dummy matrix to keep original matrix shape
                tmpUS[:] = 0
                tmpUS[Il] = ucn[Il] * us[Il]
                Us = np.sum(tmpUS, axis=0)
                # Combined deficit integrated to infinity in cross-wind direction
                tmpUSint[:] = 0
                tmpUSint[Il] = ucn[Il] * us_int[Il]
                Us_int = np.sum(tmpUSint, axis=0)

                # First sum of momentum deficit
                # sum_i^N (uc_i u_i)^2
                sum1[:], tmp1[:] = .0, 0.
                tmp1[Il] = ucn[Il]**2 * sum1_pre
                sum1 = np.sum(tmp1, axis=0)
                # Second sum which represents the cross terms
                # 2 sum i>j (uc_i u_i) (uc_j u_j)
                if n_wt > 1:
                    # Initialize
                    sum2[:] = .0
                    for j in range(n_wt - 1):
                        # Only cross with larger indices
                        k = np.arange(j + 1, n_wt, dtype=int)
                        # Find indices where deficits need to be combined
                        Ilxx = Il[j][na] & Il[k]
                        if Ilxx.any():
                            # To keep the shape, arrays are repeated and a dummy initilized
                            # Instead of a dummy one could use a loop, but this seemed faster
                            tmp2 = np.zeros(((len(k),) + sigma_sqr.shape[1:]), dtype=sigma_sqr.dtype)
                            s1, s2 = np.repeat(sigma_sqr[j][na], len(k), axis=0)[Ilxx], sigma_sqr[k][Ilxx]
                            w2w_hcw = cabs(hcw[j][na] - hcw[k])[Ilxx]
                            w2w_dh = cabs(dh[j][na] - dh[k])[Ilxx]
                            cross_sigma_jk = 2 * np.exp(-(w2w_hcw**2 + w2w_dh**2) /
                                                        (2 * (s1 + s2))) * np.pi * s1 * s2 / (s1 + s2)
                            tmp2[Ilxx] = 2 * np.repeat((ucn[j] * usc[j])[na], len(k), axis=0)[Ilxx] * \
                                (ucn[k][Ilxx] * usc[k][Ilxx]) * cross_sigma_jk
                            sum2 += np.sum(tmp2, axis=0)

                # Avoid division by zero
                Us_int[Us_int == 0] = 1
                # Update combined convection velocity
                Uc_star[Ilx] = Ws[Ilx] - (sum1 + sum2)[Ilx] / Us_int[Ilx]

                count += 1
        return Us + np.sum(np.where(~Il, us, 0), axis=0)


class CumulativeWakeSum(SuperpositionModel):
    """
    Implemention of the cumulative wake model:
    Majid Bastankhah, Bridget L. Welch, Luis A. Martínez-Tossas, Jennifer King and Paul Fleming
    Analytical solution for the cumulative wake of wind turbines in wind farms
    J. Fluid Mech. (2021), vol. 911, A53, doi:10.1017/jfm.2020.1037
    """

    def __init__(self, alpha=2.):
        # somewhat empirical factor to scale results to fit LES predictions
        self.alpha = alpha  # alpha in 8.3 (1 in eq 6.4 and 2 in eq 4.9)

    def superpose_deficit(self, deficit_jxxx, **kwargs):
        return self(**kwargs)

    def __call__(self, WS0_xxx, WS_eff_xxx, ct_xxx, D_xx, sigma_sqr_jxxx, cw_jxxx, hcw_jxxx, dh_jxxx):

        U0 = WS0_xxx
        WS_eff = WS_eff_xxx
        ct = ct_xxx
        D = D_xx
        downwind = (sigma_sqr_jxxx > 1e-10)
        sigma_sqr = np.where(downwind, sigma_sqr_jxxx, 1)
        cw = cw_jxxx * np.ones_like(sigma_sqr)
        hcw = hcw_jxxx * np.ones_like(sigma_sqr)
        dh = dh_jxxx * np.ones_like(sigma_sqr)

        n_wt = sigma_sqr.shape[0]
        lamCsum = np.zeros(sigma_sqr.shape[1:])

        C_lst = []
        for n in range(n_wt):
            if n > 0:
                sigma_sqr_tot = sigma_sqr[n:n + 1] + sigma_sqr[:n]
                # eq 4.9
                lam = self.alpha * sigma_sqr[:n] / sigma_sqr_tot * np.exp(-(hcw[n][na, ...] - hcw[:n])**2 / (
                    2. * sigma_sqr_tot)) * np.exp(-(dh[n][na, ...] - dh[:n])**2 / (2. * sigma_sqr_tot))
                # lambda sum term in 4.10
                lamCsum = np.sum(lam * np.array(C_lst)[:n], axis=0)

            # sqrt term in eq 4.10
            sqrt_term = (U0[n] - lamCsum)**2 - (ct[n] * D[n, ..., na]**2 * WS_eff[n]**2) / (8. * sigma_sqr[n])
            sqrt_term = np.maximum(sqrt_term, 0)
            Clim = (WS_eff[n] * (1. - np.sqrt(np.maximum(0, 1. - ct[n]))))
            C = (U0[n] - lamCsum) - np.sqrt(sqrt_term)  # Eq 4.10
            C = np.where((sqrt_term > 0.) & (C <= Clim), C, Clim)
            C *= downwind[n]
            C_lst.append(C)
        C = np.array(C_lst)

        exponent = -1 / (2 * sigma_sqr) * cw**2

        return np.sum(C * np.exp(exponent), axis=0)
