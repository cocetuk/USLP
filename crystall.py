import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path




q = 1.602176634e-19
hbar = 1.054571817e-34
kB = 1.380649e-23
c0 = 299792458.0
eps0 = 8.8541878128e-12
me = 9.1093837015e-31


n_l = 1.2e28
E_g = 0.75 * q
alpha0 = 2.5e7
R0 = 0.49
Nre0 = 5.0
Nim0 = 1.6
eps1_0 = Nre0 ** 2 - Nim0 ** 2
mu = 1.0
m_c = me
m_v = me
m_r = m_v * m_c / (m_v + m_c)
d_l = 3.2e-10
tau_r = 1e-12
T0 = 293.0
T_m = 880.0

rho_gst = 6400.0
cp_gst = 210.0
DeltaHm_molar = 12.13e3
M_molar = 2 * 72.63e-3 + 2 * 121.76e-3 + 5 * 127.60e-3
DeltaH_gst = rho_gst * DeltaHm_molar / M_molar


wavelength = 800e-9
omega = 2 * np.pi * c0 / wavelength


rho_sio2 = 2200.0
cp_sio2 = 703.0
k_sio2 = 1.4

k_gst = 0.2

L_film = 180e-9
L_sub = 1.0e-6
L_total = L_film + L_sub


pulse_fs_default = 45.0
F_fig8a = 19.0
F_fig8b = 6.2


dt_stage1 = 0.25e-15
t_stage1_end = 8e-12
dz = 2e-9
dt_stage2_min = 1e-13
dt_stage2_max = 2e-9
t_stage2_end = 1e-6


EA1_list_eV_default = [1.1, 1.2, 1.3, 1.4]
z_melt_boundary_nm = 62.0
nu_a1_default = 1.8e12
nu_a2_default = 1.0e13

# ---------------------------
# Plot style
# ---------------------------
mpl.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "figure.dpi": 120,
    "mathtext.fontset": "cm",
    "mathtext.default": "regular",
    "axes.unicode_minus": False,
})


# =============================================================================
# Utility functions
# =============================================================================
def laser_intensity(t, F_mJ_cm2=20.0, tau_fs=250.0):

    t = np.asarray(t)
    tau = tau_fs * 1e-15
    F = F_mJ_cm2 * 10.0
    I0_max = 2 * np.sqrt(np.log(2)) / (tau * np.sqrt(np.pi)) * F
    t_c = 3.0 * tau
    return I0_max * np.exp(-4.0 * np.log(2.0) * ((t - t_c) / tau) ** 2)


def temperature_to_enthalpy(T):

    u1 = rho_gst * cp_gst * T_m
    if np.isscalar(T):
        if T < T_m:
            return rho_gst * cp_gst * T
        return u1 + DeltaH_gst + rho_gst * cp_gst * (T - T_m)
    T = np.asarray(T)
    u = rho_gst * cp_gst * np.minimum(T, T_m)
    above = T >= T_m
    u = u + above * (DeltaH_gst + rho_gst * cp_gst * (T - T_m))
    return u


def enthalpy_to_temperature(u):

    u1 = rho_gst * cp_gst * T_m
    if np.isscalar(u):
        if u < u1:
            return u / (rho_gst * cp_gst)
        if u < u1 + DeltaH_gst:
            return T_m
        return T_m + (u - u1 - DeltaH_gst) / (rho_gst * cp_gst)
    u = np.asarray(u)
    T = np.empty_like(u, dtype=float)
    mask1 = u < u1
    mask2 = (u >= u1) & (u < u1 + DeltaH_gst)
    mask3 = u >= u1 + DeltaH_gst
    T[mask1] = u[mask1] / (rho_gst * cp_gst)
    T[mask2] = T_m
    T[mask3] = T_m + (u[mask3] - u1 - DeltaH_gst) / (rho_gst * cp_gst)
    return T


def cp_effective(T, latent_band_K=10.0):

    T = np.asarray(T)
    cp = np.full_like(T, cp_gst, dtype=float)
    cp += (DeltaH_gst / rho_gst) / latent_band_K * (np.abs(T - T_m) <= latent_band_K / 2.0)
    return cp


def material_props(z):

    if z <= L_film:
        return rho_gst, cp_gst, k_gst
    return rho_sio2, cp_sio2, k_sio2


def optical_coeffs(ne, eps_e):

    x = np.clip(ne / n_l, 0.0, 1.0)
    V_e = np.sqrt(max(2.0 * eps_e, 1e-30) / me)
    nu_e_l = V_e / d_l

    Z = 1.0
    lnL = 2.0
    A = 5e20
    B = 1.5e-10
    nu_e_i = A * lnL * Z * q ** 4 * max(ne, 0.0) / (me ** 0.5 * max(eps_e, 1e-30) ** 1.5)
    nu_e_e = B * max(eps_e, 1e-30) ** 1.5 / (me ** 0.5 * q ** 2)
    nu_e_pl = nu_e_i + nu_e_e

    nu_d = nu_e_l * (1.0 - x) + nu_e_pl * x
    nu_d = np.clip(nu_d, 5e14, 5e16)

    omega_p = np.sqrt(max(ne, 0.0) * q ** 2 / (m_r * eps0))

    Nre = Nre0
    Nim = Nim0
    for _ in range(4):
        alpha_e = (2.0 * ne * nu_d / (1.0 + (nu_d / omega) ** 2) *
                   (q ** 2 / (4.0 * m_r * omega ** 2)) *
                   (2.0 / (c0 * eps0 * max(Nre, 1e-12))))
        alpha_sum = alpha0 * (1.0 - x) + alpha_e
        Nim = alpha_sum * c0 / (2.0 * omega)
        Nre = np.sqrt(max(eps1_0 - (eps1_0 - 1.0) * x -
                          omega_p ** 2 / (omega ** 2 + nu_d ** 2) + Nim ** 2, 1e-12))

    alpha_e = (2.0 * ne * nu_d / (1.0 + (nu_d / omega) ** 2) *
               (q ** 2 / (4.0 * m_r * omega ** 2)) *
               (2.0 / (c0 * eps0 * max(Nre, 1e-12))))
    alpha_sum = alpha0 * (1.0 - x) + alpha_e
    Nim = alpha_sum * c0 / (2.0 * omega)
    R = ((Nre - 1.0) ** 2 + Nim ** 2) / ((Nre + 1.0) ** 2 + Nim ** 2)
    return R, alpha_sum, alpha_e, nu_d


def critical_energy(eps_q):
    return (1.0 + 2.0 * mu) / (1.0 + mu) * (E_g + eps_q)


def compute_impact_threshold(ne, eps_e, dedt, I_local, nu_d):

    Nre_eff = max(Nre0 * np.sqrt(max(1.0 - 0.3 * ne / n_l, 0.2)), 1.0)
    eps_q = q ** 2 / (4.0 * m_r * omega ** 2) * (2.0 * I_local / (c0 * eps0 * Nre_eff))
    E_cr = critical_energy(eps_q)

    P = 1e15  # Hz
    delta_k = np.sqrt(E_cr / P) * np.sqrt(max(dedt, 0.0))

    if ne > 0 and (1.0 - ne / n_l) > 1e-6:
        V_e = np.sqrt(2.0 * max(eps_e, 1e-30) / me)
        D_e = V_e ** 2 / (3.0 * max(nu_d, 1e-30))
        delta_d = (d_l ** 2) * max(dedt, 0.0) / (D_e * (1.0 - ne / n_l) ** (2.0 / 3.0))
    else:
        delta_d = 0.0

    delta_sh = max(delta_k, delta_d, 0.0)
    return E_cr + delta_sh


def compute_dedt(ne, eps_e, T_l, alpha_e, I, nu_d, beta_R, gamma_R):
    if ne < 1e-30:
        return 0.0
    x = ne / n_l
    eps_eq = 1.5 * kB * T_l

    dEe_dt = (alpha0 * (1.0 - x) * (hbar * omega - E_g) * I / (hbar * omega) +
              alpha_e * I +
              beta_R * ne ** (8.0 / 3.0) * E_g -
              gamma_R * ne ** (5.0 / 3.0) * eps_e / (1.0 + tau_r * gamma_R * ne ** (2.0 / 3.0)) -
              ne * (eps_e - eps_eq) / tau_r)

    dne_dt = (alpha0 * (1.0 - x) * I / (hbar * omega) -
              beta_R * ne ** (8.0 / 3.0) -
              gamma_R * ne ** (5.0 / 3.0) / (1.0 + tau_r * gamma_R * ne ** (2.0 / 3.0)))

    return (dEe_dt - eps_e * dne_dt) / max(ne, 1e-30)


def local_2t_step(ne, E_e, E_l, T_l, I, dt_loc, impact_ionization=True):
    eps_e = E_e / max(ne, 1e-30)
    R, alpha_sum, alpha_e, nu_d = optical_coeffs(ne, eps_e)

    x = ne / n_l
    V_e = np.sqrt(2.0 * max(eps_e, 1e-30) / me)
    D_e = V_e ** 2 / (3.0 * max(nu_d, 1e-30))
    beta_R = D_e / n_l
    gamma_R = D_e
    eps_eq = 1.5 * kB * T_l

    dne_dt = (alpha0 * (1.0 - x) * I / (hbar * omega) -
              beta_R * ne ** (8.0 / 3.0) -
              gamma_R * ne ** (5.0 / 3.0) / (1.0 + tau_r * gamma_R * ne ** (2.0 / 3.0)))

    dEe_dt = (alpha0 * (1.0 - x) * (hbar * omega - E_g) * I / (hbar * omega) +
              alpha_e * I +
              beta_R * ne ** (8.0 / 3.0) * E_g -
              gamma_R * ne ** (5.0 / 3.0) * eps_e / (1.0 + tau_r * gamma_R * ne ** (2.0 / 3.0)) -
              ne * (eps_e - eps_eq) / tau_r)

    dEl_dt = (ne * (eps_e - eps_eq) / tau_r +
              gamma_R * ne ** (5.0 / 3.0) * (eps_e + E_g) / (1.0 + tau_r * gamma_R * ne ** (2.0 / 3.0)))

    ne_new = max(ne + dne_dt * dt_loc, 0.0)
    E_e_new = max(E_e + dEe_dt * dt_loc, 1e-30)
    E_l_new = max(E_l + dEl_dt * dt_loc, 0.0)

    if impact_ionization and ne_new > 0:
        eps_e_new = E_e_new / max(ne_new, 1e-30)
        dedt = compute_dedt(ne, eps_e, T_l, alpha_e, I, nu_d, beta_R, gamma_R)
        E_cr_star = compute_impact_threshold(ne_new, eps_e_new, dedt, I, nu_d)

        while eps_e_new >= E_cr_star and ne_new < n_l:
            dn = ne_new * (1.0 - ne_new / n_l)
            if dn <= 0.0:
                break
            ne_new += dn
            E_e_new = max(E_e_new - dn * E_g, 1e-30)
            eps_e_new = max((E_cr_star - E_g) / 2.0, 0.0)
            E_cr_star = compute_impact_threshold(ne_new, eps_e_new, dedt, I, nu_d)

    T_new = enthalpy_to_temperature(E_l_new)
    return ne_new, E_e_new, E_l_new, T_new, R, alpha_sum, alpha_e, nu_d



def simulate_stage1_profile(
        F_mJ_cm2=19.0,
        pulse_fs=pulse_fs_default,
        dt=dt_stage1,
        t_end=t_stage1_end,
        dz=dz,
        impact_ionization=True,
):
    z_m = np.arange(0.0, L_film + 0.5 * dz, dz)
    Nz = len(z_m)
    t_s = np.arange(0.0, t_end + dt, dt)
    Nt = len(t_s)

    n_e = np.full(Nz, 1e21, dtype=float)
    E_e = n_e * (0.0375 * q)
    E_l = np.full(Nz, temperature_to_enthalpy(T0), dtype=float)
    T_l = np.full(Nz, T0, dtype=float)

    T_hist = np.zeros((Nz, Nt), dtype=float)
    ne_hist = np.zeros((Nz, Nt), dtype=float)
    alpha_hist = np.zeros((Nz, Nt), dtype=float)
    R_hist = np.zeros((Nt,), dtype=float)

    T_hist[:, 0] = T_l
    ne_hist[:, 0] = n_e

    for i in range(Nt - 1):
        I0 = laser_intensity(t_s[i], F_mJ_cm2=F_mJ_cm2, tau_fs=pulse_fs)
        eps_e_surf = E_e[0] / max(n_e[0], 1e-30)
        R_surf, _, _, _ = optical_coeffs(n_e[0], eps_e_surf)
        R_hist[i] = R_surf

        I_local = (1.0 - R_surf) * I0

        for j in range(Nz):
            ne_new, E_e_new, E_l_new, T_new, Rj, alpha_sum, alpha_e, nu_d = local_2t_step(
                n_e[j], E_e[j], E_l[j], T_l[j], I_local, dt, impact_ionization=impact_ionization
            )
            n_e[j] = ne_new
            E_e[j] = E_e_new
            E_l[j] = E_l_new
            T_l[j] = T_new
            alpha_hist[j, i] = alpha_sum
            ne_hist[j, i + 1] = n_e[j]
            T_hist[j, i + 1] = T_l[j]

            if j < Nz - 1:
                I_local = I_local * np.exp(-alpha_sum * dz)

        R_hist[i + 1] = R_hist[i]

    return z_m, t_s, T_hist, ne_hist, alpha_hist, R_hist


def tridiag_solve(a, b, c, d):
    n = len(d)
    cp = np.zeros(n, dtype=float)
    dp = np.zeros(n, dtype=float)

    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]

    for i in range(1, n):
        denom = b[i] - a[i] * cp[i - 1]
        cp[i] = c[i] / denom if i < n - 1 else 0.0
        dp[i] = (d[i] - a[i] * dp[i - 1]) / denom

    x = np.zeros(n, dtype=float)
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x


def make_log_time_grid(t0, t1, n_log=240, n_lin=80, t_lin_end=2e-10):
    t_lin = np.linspace(t0, min(t_lin_end, t1), n_lin, endpoint=False)
    t_start = max(t_lin[-1] if len(t_lin) else t0, 1e-15)
    if t_start <= 0:
        t_start = 1e-15
    t_log = np.geomspace(t_start, t1, n_log)
    t = np.unique(np.concatenate([t_lin, t_log]))
    t.sort()
    return t


def initial_full_domain_temperature(T_film_end, dz=dz):
    z_all = np.arange(0.0, L_total + 0.5 * dz, dz)
    T_init = np.full_like(z_all, T0, dtype=float)
    n_film = min(len(T_film_end), int(np.round(L_film / dz)) + 1)
    T_init[:n_film] = T_film_end[:n_film]
    return z_all, T_init


def simulate_stage2_heat(T_init, z_m, t_s, latent_band_K=10.0):
    z_m = np.asarray(z_m)
    T = np.asarray(T_init, dtype=float).copy()
    Nz = len(z_m)
    Nt = len(t_s)
    dz_loc = z_m[1] - z_m[0]

    T_hist = np.zeros((Nz, Nt), dtype=float)
    T_hist[:, 0] = T.copy()

    rho_arr = np.array([material_props(z)[0] for z in z_m], dtype=float)
    cp_base_arr = np.array([material_props(z)[1] for z in z_m], dtype=float)
    k_arr = np.array([material_props(z)[2] for z in z_m], dtype=float)

    for n in range(Nt - 1):
        dt = t_s[n + 1] - t_s[n]

        T_guess = T.copy()
        for _ in range(3):
            cp_eff = cp_base_arr.copy()
            film_mask = z_m <= L_film
            cp_eff[film_mask] = cp_effective(T_guess[film_mask], latent_band_K=latent_band_K)

            rhoCp = rho_arr * cp_eff

            a = np.zeros(Nz, dtype=float)
            b = np.zeros(Nz, dtype=float)
            c = np.zeros(Nz, dtype=float)
            d = rhoCp * T

            k_half_top = 0.5 * (k_arr[0] + k_arr[1])
            b[0] = rhoCp[0] + 2.0 * dt * k_half_top / dz_loc ** 2
            c[0] = -2.0 * dt * k_half_top / dz_loc ** 2
            a[0] = 0.0
            d[0] = rhoCp[0] * T[0]

            for i in range(1, Nz - 1):
                k_w = 0.5 * (k_arr[i - 1] + k_arr[i])
                k_e = 0.5 * (k_arr[i] + k_arr[i + 1])
                a[i] = -dt * k_w / dz_loc ** 2
                b[i] = rhoCp[i] + dt * (k_w + k_e) / dz_loc ** 2
                c[i] = -dt * k_e / dz_loc ** 2
                d[i] = rhoCp[i] * T[i]

            a[-1] = 0.0
            b[-1] = 1.0
            c[-1] = 0.0
            d[-1] = T0

            T_new = tridiag_solve(a, b, c, d)
            T_guess = 0.6 * T_guess + 0.4 * T_new

        T = T_guess
        T_hist[:, n + 1] = T

    return T_hist


def compute_psi_profile(z_m, t_s, T_hist, Ea1_eV, nu_a1=nu_a1_default, nu_a2=nu_a2_default,
                        z_boundary_nm=z_melt_boundary_nm):
    z_nm = z_m * 1e9
    psi = np.zeros(len(z_m), dtype=float)

    for i, z in enumerate(z_nm):
        nu_a = nu_a1 if z <= z_boundary_nm else nu_a2
        Tz = T_hist[i, :]
        w = (Tz < T_m).astype(float)
        integrand = w * nu_a * np.exp(-(Ea1_eV * q) / (kB * np.clip(Tz, 1.0, None)))
        integral = np.trapezoid(integrand, t_s)  # FIXED: trapz is deprecated
        psi[i] = 1.0 - np.exp(-integral)

    return np.clip(psi, 0.0, 1.0)


def compute_growth_profile(z_m, t_s, T_hist, Ea2_eV, nu_a1=nu_a1_default, nu_a2=nu_a2_default,
                           z_boundary_nm=z_melt_boundary_nm):
    z_nm = z_m * 1e9
    r_ratio = np.ones(len(z_m), dtype=float)

    for i, z in enumerate(z_nm):
        nu_a = nu_a1 if z <= z_boundary_nm else nu_a2
        Tz = T_hist[i, :]
        w = (Tz < T_m).astype(float)
        integrand = w * nu_a * np.exp(-(Ea2_eV * q) / (kB * np.clip(Tz, 1.0, None)))
        integral = np.trapezoid(integrand, t_s)  # FIXED: trapz is deprecated
        r_ratio[i] = 1.0 + integral
    return r_ratio



def plot_temperature_profiles(t_s, z_m, T_hist, depths_nm, title, out_path=None, ylim=None):
    z_nm = z_m * 1e9
    idx = [int(np.argmin(np.abs(z_nm - d))) for d in depths_nm]

    fig, ax = plt.subplots(figsize=(9.8, 6.6), constrained_layout=True)
    for j in idx:
        ax.plot(t_s, T_hist[j, :], lw=1.7, label=fr"$z={z_nm[j]:.0f}$ нм")  # Removed * 1e9 for x-axis

    ax.axhline(T_m, ls="--", lw=1.0, color="gray", label=fr"$T_m={T_m:.0f}$ К")
    ax.set_title(title, pad=12)
    ax.set_xlabel("Время, с", labelpad=10)  # Changed label to seconds
    ax.set_ylabel(r"$T_\ell$, К", labelpad=14)
    ax.tick_params(direction="in")
    ax.margins(x=0)

    # FIXED: Added log scale and boundaries exactly matching the article
    ax.set_xscale('log')
    ax.set_xlim(1e-11, 1e-6)

    if ylim is not None:
        ax.set_ylim(*ylim)
    else:
        ymin = np.min(T_hist[np.isfinite(T_hist)])
        ymax = np.max(T_hist[np.isfinite(T_hist)])
        ax.set_ylim(max(250, ymin - 20), ymax + 40)

    ax.legend(frameon=False, ncol=2)
    if out_path is not None:
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
    return fig, ax


def plot_figure_9(z_m, psi_profiles, Ea1_list_eV, out_path=None):
    z_nm = z_m * 1e9
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.8), constrained_layout=True)
    axes = axes.ravel()

    ylims = {
        1.1: (0.0, 0.20),
        1.2: (0.0, 5e-2),
        1.3: (0.0, 12e-3),
        1.4: (0.0, 4e-3),
    }

    for ax, Ea, psi in zip(axes, Ea1_list_eV, psi_profiles):
        # FIXED: Split array to avoid matplotlib drawing a straight continuous line across the mathematical gap
        mask_melt = z_nm <= z_melt_boundary_nm
        mask_solid = z_nm > z_melt_boundary_nm

        ax.plot(z_nm[mask_melt], psi[mask_melt], lw=1.7, color='C0')
        ax.plot(z_nm[mask_solid], psi[mask_solid], lw=1.7, color='C0')

        ax.axvline(z_melt_boundary_nm, ls="--", lw=1.0, color="gray")
        ax.set_title(fr"$\delta E_{{a1}}={Ea:.1f}$ eV", pad=10)
        ax.set_xlabel("Глубина, нм", labelpad=10)
        ax.set_ylabel(r"$\Psi$", labelpad=14)
        ax.tick_params(direction="in")
        ax.margins(x=0)
        ax.set_xlim(0, 180)  # FIXED: Changed limit to 180 nm
        ax.set_ylim(*ylims.get(round(float(Ea), 1), (0.0, max(0.205, float(np.max(psi)) * 1.08))))

    fig.suptitle(r"Распределение $\Psi(z)$ при $t=10^{-7}$ c", y=1.02, fontsize=13)
    if out_path is not None:
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
    return fig, axes


def plot_growth_profiles(z_m, growth_profiles, Ea2_list_eV, out_path=None):
    z_nm = z_m * 1e9
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.8), constrained_layout=True)
    axes = axes.ravel()
    for ax, Ea, rr in zip(axes, Ea2_list_eV, growth_profiles):
        # FIXED: Ensure gap rendering here too
        mask_melt = z_nm <= z_melt_boundary_nm
        mask_solid = z_nm > z_melt_boundary_nm

        ax.plot(z_nm[mask_melt], rr[mask_melt], lw=1.7, color='C0')
        ax.plot(z_nm[mask_solid], rr[mask_solid], lw=1.7, color='C0')

        ax.axvline(z_melt_boundary_nm, ls="--", lw=1.0, color="gray")
        ax.set_title(fr"$\delta E_{{a2}}={Ea:.1f}$ eV", pad=10)
        ax.set_xlabel("Глубина, нм", labelpad=10)
        ax.set_ylabel(r"$r_{\mathrm{cr}}/r_0$", labelpad=14)
        ax.tick_params(direction="in")
        ax.margins(x=0)
        ax.set_xlim(0, 180)  # FIXED: Changed limit to 180 nm
    if out_path is not None:
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
    return fig, axes



def run_all(
        depths_fig8_nm=(0, 20, 40, 60, 80, 100, 120, 140, 160),
        pulse_fs=pulse_fs_default,
        make_psi=True,
        make_growth=False,
        F_a=F_fig8a,
        F_b=F_fig8b,
        ea1_list=EA1_list_eV_default,
        ea2_list=(0.6, 0.7, 0.8, 0.9),
        out_dir=".",
):
    out_dir = Path(out_dir)


    print("Stage 1: F = 19 mJ/cm^2 ...")
    z_film, t1, T1_hist, ne1_hist, alpha1_hist, R1_hist = simulate_stage1_profile(
        F_mJ_cm2=F_a, pulse_fs=pulse_fs, impact_ionization=True
    )
    T_end_film = T1_hist[:, -1]

    z_all, T_init = initial_full_domain_temperature(T_end_film)
    t2 = make_log_time_grid(0.0, t_stage2_end, n_log=260, n_lin=90, t_lin_end=2e-10)
    print("Stage 2: F = 19 mJ/cm^2 ...")
    T2_hist_a = simulate_stage2_heat(T_init, z_all, t2, latent_band_K=10.0)

    # --- stage 1 + 2 for F = 6.2 mJ/cm^2 ---
    print("Stage 1: F = 6.2 mJ/cm^2 ...")
    z_film_b, t1b, T1_hist_b, ne1_hist_b, alpha1_hist_b, R1_hist_b = simulate_stage1_profile(
        F_mJ_cm2=F_b, pulse_fs=pulse_fs, impact_ionization=True
    )
    T_end_film_b = T1_hist_b[:, -1]
    z_all_b, T_init_b = initial_full_domain_temperature(T_end_film_b)
    print("Stage 2: F = 6.2 mJ/cm^2 ...")
    T2_hist_b = simulate_stage2_heat(T_init_b, z_all_b, t2, latent_band_K=10.0)

    # --- Fig. 8-like plots ---
    fig8a, ax8a = plot_temperature_profiles(
        t2, z_all, T2_hist_a, depths_fig8_nm,
        title=fr"Температурная динамика второй стадии, $F={F_a:.1f}$ мДж/см$^2$",
        out_path=out_dir / f"fig8a_T_F_{F_a:.1f}.png"
    )
    fig8b, ax8b = plot_temperature_profiles(
        t2, z_all_b, T2_hist_b, depths_fig8_nm,
        title=fr"Температурная динамика второй стадии, $F={F_b:.1f}$ мДж/см$^2$",
        out_path=out_dir / f"fig8b_T_F_{F_b:.1f}.png"
    )

    # --- Fig. 9-like plots ---
    psi_profiles = []
    growth_profiles = []
    if make_psi:
        for Ea in ea1_list:
            psi_profiles.append(compute_psi_profile(z_all, t2, T2_hist_a, Ea))
        plot_figure_9(
            z_all, psi_profiles, ea1_list,
            out_path=out_dir / "fig9_psi_profiles.png"
        )
    if make_growth:
        for Ea in ea2_list:
            growth_profiles.append(compute_growth_profile(z_all, t2, T2_hist_a, Ea))
        plot_growth_profiles(
            z_all, growth_profiles, ea2_list,
            out_path=out_dir / "fig10_growth_profiles.png"
        )

    print(f"Saved figures to: {out_dir.resolve()}")
    return {
        "z_film": z_film,
        "t1": t1,
        "T1_hist": T1_hist,
        "z_all": z_all,
        "t2": t2,
        "T2_hist_a": T2_hist_a,
        "T2_hist_b": T2_hist_b,
        "psi_profiles": psi_profiles,
        "growth_profiles": growth_profiles,
    }


if __name__ == "__main__":
    depths = [0, 20, 40, 60, 80, 100]
    run_all(depths_fig8_nm=depths, pulse_fs=pulse_fs_default, make_psi=True, make_growth=False)