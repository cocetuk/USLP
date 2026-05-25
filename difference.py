import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# Константы
# ============================================================================
q = 1.602176634e-19          # Кл
hbar = 1.054571817e-34       # Дж*с
kB = 1.380649e-23            # Дж/К
c = 299792458.0              # м/с
eps0 = 8.8541878128e-12      # Ф/м
me = 9.1093837015e-31        # кг

# ============================================================================
# Параметры αGST
# ============================================================================
n_l = 1.2e28                 # м^-3, плотность молекул
E_g = 0.75 * q               # Дж, ширина запрещенной зоны
alpha0 = 2.5e7               # м^-1, коэффициент поглощения
R0 = 0.49                    # отражение не возбужденного материала
Nre0 = 5.0                   # начальная действительная часть показателя
Nim0 = 1.6                   # начальная мнимая часть
eps1_0 = Nre0**2 - Nim0**2   # действительная часть диэлектрической проницаемости
mu = 1.0
m_c = me
m_v = me
m_r = m_v * m_c / (m_v + m_c) # приведенная масса
d_l = 3.2e-10                # м, межмолекулярное расстояние
tau_r = 1e-12                # с, время релаксации электрон-решетка
T0 = 293.0                   # К, начальная температура
T_m = 880.0                  # К, температура плавления
rho = 6400.0                 # кг/м^3, плотность
c_p = 210.0                  # Дж/(кг*К), теплоемкость
DeltaHm_molar = 12.13e3      # Дж/моль, скрытая теплота плавления
M_molar = 2*72.63e-3 + 2*121.76e-3 + 5*127.60e-3  # кг/моль, молярная масса Ge₂Sb₂Te₅
DeltaH = rho * DeltaHm_molar / M_molar  # Дж/м^3, объемная теплота плавления

# Параметры лазера
wavelength = 800e-9          # м
omega = 2 * np.pi * c / wavelength  # рад/с

# Параметры численного моделирования
F_mJ_cm2 = 20.0              # флюенс, мДж/см^2
pulse_fs = 250.0             # длительность импульса, фс
t_end = 6e-12                # с
dt = 0.05e-15                # с
t = np.arange(0.0, t_end + dt, dt)

# ============================================================================
# Функции
# ============================================================================
def laser_intensity(t, F_mJ_cm2=20.0, tau_fs=250.0):
    tau = tau_fs * 1e-15
    F = F_mJ_cm2 * 10.0
    I0_max = 2 * np.sqrt(np.log(2)) / (tau * np.sqrt(np.pi)) * F
    t_c = 3 * tau
    return I0_max * np.exp(-4 * np.log(2) * ((t - t_c) / tau) ** 2)

def enthalpy_to_temperature(u):
    u1 = rho * c_p * T_m
    if u < u1:
        return u / (rho * c_p)
    if u < u1 + DeltaH:
        return T_m
    return T_m + (u - u1 - DeltaH) / (rho * c_p)

def temperature_to_enthalpy(T):
    u1 = rho * c_p * T_m
    if T < T_m:
        return rho * c_p * T
    return u1 + DeltaH + rho * c_p * (T - T_m)

def optical_coeffs(ne, eps_e):
    x = np.clip(ne / n_l, 0.0, 1.0)
    V_e = np.sqrt(max(2.0 * eps_e, 1e-30) / me)


    nu_e_l = V_e / d_l

    Z = 1.0
    lnL = 2.0
    A = 5e20
    B = 1.5e-10
    nu_e_i = A * lnL * Z * q**4 * max(ne, 0.0) / (me**0.5 * max(eps_e, 1e-30)**1.5)
    nu_e_e = B * max(eps_e, 1e-30)**1.5 / (me**0.5 * q**2)
    nu_e_pl = nu_e_i + nu_e_e

    nu_d = nu_e_l * (1.0 - x) + nu_e_pl * x
    nu_d = np.clip(nu_d, 5e14, 5e16)

    omega_p = np.sqrt(max(ne, 0.0) * q**2 / (m_r * eps0))

    Nre = Nre0
    Nim = Nim0
    for _ in range(4):
        alpha_e = (2.0 * ne * nu_d / (1.0 + (nu_d / omega)**2) *
                   (q**2 / (4.0 * m_r * omega**2)) *
                   (2.0 / (c * eps0 * max(Nre, 1e-12))))
        alpha_sum = alpha0 * (1.0 - x) + alpha_e
        Nim = alpha_sum * c / (2.0 * omega)
        Nre = np.sqrt(max(eps1_0 - (eps1_0 - 1.0) * x -
                          omega_p**2 / (omega**2 + nu_d**2) + Nim**2, 1e-12))

    alpha_e = (2.0 * ne * nu_d / (1.0 + (nu_d / omega)**2) *
               (q**2 / (4.0 * m_r * omega**2)) *
               (2.0 / (c * eps0 * max(Nre, 1e-12))))
    alpha_sum = alpha0 * (1.0 - x) + alpha_e
    Nim = alpha_sum * c / (2.0 * omega)
    R = ((Nre - 1.0)**2 + Nim**2) / ((Nre + 1.0)**2 + Nim**2)
    return R, alpha_sum, alpha_e, nu_d

def critical_energy(eps_q):
    """Формула (17) из J. Appl."""
    return (1.0 + 2.0 * mu) / (1.0 + mu) * (E_g + eps_q)

def compute_impact_threshold(ne, eps_e, dedt, I_local, nu_d):
    """
    Пороговая энергия E_cr* = E_cr + max(Δε_k, Δε_d)
    Δε_k – формула (4) из JOSAB
    Δε_d – формула (19) из J. Appl.
    """
    Nre_eff = max(Nre0 * np.sqrt(max(1.0 - 0.3 * ne / n_l, 0.2)), 1.0)
    eps_q = q**2 / (4.0 * m_r * omega**2) * (2.0 * I_local / (c * eps0 * Nre_eff))
    E_cr = critical_energy(eps_q)

    P = 1e15                       # частота попыток, Гц
    delta_k = np.sqrt(E_cr / P) * np.sqrt(max(dedt, 0.0))

    if ne > 0 and (1.0 - ne / n_l) > 1e-6:
        V_e = np.sqrt(2.0 * max(eps_e, 1e-30) / me)
        D_e = V_e**2 / (3.0 * max(nu_d, 1e-30))
        delta_d = (d_l**2) * max(dedt, 0.0) / (D_e * (1.0 - ne / n_l)**(2.0/3.0))
    else:
        delta_d = 0.0

    delta_sh = max(delta_k, delta_d, 0.0)
    return E_cr + delta_sh

def compute_dedt(ne, eps_e, T_l, alpha_e, I, nu_d, beta_R, gamma_R):
    """Производная средней энергии электрона dε/dt"""
    if ne < 1e-30:
        return 0.0
    x = ne / n_l
    eps_eq = 1.5 * kB * T_l

    dEe_dt = (alpha0 * (1.0 - x) * (hbar * omega - E_g) * I / (hbar * omega) +
              alpha_e * I +
              beta_R * ne**(8.0/3.0) * E_g -
              gamma_R * ne**(5.0/3.0) * eps_e / (1.0 + tau_r * gamma_R * ne**(2.0/3.0)) -
              ne * (eps_e - eps_eq) / tau_r)

    dne_dt = (alpha0 * (1.0 - x) * I / (hbar * omega) -
              beta_R * ne**(8.0/3.0) -
              gamma_R * ne**(5.0/3.0) / (1.0 + tau_r * gamma_R * ne**(2.0/3.0)))

    dedt = (dEe_dt - eps_e * dne_dt) / max(ne, 1e-30)
    return dedt
# -----------------------------
# Модель T1
# -----------------------------
def simulate_1T(t, F_mJ_cm2=20.0, pulse_fs=250.0):
    Tl = np.zeros_like(t)
    Tl[0] = T0
    u_l = temperature_to_enthalpy(T0)

    for i in range(len(t) - 1):
        I0 = laser_intensity(t[i], F_mJ_cm2=F_mJ_cm2, tau_fs=pulse_fs)
        qdot = (1.0 - R0) * alpha0 * I0
        u_l = u_l + qdot * (t[i + 1] - t[i])
        Tl[i + 1] = enthalpy_to_temperature(u_l)

    return Tl
# ============================================================================
# Модель T2
# ============================================================================
def simulate_2T(t, impact_ionization=True):
    n_e = np.zeros_like(t)
    E_e = np.zeros_like(t)
    E_l = np.zeros_like(t)
    T_l = np.zeros_like(t)
    eps_e = np.zeros_like(t)
    R_hist = np.zeros_like(t)
    alpha_hist = np.zeros_like(t)
    alpha_e_hist = np.zeros_like(t)

    n_e[0] = 1e21
    eps_e[0] = 0.0375 * q
    E_e[0] = n_e[0] * eps_e[0]
    E_l[0] = temperature_to_enthalpy(T0)
    T_l[0] = T0
    R_hist[0] = R0
    alpha_hist[0] = alpha0
    alpha_e_hist[0] = 0.0

    for i in range(len(t) - 1):
        dt_loc = t[i + 1] - t[i]
        eps_e_i = E_e[i] / max(n_e[i], 1e-30)

        R, alpha_sum, alpha_e, nu_d = optical_coeffs(n_e[i], eps_e_i)

        R_hist[i + 1] = R
        alpha_hist[i + 1] = alpha_sum
        alpha_e_hist[i + 1] = alpha_e

        I0 = laser_intensity(t[i], F_mJ_cm2, pulse_fs)
        I = (1.0 - R) * I0

        x = n_e[i] / n_l
        V_e = np.sqrt(2.0 * eps_e_i / me)
        D_e = V_e ** 2 / (3.0 * max(nu_d, 1e-30))
        beta_R = D_e / n_l
        gamma_R = D_e
        eps_eq = 1.5 * kB * T_l[i]

        dne_dt = (alpha0 * (1.0 - x) * I / (hbar * omega) -
                  beta_R * n_e[i] ** (8.0 / 3.0) -
                  gamma_R * n_e[i] ** (5.0 / 3.0) / (1.0 + tau_r * gamma_R * n_e[i] ** (2.0 / 3.0)))

        dEe_dt = (alpha0 * (1.0 - x) * (hbar * omega - E_g) * I / (hbar * omega) +
                  alpha_e * I +
                  beta_R * n_e[i] ** (8.0 / 3.0) * E_g -
                  gamma_R * n_e[i] ** (5.0 / 3.0) * eps_e_i / (1.0 + tau_r * gamma_R * n_e[i] ** (2.0 / 3.0)) -
                  n_e[i] * (eps_e_i - eps_eq) / tau_r)

        dEl_dt = (n_e[i] * (eps_e_i - eps_eq) / tau_r +
                  gamma_R * n_e[i] ** (5.0 / 3.0) * (eps_e_i + E_g) / (1.0 + tau_r * gamma_R * n_e[i] ** (2.0 / 3.0)))

        n_e_new = max(n_e[i] + dne_dt * dt_loc, 0.0)
        E_e_new = max(E_e[i] + dEe_dt * dt_loc, 1e-30)
        E_l_new = max(E_l[i] + dEl_dt * dt_loc, 0.0)
        T_l_new = enthalpy_to_temperature(E_l_new)
        eps_e_new = E_e_new / max(n_e_new, 1e-30)

        if impact_ionization and n_e_new > 0:
            dedt = compute_dedt(n_e[i], eps_e_i, T_l[i], alpha_e, I, nu_d, beta_R, gamma_R)
            E_cr_star = compute_impact_threshold(n_e_new, eps_e_new, dedt, I, nu_d)

            while eps_e_new >= E_cr_star and n_e_new < n_l:
                dn = n_e_new * (1.0 - n_e_new / n_l)
                if dn <= 0.0:
                    break
                n_e_new += dn
                E_e_new -= dn * E_g
                eps_e_new = max((E_cr_star - E_g) / 2.0, 0.0)
                E_cr_star = compute_impact_threshold(n_e_new, eps_e_new, dedt, I, nu_d)

        n_e[i + 1] = n_e_new
        E_e[i + 1] = E_e_new
        E_l[i + 1] = E_l_new
        T_l[i + 1] = T_l_new
        eps_e[i + 1] = eps_e_new

    return n_e, E_e, E_l, T_l, eps_e, R_hist, alpha_hist, alpha_e_hist

# ============================================================================
# Расчет и построение графиков
# ============================================================================
print("Расчет T1...")
Tl_1T = simulate_1T(t, F_mJ_cm2=F_mJ_cm2, pulse_fs=pulse_fs)
print("Расчет без ионизации...")
res_no = simulate_2T(t, impact_ionization=False)
n_no, Ee_no, El_no, Tl_no, ee_no, R_no, alpha_no, alpha_e_no = res_no
print("Расчет с ионизацией...")
res_yes = simulate_2T(t, impact_ionization=True)
n_yes, Ee_yes, El_yes, Tl_yes, ee_yes, R_yes, alpha_yes, alpha_e_yes = res_yes

I_t = laser_intensity(t, F_mJ_cm2=F_mJ_cm2, tau_fs=pulse_fs)
import matplotlib as mpl
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------
# Единый стиль оформления
# --------------------------------------------------------------------------
mpl.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'legend.fontsize': 10,
    'figure.dpi': 100,

    'mathtext.fontset': 'cm',
    'mathtext.default': 'regular',
    'text.usetex': False,

    'axes.unicode_minus': False
})


# ==========================================================================
# ГРАФИК 1
# Интенсивность + температура решетки
# ==========================================================================
I_t = laser_intensity(t, F_mJ_cm2=F_mJ_cm2, tau_fs=pulse_fs)

fig2, ax = plt.subplots(
    2, 1,
    figsize=(8.5, 7.5),
    sharex=True
)

# --------------------------------------------------------------------------
# Интенсивность
# --------------------------------------------------------------------------
ax[0].plot(t * 1e12, I_t, linewidth=1.5)

ax[0].set_ylabel(r'$I(t)$, Вт/м$^2$')
ax[0].set_title('Интенсивность лазерного импульса')

ax[0].set_xlim(0, 6)
ax[0].margins(x=0)
ax[0].tick_params(direction='in')


# --------------------------------------------------------------------------
# Температура решетки
# --------------------------------------------------------------------------
ax[1].plot(t * 1e12, Tl_1T, linewidth=1.5,
           label='Однотемпературная модель')

ax[1].plot(t * 1e12, Tl_no, linewidth=1.5,
           label='Без ионизации')

ax[1].plot(t * 1e12, Tl_yes, linewidth=1.5,
           label='С ионизацией')

ax[1].set_xlabel(r'Время, пс')
ax[1].set_ylabel(r'$T_\ell$, К')

ax[1].set_title('Температура решетки на поверхности')

ax[1].legend(frameon=False)

ax[1].set_xlim(0, 6)
ax[1].margins(x=0)
ax[1].tick_params(direction='in')

fig2.tight_layout()


# ==========================================================================
# ГРАФИК 2
# Электронные характеристики
# ==========================================================================
fig5, ax = plt.subplots(
    3, 1,
    figsize=(8.5, 10),
    sharex=True
)

# --------------------------------------------------------------------------
# Средняя энергия электронов
# --------------------------------------------------------------------------
ax[0].plot(
    t * 1e12,
    ee_no / q,
    linewidth=1.5,
    label='Без ионизации'
)

ax[0].plot(
    t * 1e12,
    ee_yes / q,
    linewidth=1.5,
    label='С ионизацией'
)

ax[0].set_ylabel(r'$\varepsilon_e$, эВ')

ax[0].set_title('Средняя энергия электронов')

ax[0].legend(frameon=False)

ax[0].set_xlim(0, 6)
ax[0].margins(x=0)
ax[0].tick_params(direction='in')


# --------------------------------------------------------------------------
# Плотность свободных электронов
# --------------------------------------------------------------------------
ax[1].plot(
    t * 1e12,
    n_no / 1e27,
    linewidth=1.5,
    label='Без ионизации'
)

ax[1].plot(
    t * 1e12,
    n_yes / 1e27,
    linewidth=1.5,
    label='С ионизацией'
)

ax[1].set_ylabel(r'$n_e$, $10^{27}$ м$^{-3}$')

ax[1].set_title('Плотность свободных электронов')

ax[1].legend(frameon=False)

ax[1].set_xlim(0, 6)
ax[1].margins(x=0)
ax[1].tick_params(direction='in')


# --------------------------------------------------------------------------
# Плотность энергии электронов
# --------------------------------------------------------------------------
ax[2].plot(
    t * 1e12,
    (n_no * ee_no) / 1e9,
    linewidth=1.5,
    label='Без ионизации'
)

ax[2].plot(
    t * 1e12,
    (n_yes * ee_yes) / 1e9,
    linewidth=1.5,
    label='С ионизацией'
)

ax[2].set_xlabel(r'Время, пс')

ax[2].set_ylabel(
    r'$n_e \varepsilon_e$, $10^{9}$ Дж/м$^3$'
)

ax[2].set_title('Плотность энергии электронов')

ax[2].legend(frameon=False)

ax[2].set_xlim(0, 6)
ax[2].margins(x=0)
ax[2].tick_params(direction='in')

fig5.tight_layout()


# ==========================================================================
# ГРАФИК 3
# Коэффициент отражения
# ==========================================================================
figR, ax = plt.subplots(figsize=(8.5, 5))

ax.plot(
    t * 1e12,
    R_no,
    linewidth=1.5,
    label='Без ионизации'
)

ax.plot(
    t * 1e12,
    R_yes,
    linewidth=1.5,
    label='С ионизацией'
)

ax.axhline(
    y=R0,
    color='gray',
    linestyle='--',
    linewidth=0.8,
    label=rf'$R_0 = {R0:.3f}$'
)

ax.set_xlabel(r'Время, пс')

ax.set_ylabel(r'$R$')

ax.set_title('Динамика коэффициента отражения')

ax.legend(frameon=False)

ax.set_xlim(0, 6)
ax.margins(x=0)
ax.tick_params(direction='in')

figR.tight_layout()


# ==========================================================================
# ГРАФИК 4
# Коэффициенты поглощения
# ==========================================================================
fig3, ax = plt.subplots(
    2, 1,
    figsize=(8.5, 8),
    sharex=True
)

# --------------------------------------------------------------------------
# Полный коэффициент поглощения
# --------------------------------------------------------------------------
ax[0].plot(
    t * 1e12,
    alpha_no / 1e7,
    linewidth=1.5,
    label='Без ионизации'
)

ax[0].plot(
    t * 1e12,
    alpha_yes / 1e7,
    linewidth=1.5,
    label='С ионизацией'
)

ax[0].axhline(
    y=alpha0 / 1e7,
    color='gray',
    linestyle='--',
    linewidth=0.8,
    label=rf'$\alpha_0 = {alpha0/1e7:.1f}\cdot10^7$ м$^{{-1}}$'
)

ax[0].set_ylabel(
    r'$\alpha_{\Sigma}$, $10^{7}$ м$^{-1}$'
)

ax[0].set_title('Полный коэффициент поглощения')

ax[0].legend(frameon=False)

ax[0].set_xlim(0, 6)
ax[0].margins(x=0)
ax[0].tick_params(direction='in')


# --------------------------------------------------------------------------
# Поглощение свободными электронами
# --------------------------------------------------------------------------
ax[1].plot(
    t * 1e12,
    alpha_e_no / 1e7,
    linewidth=1.5,
    label='Без ионизации'
)

ax[1].plot(
    t * 1e12,
    alpha_e_yes / 1e7,
    linewidth=1.5,
    label='С ионизацией'
)

ax[1].set_xlabel(r'Время, пс')

ax[1].set_ylabel(
    r'$\alpha_e$, $10^{7}$ м$^{-1}$'
)

ax[1].set_title('Поглощение свободными электронами')

ax[1].legend(frameon=False)

ax[1].set_xlim(0, 6)
ax[1].margins(x=0)
ax[1].tick_params(direction='in')

fig3.tight_layout()

import os

# --------------------------------------------------------------------------
# Папка для сохранения
# --------------------------------------------------------------------------
save_path = r'D:\graphs1'

# Создать папку, если ее нет
os.makedirs(save_path, exist_ok=True)

# --------------------------------------------------------------------------
# Сохранение графиков
# --------------------------------------------------------------------------

# Интенсивность + температура
fig2.savefig(
    os.path.join(save_path, '01_Интенсивность_и_температура.png'),
    dpi=300,
    bbox_inches='tight'
)

# Электронные характеристики
fig5.savefig(
    os.path.join(save_path, '02_Электронные_характеристики.png'),
    dpi=300,
    bbox_inches='tight'
)

# Коэффициент отражения
figR.savefig(
    os.path.join(save_path, '03_Коэффициент_отражения.png'),
    dpi=300,
    bbox_inches='tight'
)

# Коэффициенты поглощения
fig3.savefig(
    os.path.join(save_path, '04_Коэффициенты_поглощения.png'),
    dpi=300,
    bbox_inches='tight'
)

print(f'Графики сохранены в папку:\n{save_path}')

plt.show()