import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

# ============================================================================
# Константы
# ============================================================================
q = 1.602176634e-19        # Кл
kB = 1.380649e-23          # Дж/К
c = 299792458.0            # м/с

# ============================================================================
# Параметры αGST
# ============================================================================
R0 = 0.49                  # отражение не возбужденного материала
Nre0 = 5.0                 # начальная действительная часть показателя
Nim0 = 1.6                 # начальная мнимая часть
alpha0 = 2.5e7             # м^-1, коэффициент поглощения

T0 = 293.0                 # К, начальная температура
T_m = 880.0                # К, температура плавления
rho = 6400.0               # кг/м^3
c_p = 210.0                # Дж/(кг*К)
DeltaHm_molar = 12.13e3    # Дж/моль
M_molar = 2 * 72.63e-3 + 2 * 121.76e-3 + 5 * 127.60e-3  # кг/моль
DeltaH = rho * DeltaHm_molar / M_molar  # Дж/м^3

# Теплопроводность αGST (из статьи для длинных импульсов)
k_T = 0.2                  # Вт/(м·К)

# ============================================================================
# Параметры лазера
# ============================================================================
wavelength = 800e-9  # м, оставляем как в вашем исходном коде
omega = 2 * np.pi * c / wavelength

# ============================================================================
# Численное моделирование
# ============================================================================
F_mJ_cm2 = 20.0

# ============================================================================
# Длительности импульсов (фс)
# Менять только эти три числа
# ============================================================================
pulse_durations = [15000, 30000, 70000]

# Автоматические подписи
pulse_labels = {}

for tau in pulse_durations:
    if tau >= 1000:
        pulse_labels[tau] = f'{tau / 1000:.0f} пс'
    else:
        pulse_labels[tau] = f'{tau:.0f} фс'

colors = ['#e41a1c', '#377eb8', '#4daf4a']

# Время: достаточно долго, чтобы увидеть и нагрев, и охлаждение
t_end = 500e-12   # с
dt = 0.1e-12      # с
t = np.arange(0.0, t_end + dt, dt)

# Полубесконечная область по глубине
z_max = 2.0e-6    # м
dz = 5.0e-9       # м
z = np.arange(0.0, z_max + dz, dz)


# ============================================================================
# Функции
# ============================================================================
def laser_intensity(t, F_mJ_cm2=20.0, tau_fs=5000.0):
    """
    Гауссов импульс по времени, как в исходном коде.
    F_mJ_cm2 -> Дж/м^2 через множитель 10
    """
    tau = tau_fs * 1e-15
    F = F_mJ_cm2 * 10.0  # мДж/см^2 -> Дж/м^2
    I0_max = 2 * np.sqrt(np.log(2)) / (tau * np.sqrt(np.pi)) * F
    t_c = 3 * tau
    return I0_max * np.exp(-4 * np.log(2) * ((t - t_c) / tau) ** 2)


def effective_cp(T):
    """
    Аппаратная теплоемкость с учетом плавления в узком интервале.
    Это удобная замена энтальпийной схемы для явной теплопроводности.
    """
    cp_eff = np.full_like(T, c_p, dtype=float)

    # Узкий интервал плавления, чтобы не делать схему слишком жесткой
    dT_melt = 1.0  # К
    melt_mask = (T >= T_m) & (T <= T_m + dT_melt)
    cp_eff[melt_mask] += DeltaH / (rho * dT_melt)

    return cp_eff


def simulate_long_pulse_heat_equation(t, z, F_mJ_cm2=20.0, pulse_fs=5000.0):
    """
    1D теплопроводность в полубесконечном теле:
        rho * cp * dT/dt = k * d2T/dz2 + Q(z,t)

    Источник тепла:
        Q(z,t) = (1 - R0) * alpha0 * exp(-alpha0 z) * I0(t)

    Где I0(t) — временной гауссов импульс.
    """
    nz = len(z)
    nt = len(t)

    T = np.full(nz, T0, dtype=float)
    T_surf = np.zeros(nt, dtype=float)
    T_surf[0] = T[0]

    exp_abs = np.exp(-alpha0 * z)

    for i in range(nt - 1):
        dt_loc = t[i + 1] - t[i]

        I0 = laser_intensity(t[i], F_mJ_cm2=F_mJ_cm2, tau_fs=pulse_fs)
        Q = (1.0 - R0) * alpha0 * exp_abs * I0   # Вт/м^3

        cp_eff = effective_cp(T)
        coeff = k_T / (rho * cp_eff)

        d2Tdz2 = np.empty_like(T)

        # Внутренние узлы
        d2Tdz2[1:-1] = (T[2:] - 2.0 * T[1:-1] + T[:-2]) / dz**2

        # Граничные условия: нулевой тепловой поток на z=0 и z=z_max
        d2Tdz2[0] = 2.0 * (T[1] - T[0]) / dz**2
        d2Tdz2[-1] = 2.0 * (T[-2] - T[-1]) / dz**2

        dTdt = coeff * d2Tdz2 + Q / (rho * cp_eff)
        T = T + dt_loc * dTdt

        T_surf[i + 1] = T[0]

    return T_surf, T


# ============================================================================
# Расчет
# ============================================================================
print("Запуск расчетов для длинных импульсов (1D теплопроводность)...")
results = {}

for pulse_fs in pulse_durations:
    print(f"  Расчет для длительности {pulse_labels[pulse_fs]}...")
    T_surf, T_final = simulate_long_pulse_heat_equation(
        t, z,
        F_mJ_cm2=F_mJ_cm2,
        pulse_fs=pulse_fs
    )

    results[pulse_fs] = {
        'T_l': T_surf,      # температура решетки на поверхности
        'T_final': T_final  # профиль по глубине на конец расчета
    }

print("Расчеты завершены!\n")

# ============================================================================
# Оформление графиков
# ============================================================================
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


def plot_stacked_quantity(quantity_key, ylabel, title, yfactor=1.0,
                          reference_line=None, ref_label=''):
    """
    Figure с subplots друг под другом — как в вашем исходном стиле.
    """
    n_pulses = len(pulse_durations)
    fig, axes = plt.subplots(
        n_pulses, 1,
        figsize=(10, 2.5 * n_pulses),
        sharex=True
    )

    if n_pulses == 1:
        axes = [axes]

    for idx, pulse_fs in enumerate(pulse_durations):
        ax = axes[idx]
        data = results[pulse_fs][quantity_key] * yfactor
        ax.plot(t * 1e12, data, color=colors[idx], linewidth=1.5)

        if reference_line is not None:
            ax.axhline(
                y=reference_line * yfactor,
                color='gray',
                linestyle='--',
                linewidth=0.8,
                alpha=0.7
            )

        ax.set_title(
            pulse_labels[pulse_fs],
            loc='right',
            fontweight='bold',
            pad=6
        )

        ax.set_ylabel(ylabel)
        ax.set_xlim(0, t_end * 1e12)
        ax.margins(x=0)
        ax.tick_params(direction='in')

    axes[-1].set_xlabel(r'Время, пс')
    fig.suptitle(title, fontsize=13, fontweight='bold', y=0.98)
    fig.subplots_adjust(top=0.92, hspace=0.28)
    return fig


# ============================================================================
# График 1: Интенсивность лазерного импульса
# ============================================================================
fig_intensity, axes_int = plt.subplots(
    len(pulse_durations), 1,
    figsize=(10, 2.5 * len(pulse_durations)),
    sharex=True
)

for idx, pulse_fs in enumerate(pulse_durations):
    ax = axes_int[idx]
    I_t = laser_intensity(t, F_mJ_cm2=F_mJ_cm2, tau_fs=pulse_fs)

    ax.fill_between(t * 1e12, 0, I_t / 1e15, color=colors[idx], alpha=0.3)
    ax.plot(t * 1e12, I_t / 1e15, color=colors[idx], linewidth=1.5)

    ax.set_title(
        pulse_labels[pulse_fs],
        loc='right',
        fontweight='bold',
        pad=6
    )

    ax.set_ylabel(r'$I$, $10^{15}$ Вт/м$^2$')
    ax.set_xlim(0, t_end * 1e12)
    ax.margins(x=0)
    ax.tick_params(direction='in')

axes_int[-1].set_xlabel(r'Время, пс')
fig_intensity.suptitle('Интенсивность лазерного импульса', fontsize=13, fontweight='bold', y=0.98)
fig_intensity.subplots_adjust(top=0.92, hspace=0.28)


# ============================================================================
# График 2: Температура решетки на поверхности
# ============================================================================
fig_Tl = plot_stacked_quantity(
    'T_l',
    r'$T_\ell$, К',
    'Температура решетки при воздействии длинных импульсов',
    reference_line=T_m
)

# ============================================================================
# Сохранение
# ============================================================================
save_path = Path('graphs_long_pulse')
save_path.mkdir(parents=True, exist_ok=True)

fig_intensity.savefig(
    save_path / '01_Интенсивность_лазерного_импульса_long_pulse.png',
    dpi=300,
    bbox_inches='tight'
)

fig_Tl.savefig(
    save_path / '02_Температура_решетки_long_pulse.png',
    dpi=300,
    bbox_inches='tight'
)

print(f'Графики сохранены в: {save_path.resolve()}')

plt.show()