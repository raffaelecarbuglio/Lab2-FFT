"""Generate explicitly synthetic exercise data, figures, and numerical results."""

from pathlib import Path
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze import spectrum

ROOT = Path(__file__).resolve().parent


def damped_transform(f, dt, n, gamma, omega_d):
    """Exact DFT of exp(-gamma*t)*sin(omega_d*t), t=j*dt.

    Finite geometric sums provide an independent reference to the FFT.
    """
    z_plus = (-gamma + 1j*omega_d - 2j*np.pi*f) * dt
    z_minus = (-gamma - 1j*omega_d - 2j*np.pi*f) * dt
    return (np.expm1(n*z_plus)/np.expm1(z_plus)
            - np.expm1(n*z_minus)/np.expm1(z_minus)) / (2j)


def save_record(name, t, signal):
    np.savetxt(ROOT / "data" / f"{name}.csv", np.column_stack((t*1e6, signal)),
               delimiter=",", header="time_us,signal_synthetic", comments="", fmt="%.12g")


def main():
    for directory in ("data", "results"):
        (ROOT / directory).mkdir(exist_ok=True)
    n, dt, fundamental_bin = 2048, 1e-4, 16
    t = np.arange(n)*dt
    f0 = fundamental_bin/(n*dt)
    phase = 2*np.pi*f0*t + 0.27
    waves = dict(sine=np.sin(phase), square=np.where(np.sin(phase) >= 0, 1., -1.),
                 triangle=2/np.pi*np.arcsin(np.sin(phase)))
    fig, axes = plt.subplots(3, 2, figsize=(9, 7), constrained_layout=True)
    rows = []
    for row, (name, signal) in enumerate(waves.items()):
        save_record(name, t, signal)
        f, amplitude, _, _ = spectrum(t, signal)
        axes[row, 0].plot(t[:384]*1e3, signal[:384], lw=1)
        axes[row, 0].set(title=f"{name.capitalize()} - synthetic", ylabel="Signal (a.u.)")
        axes[row, 1].stem(f[:161], amplitude[:161], basefmt=" ")
        axes[row, 1].set(title="One-sided amplitude", ylabel="Amplitude (a.u.)")
        for harmonic in (1, 3, 5, 7, 9):
            expected = (1/harmonic if name == "square" else 1/harmonic**2
                        if name == "triangle" else float(harmonic == 1))
            rows.append((name, harmonic, harmonic*f0,
                         amplitude[harmonic*fundamental_bin]/amplitude[fundamental_bin], expected))
    for axis in axes[:, 0]:
        axis.set_xlabel("Time (ms)")
    for axis in axes[:, 1]:
        axis.set_xlabel("Frequency (Hz)")
    fig.savefig(ROOT / "results/waveforms.png", dpi=140)
    plt.close(fig)
    with (ROOT / "results/harmonics.csv").open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("waveform", "harmonic", "frequency_hz", "fft_ratio", "continuum_ratio"))
        writer.writerows(rows)

    natural_f = 781.25
    omega0 = 2*np.pi*natural_f
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), constrained_layout=True)
    damping_rows = []
    for gamma in (50., 150.):
        omega_d = np.sqrt(omega0**2 - gamma**2)
        signal = np.exp(-gamma*t)*np.sin(omega_d*t)
        save_record(f"damped_{int(gamma)}", t, signal)
        f, _, transform, _ = spectrum(t, signal, remove_mean=False)
        exact = damped_transform(f, dt, n, gamma, omega_d)
        continuous = omega_d / ((gamma+2j*np.pi*f)**2 + omega_d**2)
        label = f"gamma = {gamma:g} s^-1"
        axes[0].plot(t[:1000]*1e3, signal[:1000], lw=1, label=label)
        mask = (f >= 650) & (f <= 900)
        axes[1].plot(f[mask], dt*np.abs(transform[mask]), "o", ms=3, label=label)
        dense = np.linspace(650, 900, 1500)
        axes[1].plot(dense, np.abs(omega_d/((gamma+2j*np.pi*dense)**2+omega_d**2)), lw=1)
        peak = np.argmax(np.abs(transform))
        error = np.max(np.abs(transform-exact)) / np.max(np.abs(exact))
        damping_rows.append((gamma, omega_d/(2*np.pi), f[peak],
                             np.sqrt(omega0**2-2*gamma**2)/(2*np.pi), gamma/np.pi, error))
        np.savetxt(ROOT / "results" / f"damped_{int(gamma)}_spectrum.csv",
                   np.column_stack((f, dt*np.abs(transform), dt*np.abs(exact), np.abs(continuous))),
                   delimiter=",", header="frequency_hz,fft_modulus_s,exact_discrete_modulus_s,continuous_modulus_s",
                   comments="")
    axes[0].set(xlabel="Time (ms)", ylabel="Signal (a.u.)", title="Damped oscillations - synthetic")
    axes[1].set(xlabel="Frequency (Hz)", ylabel="Transform modulus (a.u. s)",
                title="FFT points / continuous impulse-response curves")
    for axis in axes:
        axis.legend(fontsize=8)
    fig.savefig(ROOT / "results/damping.png", dpi=140)
    plt.close(fig)
    with (ROOT / "results/damping.csv").open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("gamma_per_s", "damped_frequency_hz", "fft_peak_bin_hz",
                         "continuous_peak_hz", "approx_power_fwhm_hz", "max_relative_dft_error"))
        writer.writerows(damping_rows)

    # A non-integer number of cycles illustrates leakage; Hann suppresses sidelobes.
    off_bin_f = 16.4/(n*dt)
    signal = np.sin(2*np.pi*off_bin_f*t)
    save_record("off_bin_sine", t, signal)
    fig, axis = plt.subplots(figsize=(8, 3), constrained_layout=True)
    for window in ("none", "hann"):
        f, amplitude, _, _ = spectrum(t, signal, window=window)
        axis.semilogy(f, np.maximum(amplitude, 1e-12), label=window)
    axis.set(xlim=(0, 300), ylim=(1e-6, 2), xlabel="Frequency (Hz)",
             ylabel="One-sided amplitude (a.u.)", title="Off-bin sine - synthetic")
    axis.legend()
    fig.savefig(ROOT / "results/leakage.png", dpi=140)
    plt.close(fig)
    print(f"N={n}; fs={1/dt:g} Hz; df={1/(n*dt):g} Hz; f0={f0:g} Hz")
    print("Generated synthetic data and results in data/ and results/.")


if __name__ == "__main__":
    main()
