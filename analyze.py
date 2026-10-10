"""Generate example signals and plot their FFT spectra."""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

n = 2048
dt = 1e-4  # seconds
t = np.arange(n) * dt
f = np.fft.rfftfreq(n, dt)
phase = 2 * np.pi * 78.125 * t + 0.27

signals = {
    "sine": np.sin(phase),
    "square": np.where(np.sin(phase) >= 0, 1., -1.),
    "triangle": 2 / np.pi * np.arcsin(np.sin(phase)),
}
omega0 = 2 * np.pi * 781.25
for gamma in (50, 150):
    omega_d = np.sqrt(omega0**2 - gamma**2)
    signals[f"damped_{gamma}"] = np.exp(-gamma * t) * np.sin(omega_d * t)

data = Path(__file__).resolve().parent / "data"
data.mkdir(exist_ok=True)
fig, axes = plt.subplots(5, 2, figsize=(9, 10), constrained_layout=True)
for (name, signal), (time_axis, fft_axis) in zip(signals.items(), axes):
    np.savetxt(data / f"{name}.csv", np.column_stack((t * 1e6, signal)),
               delimiter=",", header="time_us,signal_synthetic", comments="", fmt="%.12g")
    spectrum = np.abs(np.fft.rfft(signal - signal.mean()))
    time_axis.plot(t[:384] * 1e3, signal[:384])
    time_axis.set(title=name, xlabel="Time (ms)", ylabel="Signal (a.u.)")
    fft_axis.plot(f, spectrum)
    fft_axis.set(xlim=(0, 1000), xlabel="Frequency (Hz)", ylabel="FFT modulus (a.u.)")

fig.savefig(data.parent / "spectra.png", dpi=140)
plt.show()
