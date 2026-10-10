"""FFT of two-column time/signal records; time is converted to seconds."""

import argparse
from pathlib import Path
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def spectrum(t, signal, window="none", remove_mean=True, allow_jitter=False):
    """Return frequency, one-sided amplitude, complex DFT, and sampling info.

    Amplitudes recover an isolated, bin-centred sinusoid's peak amplitude.
    They are not a power spectral density. No resampling is performed.
    """
    t, signal = np.asarray(t, dtype=float), np.asarray(signal, dtype=float)
    if t.ndim != 1 or signal.shape != t.shape or t.size < 4:
        raise ValueError("Need at least four matching time/signal samples.")
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(signal)):
        raise ValueError("Samples must be finite.")
    steps = np.diff(t)
    if np.any(steps <= 0):
        raise ValueError("Timestamps must be strictly increasing.")
    dt = (t[-1] - t[0]) / (t.size - 1)
    jitter = np.max(np.abs(steps - dt)) / dt
    if jitter > 0.01:
        message = f"Maximum spacing deviation is {100*jitter:.2f}%; FFT assumes uniform sampling."
        if not allow_jitter:
            raise ValueError(message + " Use --allow-jitter to accept the approximation.")
        warnings.warn(message, stacklevel=2)
    if window not in ("none", "hann"):
        raise ValueError("Window must be none or hann.")
    weights = np.ones(t.size) if window == "none" else np.hanning(t.size)
    values = signal - signal.mean() if remove_mean else signal.copy()
    transform = np.fft.rfft(values * weights)
    amplitude = np.abs(transform) / weights.sum()
    amplitude[1:] *= 2
    if t.size % 2 == 0:
        amplitude[-1] /= 2  # Nyquist has no distinct negative-frequency partner.
    frequency = np.fft.rfftfreq(t.size, dt)
    info = dict(n=t.size, dt=dt, fs=1/dt, duration=t.size*dt,
                df=1/(t.size*dt), nyquist=1/(2*dt), jitter=jitter)
    return frequency, amplitude, transform, info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", type=Path)
    parser.add_argument("--time-unit", choices=("s", "ms", "us"), default="us")
    parser.add_argument("--delimiter", default=None, help="Default: whitespace. Use ',' for CSV.")
    parser.add_argument("--skiprows", type=int, default=0)
    parser.add_argument("--window", choices=("none", "hann"), default="none")
    parser.add_argument("--keep-mean", action="store_true")
    parser.add_argument("--allow-jitter", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("results/recording"))
    args = parser.parse_args()
    try:
        data = np.loadtxt(args.file, delimiter=args.delimiter, skiprows=args.skiprows, ndmin=2)
        if data.shape[1] != 2:
            raise ValueError("Expected exactly two columns: time and signal.")
        t = data[:, 0] * {"s": 1, "ms": 1e-3, "us": 1e-6}[args.time_unit]
        f, amplitude, transform, info = spectrum(
            t, data[:, 1], args.window, not args.keep_mean, args.allow_jitter)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    args.output.mkdir(parents=True, exist_ok=True)
    np.savetxt(args.output / "spectrum.csv",
               np.column_stack((f, amplitude, transform.real, transform.imag)),
               delimiter=",", header="frequency_hz,amplitude,fft_real,fft_imag", comments="")
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), constrained_layout=True)
    axes[0].plot((t-t[0])*1e3, data[:, 1], lw=1)
    axes[0].set(xlabel="Time (ms)", ylabel="Signal (input units)")
    axes[1].plot(f, amplitude, lw=1)
    axes[1].set(xlabel="Frequency (Hz)", ylabel="One-sided amplitude (input units)")
    fig.savefig(args.output / "spectrum.png", dpi=150)
    plt.close(fig)
    summary = "\n".join(f"{name} = {value:.10g}" for name, value in info.items())
    peak = np.argmax(amplitude[1:]) + 1
    summary += f"\nLargest non-DC bin: {f[peak]:.10g} Hz\n"
    (args.output / "summary.txt").write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
