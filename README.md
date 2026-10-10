# Lab2-FFT

Fourier analysis of periodic waveforms and a damped RLC oscillation, based on
[Francesco Fuso's Laboratory 2 exercise](https://osiris.df.unipi.it/~fuso/dida/fft_2016.pdf).

![Waveforms and spectra](results/waveforms.png)

## Run

Python 3.10 or newer:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python demo.py
python build_report.py
python -m unittest discover -s tests -v
```

## Analyse a recording

Input: two columns, time and signal. Time defaults to microseconds; signal stays
in its input units (Arduino digits or volts). Whitespace files may contain `#`
comments. For a CSV with one header row:

```bash
python analyze.py data/square.csv --delimiter , --skiprows 1
python analyze.py your_recording.txt --time-unit us --window hann
```

Outputs: `results/recording/spectrum.png`, `spectrum.csv`, and `summary.txt`.
Use `--keep-mean` to retain the DC component and `--output PATH` to change the
destination. If timestamps deviate by more than 1% from the mean interval, the
script requires `--allow-jitter`; this accepts a uniform-sampling approximation,
not a correction. The threshold is a practical flag, not an accuracy guarantee.

## Material

- [Solution and physical interpretation](solution.md)
- [Short PDF report](report.pdf)
- `analyze.py`: reusable FFT analysis with correct one-sided amplitude scaling.
- `demo.py`: deterministic data generation and comparison with theory.
- `build_report.py`: regenerates the report from the results.
- `data/`: synthetic time records, with microsecond timestamps.
- `results/`: figures, harmonic ratios, and damped spectra.
- `tests/`: numerical and input-validation checks.

The main results are the odd-harmonic decay laws, **1/h** for a square wave and
**1/h²** for a triangular wave, and the lower, broader spectrum of a more strongly
damped impulse response. [Damping comparison](results/damping.png) and
[finite-record leakage](results/leakage.png) illustrate the remaining cases.
