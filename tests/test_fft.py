"""Checks of normalization, sampling limits, harmonics, and the RLC reference."""

import unittest
import warnings
import numpy as np
from analyze import spectrum
from demo import damped_transform


class FFTTests(unittest.TestCase):
    def test_sine_amplitude_and_frequency_even_and_odd(self):
        for n in (1024, 999):
            t = np.arange(n)*0.001
            signal = 4 + 2.5*np.sin(2*np.pi*17*np.arange(n)/n)
            f, amplitude, _, _ = spectrum(t, signal)
            self.assertEqual(np.argmax(amplitude), 17)
            self.assertAlmostEqual(f[17], 17/(n*0.001))
            self.assertAlmostEqual(amplitude[17], 2.5)
            self.assertLess(amplitude[0], 1e-12)

    def test_dc_and_nyquist_are_not_doubled(self):
        t = np.arange(128)*0.01
        _, amplitude, _, _ = spectrum(t, 3 + 2*(-1.)**np.arange(128), remove_mean=False)
        self.assertAlmostEqual(amplitude[0], 3)
        self.assertAlmostEqual(amplitude[-1], 2)

    def test_hann_gain(self):
        t = np.arange(2048)*0.001
        _, amplitude, _, _ = spectrum(t, 1.7*np.sin(2*np.pi*100*np.arange(2048)/2048), "hann")
        self.assertAlmostEqual(amplitude[100], 1.7, places=6)

    def test_harmonic_decay(self):
        n = 16384
        t = np.arange(n)*0.001
        phase = 2*np.pi*16*np.arange(n)/n + 0.27
        for values, power in ((np.where(np.sin(phase)>=0, 1., -1.), 1),
                              (2/np.pi*np.arcsin(np.sin(phase)), 2)):
            _, amplitude, _, _ = spectrum(t, values)
            for h in (3, 5, 7):
                self.assertAlmostEqual(amplitude[16*h]/amplitude[16], 1/h**power, delta=2e-5)
            self.assertLess(amplitude[32]/amplitude[16], 1e-12)

    def test_damped_fft_against_geometric_sum(self):
        n, dt, gamma, omega_d = 2048, 1e-4, 50., 4900.
        t = np.arange(n)*dt
        f, _, transform, _ = spectrum(t, np.exp(-gamma*t)*np.sin(omega_d*t), remove_mean=False)
        np.testing.assert_allclose(transform, damped_transform(f, dt, n, gamma, omega_d), atol=1e-10)

    def test_invalid_records(self):
        for t, values in (([0, 1, 1, 2], [1]*4), ([0, 1, 2, 3], [1, np.nan, 1, 1]),
                          ([0, 1, 2], [1]*3), ([0, 1, 2, 3], [1]*5)):
            with self.assertRaises(ValueError):
                spectrum(t, values)

    def test_nonuniform_sampling_requires_opt_in(self):
        t = np.array([0., 1., 2.1, 3.])
        with self.assertRaises(ValueError):
            spectrum(t, np.ones(4))
        with warnings.catch_warnings(record=True) as emitted:
            warnings.simplefilter("always")
            spectrum(t, np.ones(4), allow_jitter=True)
        self.assertEqual(len(emitted), 1)


if __name__ == "__main__":
    unittest.main()
