from qttools import xp

from quatrex.electron.sse_coulomb_screening import (
    fft_convolve,
    fft_convolve_fock_with_hilbert,
    fft_correlate,
    hilbert_transform,
)


def test_sse(num_energies, batch_size):
    n = num_energies
    a = xp.zeros((n, batch_size), dtype=xp.complex128)
    b = xp.zeros_like(a)
    c = xp.zeros_like(a)
    d = xp.zeros_like(a)
    for i in range(batch_size):
        a[:, i] = xp.random.random(n) + 1j * xp.random.random(n)
        b[:, i] = xp.random.random(n) + 1j * xp.random.random(n)
        c[:, i] = xp.random.random(n) + 1j * xp.random.random(n)
        d[:, i] = xp.random.random(n) + 1j * xp.random.random(n)
    c[0, :] = 0
    d[0, :] = 0

    c1 = fft_convolve(a, c)[:n]
    c2 = fft_correlate(a, d.conj())[n - 1 :]
    e1 = c1 - c2

    c1 = fft_convolve(b, d)[:n]
    c2 = fft_correlate(b, c.conj())[n - 1 :]
    e2 = c1 - c2

    energies = xp.linspace(0, 1, n)
    f1, f2, f3 = fft_convolve_fock_with_hilbert(a, b, c, d, energies)

    sig = xp.imag(f2 - f1) * 1j
    e3 = hilbert_transform(sig, energies) * 1j + sig / 2

    assert xp.allclose(e1, f1)
    assert xp.allclose(e2, f2)
    assert xp.allclose(e3, f3)


if __name__ == "__main__":

    test_sse(num_energies=500, batch_size=10)
