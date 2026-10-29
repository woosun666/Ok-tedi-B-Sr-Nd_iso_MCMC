"""Forward model: concentration-weighted mixing of K end-members.

For mass fractions f_i (sum 1) and end-member element concentration C_i and
isotope value R_i (delta, ratio or epsilon), the mixture is

    C_mix = sum_i f_i C_i
    R_mix = sum_i f_i C_i R_i / C_mix

This is exact for atomic ratios with a common denominator isotope and a very
good approximation for delta / epsilon notation (differences << 1000 permil).
Each isotope system is mixed independently with its own element.
"""
import numpy as np

# isotope system -> (concentration column, isotope column)
SYSTEMS = {
    "B":  ("B_ppm", "d11B"),
    "Sr": ("Sr_ppm", "Sr87_86"),
    "Nd": ("Nd_ppm", "eNd"),
}


def mix(f, conc, iso):
    """f: (K,), conc/iso: (S, K) for S systems -> (C_mix, R_mix), each (S,)."""
    w = conc * f                       # (S, K) element mass from each member
    c_mix = w.sum(axis=1)
    return c_mix, (w * iso).sum(axis=1) / c_mix
