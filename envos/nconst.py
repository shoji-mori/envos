#### Constant Physical parameter in cgs-gauss ####
pi = 3.141592653589793238462643383279502884197169399375105820974
c = 2.99792458e10 # cm/s
h = 6.62607004e-27 # erg s
G = 6.67408e-8 # cm3 g-1 s-2
e = 4.803205e-10 # esu
m_e = me = 9.10938291e-28 # g
# m_n      = mn = 3.88549512e-24
sigma_SB = 5.670367e-5 # erg cm-2 s-1 K-4
a = 4 * sigma_SB / c # erg cm-3 K-4
k_B = kB = k = 1.38065e-16 # erg K-1
amu = 1.660468e-24 # g
sigma_en = 1e-15 # cm^2
AU = au = 1.49597870700e13 # cm
pc = 3.0856775814913674e18 # cm
m = 100 # cm
Angstrom = 1e-8 # cm
inch = 2.54 # cm
bar = 1e6 # dyne cm-2
eV = 1.60217662e-12  # erg
J = 1e7  # erg
Msun = 1.988435e33 # g
# Lsun: historical value used throughout envos (3.848e33 erg/s).
# IAU 2015 nominal value is 3.828e33 erg/s; this value differs from IAU.
# Changing Lsun is a ⚠ physics-change; left as-is per PLAN.md D7.
Lsun = 3.848e33 # erg/s
Rsun = 6.957e10  # cm = 0.00465 au
Tsun = 5772  # K
Rj = 6.9911e9  # cm
Mj = 1.89813e30  # g
year = yr = 3.15576e7  # s  Julian year (365.25 days); IAU standard
Myr = 3.15576e13  # s
Msun_per_yr = smpy = Msun / year # g/s
Mj_p_Myr = Mj / (1e6 * year) # g/s
# gamma    = 7.0/5.0
# cV       = kB/mn/(gamma-1) # heat capacity per mass
# R       = 8.31446261815324e7
# CV      = 5/2*R
kms = 1e5 # cm/s
au2pc = au / pc 
deg2rad = pi / 180 
rad2deg = 180 / pi

if __name__ == "__main__":
    import nconst
    print(vars(nconst))
