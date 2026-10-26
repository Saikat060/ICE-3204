import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

# ---------- Horn aperture size (wavelength-er unit-e) ----------
a = 5.0   # H-plane dimension (a / lambda)  -> baralei beam shoru hoy
b = 4.0   # E-plane dimension (b / lambda)  -> baralei beam shoru hoy

# ---------- Angle grid ----------
theta = np.linspace(0, np.pi, 300)        # 0 = boresight (+Z)
phi = np.linspace(0, 2 * np.pi, 300)
TH, PH = np.meshgrid(theta, phi)

# ---------- Radiation pattern ----------
# E-plane: sinc function (side lobe ekhan theke ashe)
E_plane = np.sinc(b * np.sin(TH) * np.sin(PH))

# H-plane: cosine-tapered aperture
u = np.pi * a * np.sin(TH) * np.cos(PH)
den = 1 - (2 * u / np.pi) ** 2
H_plane = np.where(np.abs(den) < 1e-6, np.pi / 4, np.cos(u) / np.where(np.abs(den) < 1e-6, 1, den))

# Obliquity factor: pichone radiation komay
obliquity = (1 + np.cos(TH)) / 2

F = np.abs(E_plane * H_plane * obliquity)
F /= F.max()                                # normalize

# ---------- Optional: dB scale (side lobe aro bhalo dekha jay) ----------
USE_DB = True
if USE_DB:
    dB_floor = -40
    F_db = 20 * np.log10(np.maximum(F, 1e-6))
    R = (np.maximum(F_db, dB_floor) - dB_floor) / (-dB_floor)   # 0..1
else:
    R = F

# ---------- Spherical -> Cartesian ----------
X = R * np.sin(TH) * np.cos(PH)
Y = R * np.sin(TH) * np.sin(PH)
Z = R * np.cos(TH)

# ---------- Plot ----------
fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection="3d")
colors = cm.cool(Normalize(R.min(), R.max())(R))
ax.plot_surface(X, Y, Z, facecolors=colors, rstride=2, cstride=2,
                linewidth=0, antialiased=True, shade=False)

ax.set_title("Horn Antenna 3D Radiation Pattern")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
lim = 1.0
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.set_zlim(-0.3, lim)
ax.set_box_aspect((1, 1, 1))
ax.view_init(elev=22, azim=-60)

plt.tight_layout()
plt.show()
