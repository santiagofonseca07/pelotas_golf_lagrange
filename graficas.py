"""Gráficas 2D y 3D del problema Pro-T (multiplicadores de Lagrange)."""
import numpy as np
import matplotlib.pyplot as plt

f = lambda x, y: 48*x + 96*y - x**2 - 2*x*y - 9*y**2
grad = lambda x, y: (48 - 2*x - 2*y, 96 - 2*x - 18*y)
g_y = lambda x: (216 - 20*x) / 4          # restricción: 20x + 4y = 216

X_OPT, Y_OPT, Z_OPT = 10, 4, 540          # máximo con restricción (λ = 1)
X_LIB, Y_LIB, Z_LIB = 21, 3, 648          # máximo libre (sin restricción)

x = np.linspace(0, 25, 300)
y = np.linspace(0, 10, 300)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

# ---------- 2D: isolíneas + gradientes + restricción ----------
fig, ax = plt.subplots(figsize=(10, 7))
cf = ax.contourf(X, Y, Z, levels=30, cmap="viridis", alpha=0.85)
cs = ax.contour(X, Y, Z, levels=[300, 400, 500, 540, 600, 640], colors="white", linewidths=0.8)
ax.clabel(cs, fmt="%d", fontsize=8)
fig.colorbar(cf, label="Ganancia z (miles USD)")

xs = np.linspace(0, 10.8, 100)
ax.plot(xs, g_y(xs), "r-", lw=2.5, label="Restricción 20x + 4y = 216")

xq, yq = np.meshgrid(np.linspace(1, 24, 14), np.linspace(0.5, 9.5, 10))
gx, gy = grad(xq, yq)
n = np.hypot(gx, gy); n[n == 0] = 1
ax.quiver(xq, yq, gx/n, gy/n, color="black", alpha=0.55, scale=30, width=0.003, label="∇f (dirección de ascenso)")

# ∇f y ∇g en el óptimo: paralelos (∇f = λ∇g)
ax.quiver(X_OPT, Y_OPT, *grad(X_OPT, Y_OPT), color="orange", scale=60, width=0.009, label="∇f(10,4) = (20, 4)")
ax.quiver(X_OPT, Y_OPT, 20, 4, color="red", scale=60, width=0.004, label="∇g = (20, 4)  → ∇f = 1·∇g")

ax.plot(X_OPT, Y_OPT, "o", ms=12, mfc="gold", mec="k", label=f"Máx. con restricción ({X_OPT},{Y_OPT}) → z={Z_OPT}")
ax.plot(X_LIB, Y_LIB, "*", ms=16, mfc="white", mec="k", label=f"Máx. libre ({X_LIB},{Y_LIB}) → z={Z_LIB}")
ax.set(xlim=(0, 25), ylim=(0, 10), xlabel="x: miles de pelotas/mes", ylabel="y: horas de publicidad/mes",
       title="Isolíneas de ganancia, gradientes y restricción de presupuesto")
ax.legend(loc="upper right", fontsize=8)
fig.tight_layout(); fig.savefig("grafica_2d.png", dpi=150)

# ---------- 3D: superficie + curva restringida ----------
xs3 = np.linspace(8.8, 10.8, 100)   # tramo de la restricción con 0 ≤ y ≤ 10
fig = plt.figure(figsize=(11, 8))
ax = fig.add_subplot(projection="3d")
ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.7, linewidth=0)
ax.contour(X, Y, Z, levels=20, zdir="z", offset=Z.min(), cmap="viridis")  # isolíneas proyectadas
ax.plot(xs3, g_y(xs3), f(xs3, g_y(xs3)), "r-", lw=3, label="Ganancia sobre la restricción")
ax.plot(xs3, g_y(xs3), np.full_like(xs3, Z.min()), "r--", lw=1.5)
ax.scatter(X_OPT, Y_OPT, Z_OPT, s=120, c="gold", edgecolors="k", depthshade=False, label=f"Máx. restringido z={Z_OPT}")
ax.scatter(X_LIB, Y_LIB, Z_LIB, s=160, c="white", marker="*", edgecolors="k", depthshade=False, label=f"Máx. libre z={Z_LIB}")
ax.set(xlabel="x (miles de pelotas)", ylabel="y (horas publicidad)", zlabel="z (miles USD)",
       title="Superficie de ganancia f(x, y)")
ax.view_init(elev=28, azim=-125); ax.legend(loc="upper left")
fig.tight_layout(); fig.savefig("grafica_3d.png", dpi=150)
print("ok")
