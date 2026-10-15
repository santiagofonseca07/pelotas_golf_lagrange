"""
App interactiva: ¿Cuántas pelotas producir y cuánta publicidad pagar?
Problema Pro-T resuelto con multiplicadores de Lagrange.

Ejecutar:  streamlit run app.py
"""
import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Pro-T · Ganancia óptima", page_icon="⛳", layout="wide")

# ---------------- Modelo ----------------
def ganancia(x, y):
    return 48 * x + 96 * y - x**2 - 2 * x * y - 9 * y**2


def gradiente(x, y):
    return np.array([48 - 2 * x - 2 * y, 96 - 2 * x - 18 * y])


def resolver_lagrange(a, b, B):
    """Resuelve ∇f = λ∇g, a·x + b·y = B  (sistema lineal 3x3)."""
    A = np.array([[-2, -2, -a],
                  [-2, -18, -b],
                  [a, b, 0]], dtype=float)
    rhs = np.array([-48, -96, B], dtype=float)
    x, y, lam = np.linalg.solve(A, rhs)
    return x, y, lam


def mejor_factible(a, b, B):
    """Si Lagrange da cantidades negativas, se toma el mejor extremo del segmento."""
    x, y, lam = resolver_lagrange(a, b, B)
    ajustado = False
    if x < 0 or y < 0:
        ajustado = True
        x = np.clip(x, 0, B / a)
        y = (B - a * x) / b
    return x, y, lam, ajustado


# ---------------- Barra lateral ----------------
st.sidebar.header("⚙️ Ajusta tu escenario")
st.sidebar.caption("Valores del enunciado: 20, 4 y 216.")
a = st.sidebar.slider("Costo de producir 1.000 pelotas (miles USD)", 5.0, 40.0, 20.0, 0.5)
b = st.sidebar.slider("Costo de 1 hora de publicidad (miles USD)", 1.0, 20.0, 4.0, 0.5)
B = st.sidebar.slider("Presupuesto mensual total (miles USD)", 50.0, 600.0, 216.0, 1.0)
if st.sidebar.button("↺ Volver al enunciado"):
    st.session_state.clear()
    st.rerun()

X_LIB, Y_LIB = 21.0, 3.0          # máximo sin restricción (∇f = 0)
Z_LIB = ganancia(X_LIB, Y_LIB)
costo_libre = a * X_LIB + b * Y_LIB
x_opt, y_opt, lam, ajustado = mejor_factible(a, b, B)
sobra = costo_libre <= B          # el presupuesto ya no limita (λ = 0)
if sobra:
    x_opt, y_opt, lam, ajustado = X_LIB, Y_LIB, 0.0, False
z_opt = ganancia(x_opt, y_opt)

# ---------------- Encabezado y respuesta ----------------
st.title("⛳ Pro-T: ¿cuánto producir y cuánta publicidad pagar?")
st.write(
    "Esta herramienta encuentra la combinación de **pelotas vendidas** y **horas de publicidad** "
    "que deja **la mayor ganancia posible sin pasarse del presupuesto**."
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Pelotas al mes", f"{x_opt * 1000:,.0f}")
c2.metric("Horas de publicidad al mes", f"{y_opt:,.2f}")
c3.metric("Ganancia máxima", f"US$ {z_opt * 1000:,.0f}")
c4.metric("Valor de 1 dólar extra de presupuesto", f"US$ {lam:,.2f}",
          help="Es el multiplicador λ: cuánto sube la ganancia si el presupuesto aumenta en 1.")

if ajustado:
    st.warning("Con estos costos la solución ideal pedía cantidades negativas; se muestra la mejor opción realista "
               "(sin producir o sin publicidad).")
elif sobra:
    st.info(f"💡 El presupuesto alcanza de sobra: la mejor opción cuesta {costo_libre:,.0f} mil USD y te "
            f"sobran {B - costo_libre:,.0f} mil. Gastar más **reduce** la ganancia, por eso un dólar extra vale 0.")
elif lam > 0:
    st.success(f"El presupuesto es el factor que limita: cada **US$ 1.000** adicionales subirían la ganancia "
               f"en unos **US$ {lam * 1000:,.0f}**.")

# ---------------- Explorador ----------------
st.subheader("🔎 Prueba tú mismo")
st.write("Mueve el control para repartir el presupuesto. La app calcula las horas de publicidad que alcanzan "
         "con el dinero restante y te dice cuánto ganarías.")
x_max = B / a
x_ini = float(round(resolver_lagrange(a, b, B)[0] if sobra else x_opt, 2))
x_user = st.slider("Miles de pelotas a producir", 0.0, float(round(x_max, 2)),
                   float(np.clip(x_ini, 0, round(x_max, 2))), 0.1)
y_user = (B - a * x_user) / b
z_user = ganancia(x_user, y_user)
d1, d2, d3 = st.columns(3)
d1.metric("Horas de publicidad que alcanzan", f"{y_user:,.2f}")
d2.metric("Ganancia con tu elección", f"US$ {z_user * 1000:,.0f}")
d3.metric("Diferencia vs. el óptimo", f"US$ {(z_user - z_opt) * 1000:,.0f}")

xs_line = np.linspace(0, x_max, 300)
fig_line = go.Figure()
fig_line.add_scatter(x=xs_line, y=ganancia(xs_line, (B - a * xs_line) / b), mode="lines",
                     name="Ganancia gastando todo el presupuesto", line=dict(color="#e45756", width=3))
fig_line.add_scatter(x=[x_opt], y=[z_opt], mode="markers", name="Óptimo",
                     marker=dict(size=14, color="gold", line=dict(color="black", width=1)))
fig_line.add_scatter(x=[x_user], y=[z_user], mode="markers", name="Tu elección",
                     marker=dict(size=12, color="#4c78a8", symbol="diamond"))
fig_line.update_layout(xaxis_title="Miles de pelotas", yaxis_title="Ganancia (miles USD)", height=330,
                       margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.25))
st.plotly_chart(fig_line, width="stretch")

# ---------------- Gráficas 2D y 3D ----------------
xr = (0, max(25, x_opt + 5, x_user + 5))
yr = (0, max(10, y_opt + 3))
gx = np.linspace(*xr, 160)
gy = np.linspace(*yr, 160)
GX, GY = np.meshgrid(gx, gy)
GZ = ganancia(GX, GY)
ys_line = (B - a * xs_line) / b
mask = (ys_line >= yr[0]) & (ys_line <= yr[1])

tab2d, tab3d = st.tabs(["🗺️ Mapa 2D (isolíneas)", "⛰️ Montaña 3D"])

with tab2d:
    st.caption("Cada línea une puntos con **la misma ganancia** (como curvas de nivel en un mapa). "
               "Las flechas señalan hacia dónde crece la ganancia. La línea roja es todo lo que puedes pagar.")
    mostrar_flechas = st.checkbox("Mostrar flechas (gradiente)", value=True)
    fig2 = go.Figure()
    fig2.add_contour(x=gx, y=gy, z=GZ, colorscale="Viridis", ncontours=25,
                     contours=dict(showlabels=True, labelfont=dict(size=10, color="white")),
                     colorbar=dict(title="Ganancia<br>(miles USD)"), name="Ganancia")
    if mostrar_flechas:
        qx, qy = np.meshgrid(np.linspace(xr[0] + 1, xr[1] - 1, 13), np.linspace(yr[0] + 0.5, yr[1] - 0.5, 9))
        G = gradiente(qx, qy)
        n = np.hypot(*G); n[n == 0] = 1
        sx, sy = 0.045 * (xr[1] - xr[0]), 0.045 * (yr[1] - yr[0])
        for x0, y0, ux, uy in zip(qx.ravel(), qy.ravel(), (G[0] / n).ravel(), (G[1] / n).ravel()):
            fig2.add_annotation(x=x0 + ux * sx, y=y0 + uy * sy, ax=x0, ay=y0, xref="x", yref="y",
                                axref="x", ayref="y", showarrow=True, arrowhead=2, arrowsize=1,
                                arrowwidth=1.2, arrowcolor="rgba(0,0,0,0.55)", text="")
    fig2.add_scatter(x=xs_line[mask], y=ys_line[mask], mode="lines", name=f"Presupuesto {a:g}x + {b:g}y = {B:g}",
                     line=dict(color="red", width=3))
    fig2.add_scatter(x=[x_opt], y=[y_opt], mode="markers", name="Máximo con presupuesto",
                     marker=dict(size=15, color="gold", line=dict(color="black", width=1.5)))
    fig2.add_scatter(x=[X_LIB], y=[Y_LIB], mode="markers", name="Máximo sin límite de dinero",
                     marker=dict(size=17, color="white", symbol="star", line=dict(color="black", width=1.5)))
    fig2.add_scatter(x=[x_user], y=[y_user], mode="markers", name="Tu elección",
                     marker=dict(size=12, color="#4c78a8", symbol="diamond"))
    fig2.update_layout(xaxis_title="Miles de pelotas al mes", yaxis_title="Horas de publicidad al mes",
                       xaxis_range=xr, yaxis_range=yr, height=560, margin=dict(l=10, r=10, t=10, b=10),
                       legend=dict(orientation="h", y=-0.15))
    st.plotly_chart(fig2, width="stretch")

with tab3d:
    st.caption("La ganancia es una **montaña**: la cima (estrella) es lo máximo posible. La curva roja es el "
               "camino que permite el presupuesto; el punto dorado es lo más alto de ese camino. "
               "Arrastra para girar.")
    fig3 = go.Figure()
    fig3.add_surface(x=gx, y=gy, z=GZ, colorscale="Viridis", opacity=0.85, showscale=False,
                     contours=dict(z=dict(show=True, usecolormap=True, project_z=True, highlightwidth=2)))
    fig3.add_scatter3d(x=xs_line[mask], y=ys_line[mask], z=ganancia(xs_line[mask], ys_line[mask]),
                       mode="lines", line=dict(color="red", width=8), name="Camino del presupuesto")
    fig3.add_scatter3d(x=[x_opt], y=[y_opt], z=[z_opt], mode="markers", name="Máximo con presupuesto",
                       marker=dict(size=8, color="gold", line=dict(color="black", width=2)))
    fig3.add_scatter3d(x=[X_LIB], y=[Y_LIB], z=[Z_LIB], mode="markers", name="Cima (sin límite)",
                       marker=dict(size=8, color="white", symbol="diamond", line=dict(color="black", width=2)))
    fig3.update_layout(scene=dict(xaxis_title="Miles de pelotas", yaxis_title="Horas publicidad",
                                  zaxis_title="Ganancia (miles USD)"),
                       height=620, margin=dict(l=0, r=0, t=0, b=0), legend=dict(orientation="h"))
    st.plotly_chart(fig3, width="stretch")

# ---------------- Explicación ----------------
with st.expander("📘 ¿Cómo se calcula? (método de Lagrange, paso a paso)"):
    st.markdown("**Modelo de ganancia** (miles de USD):")
    st.latex(r"f(x,y)=48x+96y-x^2-2xy-9y^2")
    st.markdown("**Restricción de presupuesto:**")
    st.latex(rf"g(x,y)= {a:g}x + {b:g}y = {B:g}")
    st.markdown("**Idea:** en el mejor punto, la línea de presupuesto *toca* una curva de ganancia sin cruzarla. "
                "Ahí las dos flechas (gradientes) apuntan en la misma dirección:")
    st.latex(r"\nabla f = \lambda \nabla g")
    st.latex(rf"""\begin{{cases}}
48-2x-2y = {a:g}\lambda \\
96-2x-18y = {b:g}\lambda \\
{a:g}x+{b:g}y = {B:g}
\end{{cases}}""")
    st.markdown("**Solución:**")
    st.latex(rf"x={x_opt:.3f},\quad y={y_opt:.3f},\quad \lambda={lam:.3f},\quad f={z_opt:.2f}")
    st.markdown(
        "**¿Es máximo?** Sobre la restricción la ganancia es una parábola que abre hacia abajo "
        "(ver gráfica de *Prueba tú mismo*), así que el punto encontrado es el máximo. "
        "El **mínimo** sobre la línea está en un extremo del presupuesto (todo en pelotas o todo en publicidad). "
        "Sin restricción, el máximo absoluto es (21, 3) con 648 mil USD, y no existe mínimo global "
        "porque la ganancia baja sin límite al alejarse."
    )
