import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Na Dendrite Phase Field", layout="wide")
st.title("Phase-field simulation of Na dendrite growth in solid electrolyte")
st.markdown("Kobayashi-type phase-field model with 4-fold anisotropy coupled to a diffusion field.")


# ---------------- numerical helpers ----------------
def laplacian(f, dx):
    lap = np.zeros_like(f)
    lap[1:-1, 1:-1] = (
        f[0:-2, 1:-1] + f[2:, 1:-1] + f[1:-1, 0:-2] + f[1:-1, 2:]
        - 4 * f[1:-1, 1:-1]
    ) / (dx * dx)
    return lap


def grad_x(f, dx):
    g = np.zeros_like(f)
    g[1:-1, 1:-1] = (f[1:-1, 2:] - f[1:-1, 0:-2]) / (2 * dx)
    return g


def grad_y(f, dx):
    g = np.zeros_like(f)
    g[1:-1, 1:-1] = (f[2:, 1:-1] - f[0:-2, 1:-1]) / (2 * dx)
    return g


def run_simulation(nx, ny, nsteps, dt, dx, eps_bar, tau, delta,
                   alpha, gamma, undercooling, D, K, noise, progress):
    # phase field p: 0 = electrolyte/liquid, 1 = solid Na
    p = np.zeros((ny, nx))
    u = np.full((ny, nx), -undercooling)  # dimensionless supersaturation field

    # seed: small solid circle in the centre
    cy, cx = ny // 2, nx // 2
    yy, xx = np.ogrid[:ny, :nx]
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    p[r < 4] = 1.0

    save_every = max(1, nsteps // 6)
    frames = []

    for step in range(nsteps):
        px = grad_x(p, dx)
        py = grad_y(p, dx)

        theta = np.arctan2(py, px)
        eps = eps_bar * (1.0 + delta * np.cos(4.0 * theta))
        eps2 = eps * eps
        eps_eps_prime = (
            -4.0 * eps_bar ** 2 * delta
            * (1.0 + delta * np.cos(4.0 * theta))
            * np.sin(4.0 * theta)
        )

        # div(eps^2 grad p)
        div_eps2_gradp = (
            (eps2[1:-1, 2:] * px[1:-1, 2:] - eps2[1:-1, 0:-2] * px[1:-1, 0:-2]) / (2 * dx)
            + (eps2[2:, 1:-1] * py[2:, 1:-1] - eps2[0:-2, 1:-1] * py[0:-2, 1:-1]) / (2 * dx)
        )

        # anisotropic correction: -d/dx(eps*eps'*py) + d/dy(eps*eps'*px)
        corr = (
            -(eps_eps_prime[1:-1, 2:] * py[1:-1, 2:]
              - eps_eps_prime[1:-1, 0:-2] * py[1:-1, 0:-2]) / (2 * dx)
            + (eps_eps_prime[2:, 1:-1] * px[2:, 1:-1]
               - eps_eps_prime[0:-2, 1:-1] * px[0:-2, 1:-1]) / (2 * dx)
        )

        m = (alpha / np.pi) * np.arctan(gamma * undercooling)
        interior = p[1:-1, 1:-1]

        dpdt = (
            div_eps2_gradp
            + interior * (1 - interior) * (interior - 0.5 + m)
            + corr
        ) / tau

        p_before = p.copy()
        noise_term = noise * np.random.randn(ny - 2, nx - 2) * interior * (1 - interior)
        p[1:-1, 1:-1] += dt * (dpdt + noise_term)
        p = np.clip(p, 0.0, 1.0)

        dp = p - p_before
        u[1:-1, 1:-1] += dt * D * laplacian(u, dx)[1:-1, 1:-1] + K * dp[1:-1, 1:-1]

        if step % save_every == 0:
            frames.append(p.copy())
            progress.progress((step + 1) / nsteps)

    return p, u, frames


# ---------------- sidebar ----------------
with st.sidebar:
    st.header("Parameters")
    nx = st.slider("Grid size", 100, 400, 200, step=50)
    ny = nx
    nsteps = st.slider("Number of steps", 1000, 10000, 5000, step=500)
    dt = st.number_input("Time step dt", 1e-6, 1e-3, 1e-4, format="%.5f")
    dx = st.number_input("Grid spacing dx", 0.01, 0.1, 0.03, format="%.2f")

    st.subheader("Phase-field parameters")
    eps_bar = st.number_input("Interface width eps", 0.005, 0.05, 0.01, format="%.3f")
    tau = st.number_input("Relaxation tau", 1e-5, 1e-3, 3e-4, format="%.5f")
    delta = st.slider("Anisotropy strength delta", 0.0, 0.1, 0.04, step=0.005)
    alpha = st.number_input("alpha", 0.1, 1.0, 0.9, format="%.2f")
    gamma = st.number_input("gamma", 1.0, 20.0, 10.0, format="%.1f")

    st.subheader("Diffusion field")
    undercooling = st.slider("Undercooling (driving force)", 0.1, 1.0, 0.5, step=0.05)
    D = st.number_input("Diffusivity D", 0.1, 5.0, 1.0, format="%.1f")
    K = st.number_input("Latent-heat coupling K", 0.1, 5.0, 1.0, format="%.1f")
    noise = st.slider("Interface noise", 0.0, 0.5, 0.0, step=0.01)

    run = st.button("Run simulation", type="primary")

if run:
    progress = st.progress(0.0, text="Simulating...")
    p, u, frames = run_simulation(
        nx, ny, nsteps, dt, dx, eps_bar, tau, delta,
        alpha, gamma, undercooling, D, K, noise, progress
    )
    progress.progress(1.0, text="Done!")

    solid_fraction = float(p.mean())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    im0 = axes[0].imshow(p, origin="lower", cmap="inferno", extent=[0, nx * dx, 0, ny * dx])
    axes[0].contour(p, levels=[0.5], colors="cyan", linewidths=1.0)
    axes[0].set_title("Phase field p (solid Na = bright)")
    axes[0].set_xlabel("x"); axes[0].set_ylabel("y")
    fig.colorbar(im0, ax=axes[0], label="p")

    im1 = axes[1].imshow(u, origin="lower", cmap="viridis", extent=[0, nx * dx, 0, ny * dx])
    axes[1].set_title("Diffusion field u")
    axes[1].set_xlabel("x"); axes[1].set_ylabel("y")
    fig.colorbar(im1, ax=axes[1], label="u")
    st.pyplot(fig)

    st.metric("Solid fraction", f"{solid_fraction:.3f}")

    st.subheader("Growth evolution")
    fig2, axes2 = plt.subplots(1, len(frames), figsize=(len(frames) * 3, 3.2))
    for ax, fr in zip(axes2, frames):
        ax.imshow(fr, origin="lower", cmap="inferno")
        ax.set_xticks([]); ax.set_yticks([])
    st.pyplot(fig2)
else:
    st.info("Set the parameters in the sidebar and click **Run simulation**.")
