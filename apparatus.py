import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timezone, timedelta

# ================== Time (JST) ==================
JST = timezone(timedelta(hours=9))
today = datetime.now(JST).strftime("%Y-%m-%d")
rng = np.random.default_rng()

# ================== Simulation time ==================
T = 10.0
N = 2000
t = np.linspace(0, T, N)
dt = t[1] - t[0]

# ================== Input signals ==================
def make_input(kind, t):
    A = rng.uniform(0.8, 1.5)
    x = np.zeros_like(t)

    if kind == "step":
        x[t > t[-1]*0.1] = A
        desc = f"step (A={A:.2f})"

    elif kind == "impulse":
        x[int(0.1*len(t))] = A / dt
        desc = f"impulse (area≈{A:.2f})"

    elif kind == "sine":
        f = rng.uniform(0.2, 2.0)
        x = A * np.sin(2*np.pi*f*t)
        desc = f"sine (A={A:.2f}, f={f:.2f} Hz)"

    elif kind == "square":
        f = rng.uniform(0.2, 1.0)
        x = A * np.sign(np.sin(2*np.pi*f*t))
        desc = f"square (A={A:.2f}, f={f:.2f} Hz)"

    elif kind == "ramp":
        x = np.clip(A*(t/t[-1]), 0, A)
        desc = f"ramp (A={A:.2f})"

    elif kind == "noise":
        x = A * rng.normal(0, 0.5, size=len(t))
        desc = f"noise (σ≈{0.5*A:.2f})"

    else:
        raise ValueError("unknown input kind")

    return x, desc

input_kind = rng.choice(["step","impulse","sine","square","ramp","noise"])
x, x_desc = make_input(input_kind, t)

# ================== System selection ==================
system = rng.choice(["RC","RL","RLC"])

# ---------- RC ----------
if system == "RC":
    R = rng.uniform(0.5, 5.0)
    C = rng.uniform(0.2, 5.0)
    tau = R * C

    y = np.zeros_like(x)
    a = np.exp(-dt/tau)
    b = 1 - a
    for n in range(1, N):
        y[n] = a*y[n-1] + b*x[n]

    params = f"R={R:.2f}, C={C:.2f}, τ={tau:.2f}"

# ---------- RL ----------
elif system == "RL":
    R = rng.uniform(0.5, 5.0)
    L = rng.uniform(0.5, 5.0)
    tau = L / R

    y = np.zeros_like(x)
    a = np.exp(-dt/tau)
    b = 1 - a
    for n in range(1, N):
        y[n] = a*y[n-1] + b*x[n]

    params = f"R={R:.2f}, L={L:.2f}, τ={tau:.2f}"

# ---------- RLC (standard 2nd-order form) ----------
else:
    R = rng.uniform(0.3, 2.0)
    L = rng.uniform(0.5, 3.0)
    C = rng.uniform(0.2, 3.0)

    wn = 1/np.sqrt(L*C)                 # natural frequency
    zeta = (R/2)*np.sqrt(C/L)           # damping ratio (typical mapping)

    y = np.zeros_like(x)
    yd = np.zeros_like(x)               # y' (derivative)

    # y'' = wn^2 (x - y) - 2 zeta wn y'
    for n in range(1, N):
        ydd = (wn**2) * (x[n] - y[n-1]) - (2*zeta*wn) * yd[n-1]
        yd[n] = yd[n-1] + dt * ydd
        y[n]  = y[n-1]  + dt * yd[n]

    params = f"R={R:.2f}, L={L:.2f}, C={C:.2f}, ωn={wn:.2f}, ζ={zeta:.2f}"

# ================== Plot ==================
plt.figure(figsize=(8,4.5))
plt.plot(t, x, label="input")
plt.plot(t, y, label="output")
plt.xlabel("t [arb.]")
plt.ylabel("[arb.]")
plt.title(f"Daily Useless {system} System ({today})\nInput: {input_kind}, {x_desc}\n{params}")
plt.legend()
plt.tight_layout()
plt.savefig("result.svg")
plt.close()

# ================== README ==================
readme = f"""# Daily Useless Physics

Every day, GitHub Actions generates a **random dynamical system**
and excites it with a **random input signal**.

## Today's Result ({today})

- **System**: {system}
- **Parameters**: {params}
- **Input**: {x_desc}

![result](result.svg)

> There is no real circuit.
> But the equations are behaving.
"""

with open("README.md","w",encoding="utf-8") as f:
    f.write(readme)
