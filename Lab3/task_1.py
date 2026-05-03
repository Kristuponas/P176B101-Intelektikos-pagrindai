# # Laboratorinis darbas 3-1 — Saulės dėmių autoregresija tiesiniu neuronu
# 
# **Data file:** `sunspot.txt` (315 metų: 1700–2014, 2 stulpeliai)
# **Modelio tipas:** AR(n) tiesinė autoregresija
# **Tikslas:** suprognozuoti $a(k)$ remiantis $a(k-1), a(k-2), \ldots, a(k-n)$.
# 
# This notebook walks through all **20 steps** of the 3-1 task:
# 1. Load and plot the time series.
# 2. Build the AR(n) feature matrix $P$ and target vector $T$.
# 3. Visualise the relationship in 3D (for $n=2$).
# 4. Split into training (200 samples) and test sets.
# 5. Fit the model with the **closed-form least squares method (MKM)** — the
#    "ground-truth" reference solution.
# 6. Generate predictions on training and test sets, plot residuals and the
#    error histogram, compute MSE and MAD.
# 7. Re-train the same model **iteratively** as a single linear neuron
#    (Adaline-style SGD).
# 8. Study the influence of the learning rate $lr$ — find the maximum
#    stable value.
# 9. Repeat the experiment for higher autoregression orders $n=6$ and $n=10$
#    and compare results.
# 
# The mathematical notation follows `Ataskaita2-4.docx`.
# 


# ## 1. Imports


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

np.random.seed(0)


# ## 2. Load the data (steps 1–3)
# 
# The file is two whitespace-separated columns: year, sunspot count.
# 


data = np.loadtxt("sunspot.txt")
years  = data[:, 0].astype(int)
counts = data[:, 1].astype(float)
print(f"Loaded {len(data)} rows, years {years[0]}..{years[-1]}")
assert data.shape == (315, 2)


# ## 3. Plot the time series (step 4)
# 
# The 11-year cycle is clearly visible.


plt.figure(figsize=(9, 3.5))
plt.plot(years, counts, color="#1976d2", lw=1.0)
plt.xlabel("Metai"); plt.ylabel("Saulės dėmių skaičius")
plt.title("Saulės dėmių aktyvumas 1700–2014 m.")
plt.grid(alpha=0.3); plt.tight_layout(); plt.show()


# ## 4. Build the AR(n) feature matrix (step 5)
# 
# For order $n$, sample $i$ uses
# $\;[a(i), a(i+1), \ldots, a(i+n-1)] \to a(i+n)$.
# 
# We start with $n=2$.


def build_ar_matrices(series, n):
    """Return P (N x n) and T (N,) for an AR(n) model."""
    N = len(series) - n
    P = np.zeros((N, n))
    T = np.zeros(N)
    for i in range(N):
        P[i] = series[i:i+n]
        T[i] = series[i+n]
    return P, T

n = 2
P, T = build_ar_matrices(counts, n)
print(f"AR({n}): P {P.shape}, T {T.shape}")


# ## 5. 3D scatter of the data (step 6)
# 
# The autoregression equation $a(k) = w_1 a(k{-}2) + w_2 a(k{-}1) + b$ is
# geometrically a **plane** in 3D space. We expect the data points to be
# well approximated by such a plane — i.e. lie close to a planar surface.


fig = plt.figure(figsize=(6.5, 5))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(P[:, 0], P[:, 1], T, c=T, cmap="viridis", s=10)
ax.set_xlabel("a(k-2)"); ax.set_ylabel("a(k-1)"); ax.set_zlabel("a(k)")
ax.set_title("Įvesties (P) ir išvesties (T) duomenys, n=2")
plt.tight_layout(); plt.show()


# ## 6. Train/test split (step 7)
# 
# First 200 samples → training (Pu, Tu); the remaining 113 → test (Pv, Tv).
# 


Pu, Tu = P[:200], T[:200]
Pv, Tv = P[200:], T[200:]
print("Pu", Pu.shape, "Tu", Tu.shape, "| Pv", Pv.shape, "Tv", Tv.shape)


# ## 7. Closed-form least squares (steps 8–9)
# 
# We fit
# \[
#   \hat{a}(k) = b + \mathbf{w}^\top \mathbf{p}(k)
# \]
# by solving the normal equations $(X^\top X)\,\mathbf{w} = X^\top \mathbf{y}$
# with $X = [\mathbf{1}\;|\;P]$ — equation (12) in `Ataskaita2-4.docx`.


def fit_linear_regression(P_train, y_train):
    X = np.hstack([np.ones((len(P_train), 1)), P_train])
    coef = np.linalg.pinv(X.T @ X) @ X.T @ y_train
    return float(coef[0]), coef[1:]

b_lr, w_lr = fit_linear_regression(Pu, Tu)
print(f"MKM coefficients:  b = {b_lr:.4f},  w = {np.round(w_lr, 4)}")


# ## 8. Predictions on train and test sets (steps 10–12)


Tsu_lr = Pu @ w_lr + b_lr        # 1702..1901
Tsv_lr = Pv @ w_lr + b_lr        # 1902..2014

years_train = years[n:n+200]
years_test  = years[n+200:]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(years_train, Tu, label="Tikrosios reikšmės", color="#1976d2")
axes[0].plot(years_train, Tsu_lr, label="MKM prognozė", color="#e53935", lw=1)
axes[0].set_title("Mokymo aibė (1702–1901)")
axes[0].set_xlabel("Metai"); axes[0].set_ylabel("Dėmių skaičius")
axes[0].legend(); axes[0].grid(alpha=0.3)
axes[1].plot(years_test, Tv, label="Tikrosios reikšmės", color="#1976d2")
axes[1].plot(years_test, Tsv_lr, label="MKM prognozė", color="#e53935", lw=1)
axes[1].set_title("Testavimo aibė (1902–2014)")
axes[1].set_xlabel("Metai"); axes[1].set_ylabel("Dėmių skaičius")
axes[1].legend(); axes[1].grid(alpha=0.3)
plt.tight_layout(); plt.show()


# ## 9. Residuals and error histogram (steps 13–14)


e_train = Tu - Tsu_lr
e_test  = Tv - Tsv_lr
e_all   = np.concatenate([e_train, e_test])

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(years_train, e_train, color="#43a047", label="Mokymo aibė")
axes[0].plot(years_test,  e_test,  color="#e53935", label="Testavimo aibė")
axes[0].axhline(0, color="black", lw=0.5)
axes[0].set_xlabel("Metai"); axes[0].set_ylabel("Klaida e(k)")
axes[0].set_title("Prognozės klaidos kreivė")
axes[0].legend(); axes[0].grid(alpha=0.3)
axes[1].hist(e_all, bins=30, color="#1976d2", alpha=0.85, edgecolor="white")
axes[1].set_xlabel("Klaidos reikšmė"); axes[1].set_ylabel("Dažnis")
axes[1].set_title("Prognozės klaidų histograma")
axes[1].grid(alpha=0.3)
plt.tight_layout(); plt.show()


# **Komentaras.** Liekanos svyruoja apie nulį, ekstremalių iškritusių
# verčių nematyti, histograma yra apytikriai simetriška ir pakankamai gerai
# aprašoma normaliuoju skirstiniu. Tai rodo, kad AR(2) modelis tinkamas, o
# pasirinktas tiesinis modelis neturi sisteminio šališkumo.


# ## 10. Quality metrics MSE and MAD (steps 15–17)
# 
# \[
# \text{MSE} = \frac{1}{N}\sum_k (a(k) - \hat{a}(k))^2,\qquad
# \text{MAD} = \operatorname{median}_k\,|a(k) - \hat{a}(k)|
# \]
# 
# The lab requires $\text{MSE}_{\text{train}} \le 300$.


def mse(e): return float(np.mean(e ** 2))
def mad(e): return float(np.median(np.abs(e)))

print(f"MSE  train={mse(e_train):.2f}  test={mse(e_test):.2f}  all={mse(e_all):.2f}")
print(f"MAD  train={mad(e_train):.2f}  test={mad(e_test):.2f}  all={mad(e_all):.2f}")
assert mse(e_train) <= 300, "MSE on training set must not exceed 300!"


# **MSE vs MAD.** MSE has the squared units of the time series and is
# very sensitive to large errors (the few prediction misses near solar
# maxima dominate the value). MAD has the same units as the data and is
# robust to outliers — the typical absolute error is only ~9 sunspots,
# whereas MSE suggests an "RMS error" of $\sqrt{217}\approx14.7$. The two
# metrics are thus complementary, not directly comparable.


# ## 11. Iterative training: linear neuron / Adaline (steps 18–20)
# 
# We re-train the same affine model using batch SGD on the MSE loss. The
# update rule comes from the gradient of MSE:
# \[
#   \Delta w_j = \eta \cdot \tfrac{2}{N}\sum_k (T_k - \hat{T}_k)\, P_{kj},\qquad
#   \Delta b   = \eta \cdot \tfrac{2}{N}\sum_k (T_k - \hat{T}_k).
# \]
# The factor 2 is absorbed into $\eta$.
# 
# **Choosing the learning rate.** Inputs are in the range 0–200, so the
# gradient magnitudes are fairly large. We start with $\eta=5\cdot 10^{-5}$
# and an MSE goal between 150 and 300, max 2000 epochs.


def train_linear_neuron(P_train, y_train, lr, epochs):
    """Batch-mode Adaline: returns (w, b, history of per-epoch MSE)."""
    w = np.zeros(P_train.shape[1])
    b = 0.0
    history = []
    for _ in range(epochs):
        y_hat = P_train @ w + b
        err = y_train - y_hat
        cur_mse = float(np.mean(err ** 2))
        if not np.isfinite(cur_mse) or cur_mse > 1e12:
            history.extend([history[-1] if history else 1e12] * (epochs - len(history)))
            return w, b, history
        w += lr * P_train.T @ err / len(y_train)
        b += lr * err.mean()
        history.append(cur_mse)
    return w, b, history

lr_choice = 5e-5
max_epochs = 2000
mse_goal = 250.0

w_nn, b_nn, hist_nn = train_linear_neuron(Pu, Tu, lr=lr_choice, epochs=max_epochs)
print(f"Neuron weights:  b = {b_nn:.4f},  w = {np.round(w_nn, 4)}")
print(f"Final training MSE = {hist_nn[-1]:.2f}  (goal {mse_goal})")
print(f"Compare with closed-form:  b = {b_lr:.4f},  w = {np.round(w_lr, 4)}")


# **Pastaba.** Per `max_epochs` epochas neurono svoriai dar nepilnai
# sutampa su MKM verčiomis (ypač $b$ atsilieka — bendros optimalios
# apytikslės plokštumos paviršiaus „sukimasis" lėtas), bet $w_2$ jau labai
# artimas. Tai sutampa su `Ataskaita2-4.docx` 4.4 skyrelio pastebėjimu, kad
# neurono konvergavimas yra tolydus, o MKM duoda tikslią atsakymą iš karto.


# Predictions of the neuron and its quality metrics
Tsu_nn = Pu @ w_nn + b_nn
Tsv_nn = Pv @ w_nn + b_nn
print(f"Neuron MSE  train={mse(Tu - Tsu_nn):.2f}  test={mse(Tv - Tsv_nn):.2f}")
print(f"Neuron MAD  train={mad(Tu - Tsu_nn):.2f}  test={mad(Tv - Tsv_nn):.2f}")


# Convergence plot for the chosen lr
plt.figure(figsize=(8, 3.6))
plt.plot(hist_nn, color="#1976d2")
plt.axhline(mse(e_train), color="#e53935", ls="--", label=f"MKM MSE = {mse(e_train):.1f}")
plt.axhline(mse_goal,    color="#43a047", ls=":",  label=f"MSE goal = {mse_goal}")
plt.xlabel("Epocha"); plt.ylabel("Mokymo aibės MSE")
plt.title(f"Neurono mokymosi konvergavimas, lr = {lr_choice}")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout(); plt.show()


# ## 12. Influence of the learning rate (step 19)
# 
# We sweep $lr \in \{10^{-6}, 5\cdot10^{-6}, \ldots, 10^{-3}\}$ and look
# at the final MSE / convergence behaviour.


lr_grid = [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3]
fig, ax = plt.subplots(figsize=(8, 4.5))
final_mses = []
for lr_val in lr_grid:
    _, _, h = train_linear_neuron(Pu, Tu, lr=lr_val, epochs=2000)
    diverged = (not np.isfinite(h[-1])) or h[-1] > 1e6
    final_mses.append((lr_val, "DIVERGED" if diverged else f"{h[-1]:.2f}"))
    label = f"lr={lr_val:g}" + (" (DIVERGED)" if diverged else "")
    ax.plot(np.clip(h, 0, 1e6), label=label, ls="--" if diverged else "-")
ax.set_yscale("log"); ax.set_xlabel("Epocha"); ax.set_ylabel("MSE (log)")
ax.set_title("Konvergavimas esant skirtingam mokymosi greičiui lr")
ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")
plt.tight_layout(); plt.show()

print("Sweep results:")
for lr_val, msg in final_mses:
    print(f"  lr = {lr_val:>8g} -> final MSE = {msg}")


# **Atsakymas į užduotį.** Šio tinklelio rėmuose maksimalus
# konvergavimą užtikrinantis $lr$ yra $2\cdot 10^{-4}$; jau $lr=5\cdot 10^{-4}$
# sukelia svorių diverganciją (svoriai eksponentiškai auga, nes kiekviena
# iteracija svorius perstumia per minimumą). Taigi praktinis darbinis
# intervalas yra $lr \in [10^{-5},\,2\cdot 10^{-4}]$.


# ## 13. Higher-order models n = 6 and n = 10 (steps 20–21)
# 
# Increasing the AR order means more inputs and more weights. The lab asks
# us to compare prediction quality across $n=2, 6, 10$.


rows = []
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.set_xlabel("Epocha"); ax.set_ylabel("Mokymo aibės MSE")
ax.set_title("Neurono konvergavimas skirtingoms n eilėms")

for n_high in [2, 6, 10]:
    P_h, T_h = build_ar_matrices(counts, n_high)
    Pu_h, Tu_h = P_h[:200], T_h[:200]
    Pv_h, Tv_h = P_h[200:], T_h[200:]

    # closed-form
    b_h, w_h = fit_linear_regression(Pu_h, Tu_h)
    rows.append((n_high, "MKM", b_h, mse(Tu_h - (Pu_h @ w_h + b_h)),
                 mse(Tv_h - (Pv_h @ w_h + b_h)), np.round(w_h, 4)))

    # neuron — smaller lr for higher orders to keep things stable
    lr_h = 5e-5 if n_high == 2 else (1e-5 if n_high == 6 else 5e-6)
    w_n, b_n, h_n = train_linear_neuron(Pu_h, Tu_h, lr=lr_h, epochs=2000)
    rows.append((n_high, "Neuronas", b_n,
                 mse(Tu_h - (Pu_h @ w_n + b_n)),
                 mse(Tv_h - (Pv_h @ w_n + b_n)),
                 np.round(w_n, 4)))
    ax.plot(h_n, label=f"n={n_high}, lr={lr_h:g}")

ax.set_yscale("log"); ax.legend(); ax.grid(alpha=0.3, which="both")
plt.tight_layout(); plt.show()


import pandas as pd
df = pd.DataFrame(rows, columns=["n", "method", "b", "MSE_train", "MSE_test", "weights"])
df["b"]         = df["b"].round(4)
df["MSE_train"] = df["MSE_train"].round(2)
df["MSE_test"]  = df["MSE_test"].round(2)
df.to_csv("lab31_results.csv", index=False)
df


# **Komentaras.**
# - $n$ didėjant, MKM mokymo aibės MSE tolydžiai mažėja (217 → 211 → 191) —
#   daugiau praeities reikšmių leidžia geriau aprašyti netiesinę dinamiką.
# - Vis dėlto, kaip pastebi `Ataskaita2-4.docx`, dar didinant $n$ iki 20–25,
#   MSE pradeda **didėti** dėl multikolinearumo: artimi praeities mėginiai
#   yra stipriai koreliuoti su viduriniaisiais, taigi normaliųjų lygčių
#   matrica $X^\top X$ tampa beveik singuliari.
# - Neurono konvergavimas didesnėms $n$ vertėms reikalauja mažesnio $lr$,
#   nes požymių energija $\|P\|^2$ auga proporcingai $n$.
# 


# ## 14. Atsakymai į užduotyje pateiktus klausimus
# 
# **Ar mokymosi procesas konverguoja?**
# Taip — pasirinktam $lr=5\cdot 10^{-5}$ MSE monotoniškai mažėja iki ~277,
# patenka į užduoties leistiną intervalą [150, 300]. Mokymas nediverguoja.
# 
# **Naujos svorių reikšmės?**
# $b \approx 0.48,\; w \approx [-0.58,\; 1.47]$.
# Palyginkite su MKM: $b\approx 13.40,\; w\approx [-0.68,\; 1.37]$.
# Krypčių (svorio dydžių) sutapimas geras; bias atsilieka, nes prie nulinės
# inicializacijos pasislinkti į ~13 reikia daugiau epochų.
# 
# **MSE ir MAD reikšmės?**
# Neuronui: MSE_train ≈ 277, MAD_train ≈ 8.5;
# MKM-ui: MSE_train ≈ 217, MAD_train ≈ 8.7. MAD vertė beveik identiška —
# tai patvirtina, kad „tipiškoji" prognozės klaida yra panaši; skirtumą MSE
# sukuria keli stambūs nuokrypiai, kurių MKM atveju yra mažiau.
# 
# **Maksimali leistina lr reikšmė?**
# $lr_{\max} \approx 2\cdot 10^{-4}$ pagal mūsų tinklelį. Didesnis $lr$
# sukelia diverganciją.
# 
# **$n$ keitimo įtaka?**
# Mokymo aibės MSE mažėja didinant $n$ iki tam tikros ribos
# (MKM: 217 → 211 → 191), tačiau labai dideliems $n$ ($\geq 20$) MSE
# pradeda augti dėl multikolinearumo (žr. sk. 13).
# 


# ## 15. Išvados
# 
# - AR(2) modelis su MKM pasiekia $\text{MSE}_{\text{train}}=217.17$ —
#   gerokai mažiau už užduoties slenkstį 300.
# - Iteracinis tiesinio neurono mokymas duoda artimas (bet ne tikslias)
#   svorio reikšmes: skirtumas atsiranda dėl mokymosi proceso lėto
#   konvergavimo prie globalaus minimumo.
# - Mokymosi greitis $lr$ turi būti pakankamai mažas, kad gradiento žingsnis
#   nebūtų didesnis už atstumą iki minimumo: per didelis $lr$ sukelia
#   diverganciją.
# - Aukštesnės eilės modeliai pagerina aproksimaciją, tačiau ne be ribų —
#   multikolinearumas daro modelį jautrų ir mažina apibendrinimo kokybę.
#