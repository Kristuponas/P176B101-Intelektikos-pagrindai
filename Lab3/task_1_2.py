import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression

# 1. Atsisiųsti sunspot.txt. Faile pateikti duomenys apie saulės dėmių aktyvumą nuo 1700 iki 2014 metų.
# 2. Užkrauti failo turinį į darbinę atmintį.

data = np.loadtxt("sunspot.txt")
years  = data[:, 0].astype(int)
counts = data[:, 1].astype(float)

# 3. Patikrinti ar užkrauta atitinkama matrica – pirmas stulpelis atitinka metus, antras – saulės dienų aktyvumą.

assert data.shape == (315, 2)                       # Tikrina ar visos 315 eiluciu ir 2 stulpeliai
assert years [0] == 1700 and years[-1] == 2014      # TIkrina ar metai prasideda 1700 ir baigiasi 2014
assert np.all(years[1:] == years[:-1] + 1)          # Tikrina ar metai yra nuosekliai dideja po viena
assert np.all(counts >= 0)                          # Tikrina ar saules demiu skaicius nera neigiamas

# 4. Pirma užduotis, kurią turi realizuoti mūsų programa – nubrėžti saulės dėmių aktyvumo už 1700-2014 metus grafiką. 
#    Grafikas turi būti pilnai aprašytas – pateikti ašių ir grafiko pavadinimus. 

plt.figure(figsize=(9, 3.5))
plt.plot(years, counts, color="#1976d2", lw=1.0)
plt.xlabel("Metai"); plt.ylabel("Saules demiu skaicius")
plt.title("Saules demiu aktyvumas 1700-2014 m.")
plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig("lab31_grafikas_1.png", dpi=300)
plt.show()

# 5. Priimkime, kad autoregresinio modelio eilė bus lygi 2 (n=2). T.y priimame, kad sekančių metų dėmių prognozė 
#    yra įmanoma turint tik dviejų ankstesnių metų dėmių skaičių. Tuomet neuronas turės tik du įėjimus. Papildykite 
#    scenarijų, aprašant matricas P ir T, kuriose atitinkamai pateikiami (mokymosi) įvesties duomenys o taip pat 
#    išvesties duomenys.

def ar_matrix(series, n):
    N = len(series) - n
    P = np.zeros((N, n))
    T = np.zeros(N)
    for i in range(N):
        P[i] = series[i:i+n]
        T[i] = series[i+n]
    return P, T

n = 2
P, T = ar_matrix(counts, n)
print(f"AR({n}): P {P.shape}, T {T.shape}\n")

# 6. Nubrėžti trimatę diagramą, joje vaizduojant įvesties ir išvesties duomenis P ir T atitinkamai. Išanalizuoti 
#    gautą grafiką - sukiokite kol nepamatysite koreliacijos tarp duomenų požymius. Kokia yra neurono svorio 
#    koeficientų w1, w2, b optimalių reikšmių parinkimo grafinė interpretacija? Pridėti ašių ir grafiko pavadinimus. 

fig = plt.figure(figsize=(6.5, 5))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(P[:, 0], P[:, 1], T, c=T, cmap="viridis", s=10)
ax.set_xlabel("a(k-2)"); ax.set_ylabel("a(k-1)"); ax.set_zlabel("a(k)")
ax.set_title("Ivesties (P) ir isvesties (T) duomenys, n=2")
plt.tight_layout()
plt.savefig("lab31_grafikas_2.png", dpi=300)
plt.show()

# 7. Išskirkime iš įvesties P ir išvesties T duomenų rinkinių fragmentus, turinčius po 200 pradžioje esamų 
#    duomenų – taip vadinamą apmokymo duomenų rinkinį. Remiantis šiuo rinkiniu apskaičiuosime optimalias neurono 
#    svorio koeficientų reikšmes (autoregresinio modelio parametrus). Likę duomenys bus panaudoti modeliui verifikuoti. 
#    Tuomet, panaudojant jau esamas P ir T matricas, apibrėžkime dvi naujas – Pu ir Tu, kurios turės pirmus 200 duomenų.

Pu, Tu = P[:200], T[:200]
Pv, Tv = P[200:], T[200:]
print(f"Pu {Pu.shape} Tu {Tu.shape} | Pv {Pv.shape} Tv {Tv.shape}\n")

# 8. Sukurkite tiesinės autoregresijos modelį panaudojant apmokymo duomenų matricas Pu ir Tu. Python mokomoji medžiaga 
# pateikta adresu https://realpython.com/linear-regression-in-python/.

lr_model = LinearRegression(fit_intercept=True)  # leisim modeliui pats mokytis b
lr_model.fit(Pu, Tu)                             # apmokom ant (Pu, Tu)

# 9. Pavaizduoti gautas koeficientų reikšmes

b_lr = float(lr_model.intercept_)
w_lr = lr_model.coef_
print(f"Gautos koeficientu reiksmes b = {b_lr:.4f},  w = {np.round(w_lr, 4)}\n")

# 10. Sekančiame žingsnyje atliksime modelio verifikaciją – t.y. patikrinsime prognozavimo kokybę atliekant modelio veikimo 
#     imitaciją. Pradžioje tai atliksime su apmokymo duomenų rinkiniu, kuris buvo panaudotas svorio koeficientams apskaičiuoti.

Tsu_lr = Pu @ w_lr + b_lr        # 1702..1901
Tsv_lr = Pv @ w_lr + b_lr        # 1902..2014

years_train = years[n:n+200]
years_test = years[n+200:]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(years_train, Tu, label="Tikrosios reiksmes", color="#1976d2")
axes[0].plot(years_train, Tsu_lr, label="MKM prognoze", color="#e53935", lw=1)
axes[0].set_title("Mokymo aibe (1702-1901)")
axes[0].set_xlabel("Metai"); axes[0].set_ylabel("Demiu skaicius")
axes[0].legend(); axes[0].grid(alpha=0.3)
axes[1].plot(years_test, Tv, label="Tikrosios reiksmes", color="#1976d2")
axes[1].plot(years_test, Tsv_lr, label="MKM prognoze", color="#e53935", lw=1)
axes[1].set_title("Testavimo aibe (1902-2014)")
axes[1].set_xlabel("Metai"); axes[1].set_ylabel("Demiu skaicius")
axes[1].legend(); axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig("lab31_grafikas_3.png", dpi=300)
plt.show()

# 11. Sukurti prognozės klaidos vektorių e  (žr. išraiškos 1.2 paaiškinimą). Nubraižyti prognozės klaidos grafiką. Aprašyti 
#     jo ašis ir suteikti pavadinimą.

# 12. Nubraižyti prognozės klaidų histogramą (hist). Ją pakomentuokite.

e_train = Tu - Tsu_lr
e_test = Tv - Tsv_lr
e_all = np.concatenate([e_train, e_test])

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(years_train, e_train, color="#43a047", label="Mokymo aibe")
axes[0].plot(years_test,  e_test,  color="#e53935", label="Testavimo aibe")
axes[0].axhline(0, color="black", lw=0.5)
axes[0].set_xlabel("Metai"); axes[0].set_ylabel("Klaida e(k)")
axes[0].set_title("Prognozes klaidos kreive")
axes[0].legend(); axes[0].grid(alpha=0.3)
axes[1].hist(e_all, bins=30, color="#1976d2", alpha=0.85, edgecolor="white")
axes[1].set_xlabel("Klaidos reiksme"); axes[1].set_ylabel("Dažnis")
axes[1].set_title("Prognozes klaidu histograma")
axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig("lab31_grafikas_4.png", dpi=300)
plt.show()

# 13. Remiantis (1.3) apskaičiuoti vidutinės kvadratinės prognozės klaidos reikšmę (ang. Mean-Square-Error, MSE):
#     MSE = 1/N * sum(a(k) - â(k))^2) = 1/N * sum(e(k)^2)     
#     Šiame darbe MSE įvertis neturi viršyti 300.
#
#     Apskaičiuokite prognozės absoliutaus nuokrypio medianą (ang. Median Absolute Deviation)
#     MAD = median(|e(k)|)
#
#     Palyginkite skirtumus tarp MSE ir MAD įverčių ir pakomentuokite.
#
#     Sekančiuose punktuose sukurto scenarijaus tekstą modifikuosime, kad modelio svorio koeficientai būtų skaičiuojami 
#     iteraciniu metodu – atliekant neurono apmokymo procedūrą. 

def mse(e): return float(np.mean(e ** 2))
def mad(e): return float(np.median(np.abs(e)))

print(f"MSE  train = {mse(e_train):.2f}  test = {mse(e_test):.2f}  all = {mse(e_all):.2f}")
print(f"MAD  train = {mad(e_train):.2f}  test = {mad(e_test):.2f}  all = {mad(e_all):.2f}\n")
assert mse(e_train) <= 300, "Šiame darbe MSE įvertis neturi viršyti 300.\n"

# 14. Scenarijų išsaugokite nauju vardu (padarykite sukurtos programos kopiją). Eksperimento būdu parinkite mokymosi 
#     greičio lr reikšmę (0< lr <= 1). Parenkama reikšmė (pvz. 0.1), ji naudojama modelyje (žr. sekantį žingsnį). 
#     Jei modelis nekonverguoja, parinkta reikšmė mažinama (pvz. 10 kartų) ir procesas kartojamas kol modelis 
#     nepradės konverguoti.

# 15. Panaudojant aprašą pateiktą adresu https://www.bogotobogo.com/python/scikit-learn/Single-Layer-Neural-Network-Adaptive-Linear-Neuron.php 
#     sukurti tiesinį neuroną.

def train_linear_neuron(P_train, y_train, lr, epochs):
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

# 16. Apibrėžti siekiamą mokymosi klaidos MSE reikšmę (ang. error goal)  intervale 150 – 300 ir maksimalų epochų 
#     kiekį (pvz. 1000). (Pastaba. Vienos epochos metu modelis panaudoja visą duomenų rinkinį. Vienos iteracijos 
#     metu modelis panaudoja tik vieną duomenų rinkinio eilutę.)

lr_choice = 5e-5
max_epochs = 2000
mse_goal = 250.0

# 17. Įvykdyti modelį. Atspausdinti gautas po apmokymo svorio koeficientų reikšmes. Jas palyginti su gautais 9 žingsnyje.

w_nn, b_nn, hist_nn = train_linear_neuron(Pu, Tu, lr=lr_choice, epochs=max_epochs)
print(f"Neurono svoriai:  b = {b_nn:.4f},  w = {np.round(w_nn, 4)}\n")
print(f"Galutinis MSE = {hist_nn[-1]:.2f}  (goal {mse_goal})\n")
print(f"Palyginimas su 9 zingsnio rezultatais:  b = {b_lr:.4f},  w = {np.round(w_lr, 4)}\n")

Tsu_nn = Pu @ w_nn + b_nn
Tsv_nn = Pv @ w_nn + b_nn
print(f"Neurono MSE  train = {mse(Tu - Tsu_nn):.2f}  test = {mse(Tv - Tsv_nn):.2f}")
print(f"Neurono MAD  train = {mad(Tu - Tsu_nn):.2f}  test = {mad(Tv - Tsv_nn):.2f}\n")

# 18. Papildomai prie užduotų darbo metu klausimų, atsakykite raštu ir šiuos klausimus: 
#     • Ar mokymosi procesas yra konverguojantis? Jeigu ne, pamąstyti kas gali būti priežastimi ir pakeisti atitinkamą parametrą.
#     • Kokios yra naujos neurono svorių koeficientų reikšmės?
#     • Kokia yra neurono darbo kokybės įverčio MSE ir MAD reikšmės?

plt.figure(figsize=(8, 3.6))
plt.plot(hist_nn, color="#1976d2")
plt.axhline(mse(e_train), color="#e53935", ls="--", label=f"MKM MSE = {mse(e_train):.1f}")
plt.axhline(mse_goal,    color="#43a047", ls=":",  label=f"MSE goal = {mse_goal}")
plt.xlabel("Epocha"); plt.ylabel("Mokymo aibes MSE")
plt.title(f"Neurono mokymosi konvergavimas, lr = {lr_choice}")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("lab31_grafikas_5.png", dpi=300)
plt.show()

# 18a. Atsakymas į klausimus (Ataskaitai):
#      • Ar mokymosi procesas yra konverguojantis?
#           - Taip. Is grafiko matyti, kad MSE pradzioje didelis, bet greitai mazeja ir stabilizuojasi apie 277, nebedideja, taigi procesas konverguoja.
#      • Kokios yra naujos neurono svorių koeficientų reikšmės?
#           - b ≈ 0.48, w ≈ [-0.58, 1.47]
#      • Kokia yra neurono darbo kokybės įverčio MSE ir MAD reikšmės?
#           - MSE_train ≈ 277, MSE_test ≈ 486
#           - MAD_train ≈ 8.5, MAD_test ≈ 12.7

# 19. Procedūrą pakartoti kitoms 17 punkto parametrų reikšmėms. Ištirti jų reikšmės įtaką į mokymosi proceso eigą ir prognozavimo 
#     kokybę. Kokia yra maksimali leistina mokymosi proceso greičio koeficiento lr reikšmė, kuri užtikrina proceso konvergenciją? 

lr_grid = [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3]
fig, ax = plt.subplots(figsize=(8, 4.5))
final_mses = []
for lr_val in lr_grid:
    _, _, h = train_linear_neuron(Pu, Tu, lr=lr_val, epochs=2000)
    diverged = (not np.isfinite(h[-1])) or h[-1] > 1e6
    final_mses.append((lr_val, "Divergavo" if diverged else f"{h[-1]:.2f}"))
    label = f"lr={lr_val:g}" + (" (Divergavo)" if diverged else "")
    ax.plot(np.clip(h, 0, 1e6), label=label, ls="--" if diverged else "-")
ax.set_yscale("log"); ax.set_xlabel("Epocha"); ax.set_ylabel("MSE (log)")
ax.set_title("Konvergavimas esant skirtingam mokymosi greičiui lr")
ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")
plt.tight_layout()
plt.savefig("lab31_grafikas_6.png", dpi=300)
plt.show()

print("Rezultatai:")
for lr_val, msg in final_mses:
    print(f"  lr = {lr_val:>8g} -> final MSE = {msg}")

# 20. Darbą atlikome priimant pradžioje pasiūlytą mūsų modelio struktūrą – sekančios reikšmės prognozavimas atliekamas remiantis 
#     dviejų ankstesniųjų metų duomenimis (t.y. modelio eilė n=2). Tiesinės autoregresijos ir tiesinio neurono modelių scenarijus 
#     pakoreguoti tokiu būdu, kad prognozė remtųsi didesniu nei anksčiau duomenų kiekiu – kai n=6 ir kai n=10. Tuo tikslu reikės 
#     atitinkamai modifikuoti matricų P ir T apibrėžimus. Ištirti (grafiškai ir pakomentuojant raštu) modelio struktūros keitimo 
#     įtaką į prognozavimo kokybę.

rows = []
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.set_xlabel("Epocha"); ax.set_ylabel("Mokymo aibes MSE")
ax.set_title("Neurono konvergavimas skirtingoms n eilems")

for n_high in [2, 6, 10]:
    P_h, T_h = ar_matrix(counts, n_high)
    Pu_h, Tu_h = P_h[:200], T_h[:200]
    Pv_h, Tv_h = P_h[200:], T_h[200:]

    # closed-form
    h_model = LinearRegression(fit_intercept=True)
    h_model.fit(Pu_h, Tu_h)

    b_h = float(h_model.intercept_)
    w_h = h_model.coef_

    rows.append((n_high, "MKM", b_h, mse(Tu_h - (Pu_h @ w_h + b_h)),
                 mse(Tv_h - (Pv_h @ w_h + b_h)), np.round(w_h, 4)))

    lr_h = 5e-5 if n_high == 2 else (1e-5 if n_high == 6 else 5e-6)
    w_n, b_n, h_n = train_linear_neuron(Pu_h, Tu_h, lr=lr_h, epochs=2000)
    rows.append((n_high, "Neuronas", b_n,
                 mse(Tu_h - (Pu_h @ w_n + b_n)),
                 mse(Tv_h - (Pv_h @ w_n + b_n)),
                 np.round(w_n, 4)))
    ax.plot(h_n, label=f"n={n_high}, lr={lr_h:g}")

ax.set_yscale("log"); ax.legend(); ax.grid(alpha=0.3, which="both")
plt.tight_layout()
plt.savefig("lab31_grafikas_7.png", dpi=300)
plt.show()

df = pd.DataFrame(rows, columns=["n", "method", "b", "MSE_train", "MSE_test", "weights"])
df["b"]         = df["b"].round(4)
df["MSE_train"] = df["MSE_train"].round(2)
df["MSE_test"]  = df["MSE_test"].round(2)
df.to_csv("lab31_results.csv", index=False)
df


#   Išvados
# 
# - AR(2) modelis su MKM pasiekia MSE_train = 217.17$ ir MSE_test = 386.40
#   gerokai mažiau už užduoties slenkstį 300.

# - Iteracinis tiesinio neurono mokymas duoda artimas (bet ne tikslias)
#   svorio reikšmes: skirtumas atsiranda dėl mokymosi proceso lėto
#   konvergavimo prie globalaus minimumo.

# - Mokymosi greitis kr turi būti pakankamai mažas, kad gradiento žingsnis
#   nebūtų didesnis už atstumą iki minimumo: per didelis lr sukelia
#   diverganciją.

# - Aukštesnės eilės modeliai pagerina aproksimaciją, tačiau ne be ribų —
#   multikolinearumas daro modelį jautrų ir mažina apibendrinimo kokybę.