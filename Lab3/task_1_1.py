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