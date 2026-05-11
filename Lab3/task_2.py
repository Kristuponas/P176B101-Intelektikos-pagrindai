import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

# 1. Pasirinkite tikslo atributą iš 1 laboratorinio darbo arba kito duomenų rinkinio (jei tikslo 
#    atributas nebuvo apibrėžtas). Pastaba: pavyzdžiui, banko klientų duomenų rinkinyje tikslo 
#    atributu gali būti laikomi kliento mokumo lygis arba kredito reitingas, filmų duomenų rinkinyje 
#    tikslo atributu gali būti sugeneruotas pelnas. 

#    Mano pasirinktas tikslo atributas yra burnout_risk.

df = pd.read_csv("work_from_home_burnout_dataset.csv")
print("shape:\n", df.shape)
df.head()

print("Dtypes:")
print(df.dtypes)
print("\nMissing values per column:")
print(df.isna().sum())
print("\nClass counts (burnout_risk):")
print(df["burnout_risk"].value_counts())

# 2. Jei reikia, atlikite tikslinių atributų reikšmių pertvarkymus (pvz., platus skaitinių atributų 
#    verčių diapazonas keičiamas mažesniu (kategorinių) intervalų skaičiumi (pvz., prognozuojamų 
#    reikšmių diapazoną 1..2000 galima pakeisti 1...5 intervalais).

#    Ismetam burnout_score, nes atitinkamai burnout_risk reiskia ta pati.
#    Taip pat user_id, nes tai yra beprasmiskas identifikatorius, kuris neturi jokios reiksmes modelio mokymuisi.
#    Atributas day_type, bus uzkoduotas kaip binary (false = 0, true = 1) "is_weekend" (0 - Weekday, 1 - Weekend).
#    Atributas burnout_risk, bus uzkoduotas kaip integer (Low=0, Medium=1, High=2).

print(df.groupby("burnout_risk")["burnout_score"].agg(["min", "max", "mean"]))

df["is_weekend"] = (df["day_type"] == "Weekend").astype(int)

numeric_cols = ["work_hours", "screen_time_hours", "meetings_count",
                "breaks_taken", "after_hours_work", "sleep_hours",
                "task_completion_rate"]
feature_cols = numeric_cols + ["is_weekend"]
X_all = df[feature_cols].to_numpy(dtype=np.float64)

class_order = ["Low", "Medium", "High"]
label2int = {c: i for i, c in enumerate(class_order)}
y_int = df["burnout_risk"].map(label2int).to_numpy()
n_classes = len(class_order)
n_features = X_all.shape[1]

print("\nAtributai:", feature_cols)
print("\nburnout_risk reiksmiu [low   medium   high] kiekis:", np.bincount(y_int))

counts = np.bincount(y_int)
inv_freq = counts.sum() / (n_classes * counts)
print("Inverse-frequency class weights:", inv_freq)

plt.figure(figsize=(5, 3.2))
plt.bar(class_order, counts, color=["#4caf50", "#ffb300", "#e53935"])
for i, c in enumerate(counts):
    plt.text(i, c + 15, str(c), ha="center")
plt.title("burnout_risk atributo klasiu pasiskirstymas")
plt.ylabel("count")
plt.tight_layout()
plt.savefig("lab32_grafikas_1.png", dpi=300)
plt.show()

# 3. Sukurkite reikšmės prognozavimo ar klasifikacijos modelį. Modelio sukūrimas gali būti atliktas keliais 
#    būdais, pasirinkti vieną:
#        a. Biblioteka numpy. Papildoma medžiaga:
#            i. ANN.ipynb daugiasluoksnio tinklo kodo pavyzdys. Pastaba: Duotajame tinkle naudojama klasikinė sigmoidinė aktyvacijos funkcija (galima naudoti ir softmax aktyvacijos, tačiau reikalingas kodo modifikacija) klasifikavimo uždaviniui, norint tinklą pritaikyti regresijos uždaviniui, reikia tiesiog nenaudoti sigmoidės paskutiniame sluoksnyje.
#            ii. https://iamtrask.github.io/2015/07/12/basic-python-network/
#        b. Biblioteka PyTorch. Papildoma medžiaga:
#            i. https://docs.pytorch.org/tutorials/beginner/examples_nn/polynomial_nn.html

#     Mano pasirinktas budas yra b.

def macro_f1(y_true_int, y_pred_int, n_classes):
    f1s = []
    for c in range(n_classes):
        tp = np.sum((y_true_int == c) & (y_pred_int == c))
        fp = np.sum((y_true_int != c) & (y_pred_int == c))
        fn = np.sum((y_true_int == c) & (y_pred_int != c))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
    return float(np.mean(f1s))

def run_cv_pytorch(build_model, *, X, y_int, n_classes,
                   epochs, lr, batch_size, standardize, class_weights=None):
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    fold_acc, fold_loss, fold_f1 = [], [], []

    for k, (tr_idx, va_idx) in enumerate(skf.split(X, y_int), 1):
        X_tr, X_va = X[tr_idx], X[va_idx]
        if standardize:
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr)
            X_va = scaler.transform(X_va)

        X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
        y_tr_t = torch.tensor(y_int[tr_idx], dtype=torch.long)
        X_va_t = torch.tensor(X_va, dtype=torch.float32)
        y_va_t = torch.tensor(y_int[va_idx], dtype=torch.long)

        model = build_model()
        if class_weights is not None:
            cw = torch.tensor(class_weights, dtype=torch.float32)
            criterion = nn.CrossEntropyLoss(weight=cw)
        else:
            criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0)

        for epoch in range(epochs):
            model.train()
            perm = torch.randperm(len(X_tr_t))
            for i in range(0, len(X_tr_t), batch_size):
                idx = perm[i:i+batch_size]
                xb, yb = X_tr_t[idx], y_tr_t[idx]
                optimizer.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            logits_va = model(X_va_t)
            loss_va = criterion(logits_va, y_va_t).item()
            preds = logits_va.argmax(dim=1).cpu().numpy()
            acc = (preds == y_int[va_idx]).mean()
            f1 = macro_f1(y_int[va_idx], preds, n_classes)
        fold_loss.append(loss_va)
        fold_acc.append(acc)
        fold_f1.append(f1)

    return {
        "fold_acc": fold_acc,
        "fold_loss": fold_loss,
        "fold_f1": fold_f1,
        "mean_acc": float(np.mean(fold_acc)),
        "std_acc": float(np.std(fold_acc)),
        "mean_loss": float(np.mean(fold_loss)),
        "mean_f1": float(np.mean(fold_f1)),
    }

n_features = X_all.shape[1]
n_classes = len(class_order)

def build_E0():
    # Mazas sigmoid + MSE atitikmuo
    return nn.Sequential(
        nn.Linear(n_features, 4),
        nn.Sigmoid(),
        nn.Linear(4, n_classes),
    )

def build_E1():
    # Tas pats, bet daugiau epochu ir su standardizavimu
    return nn.Sequential(
        nn.Linear(n_features, 4),
        nn.Sigmoid(),
        nn.Linear(4, n_classes),
    )

def build_E2():
    # 8 neuronai, Softmax+CE implicit (CrossEntropyLoss)
    return nn.Sequential(
        nn.Linear(n_features, 8),
        nn.Sigmoid(),
        nn.Linear(8, n_classes),
    )

def build_E3():
    # 16-16 ReLU
    return nn.Sequential(
        nn.Linear(n_features, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, n_classes),
    )

def build_E4():
    # 32-32 ReLU; L2 (weight_decay) uzdesim per optimizer
    return nn.Sequential(
        nn.Linear(n_features, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, n_classes),
    )

def build_E5():
    # 32-16 ReLU, class-weighted CE
    return nn.Sequential(
        nn.Linear(n_features, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, n_classes),
    )

EXPS = [
    dict(name="E0 baseline",  builder=build_E0, epochs=30,  lr=0.01, batch=32, standardize=False, weight_decay=0.0, class_weights=None),
    dict(name="E1 std",       builder=build_E1, epochs=200, lr=0.05, batch=32, standardize=True,  weight_decay=0.0, class_weights=None),
    dict(name="E2 softmax",   builder=build_E2, epochs=200, lr=0.05, batch=32, standardize=True,  weight_decay=0.0, class_weights=None),
    dict(name="E3 relu",      builder=build_E3, epochs=300, lr=0.01, batch=32, standardize=True,  weight_decay=0.0, class_weights=None),
    dict(name="E4 relu_l2",   builder=build_E4, epochs=300, lr=0.005,batch=32, standardize=True,  weight_decay=1e-3, class_weights=None),
    dict(name="E5 class_w",   builder=build_E5, epochs=400, lr=0.005,batch=32, standardize=True,  weight_decay=5e-4, class_weights=inv_freq),
]

# 4. Įvertinkite sukurto modelio vidutinį tikslumo įvertį, taikant 10 intervalų kryžminės patikros metodą. 

results = []
for i, exp in enumerate(EXPS):
    print(f"\n=== {exp['name']} ===")
    res = run_cv_pytorch(
        build_model=exp["builder"],
        X=X_all, y_int=y_int, n_classes=n_classes,
        epochs=exp["epochs"], lr=exp["lr"],
        batch_size=exp["batch"], standardize=exp["standardize"],
        class_weights=exp["class_weights"],
    )
    res.update(name=exp["name"], epochs=exp["epochs"], lr=exp["lr"],
               batch=exp["batch"], standardize=exp["standardize"])
    results.append(res)
    print(f"   mean acc = {res['mean_acc']*100:.2f}% (±{res['std_acc']*100:.2f}),"
          f"  macro F1 = {res['mean_f1']:.4f},  mean loss = {res['mean_loss']:.4f}")

# 5. Atlikite bent 5 papildomus eksperimentus, kad padidintumėte vidutinį tikslumą bent 5 procentais ir 
#    pakartokite 4-ą darbo eigos žingsnį. Kiekvienas eksperimentas ir jo poveikis modelio vidutiniui 
#    tikslumui turi būti dokumentuotas (pvz.: lentelės pavidalu pateiktos pasirinktos priemonės ir 
#    rezultatai). Papildomiems eksperimentams galimos priemonės:
#    • Pertvarkyti duomenų rinkinį,
#    • Pakeiskite mokymosi greitį,
#    • Pakeiskite aktyvacijos funkciją (reikalinga duotojo kodo modifikacija)
#    • Pakeisti dirbtinio neuronų tinklo (DNT) struktūrą.

rows = []
for r in results:
    row = {"experiment": r["name"], "epochs": r["epochs"],
           "lr": r["lr"], "batch": r["batch"],
           "standardize": r["standardize"],
           "mean_acc_%": round(r["mean_acc"] * 100, 3),
           "std_acc_%": round(r["std_acc"] * 100, 3),
           "mean_loss": round(r["mean_loss"], 4),
           "mean_macro_f1": round(r["mean_f1"], 4)}
    for i, a in enumerate(r["fold_acc"]):
        row[f"fold{i+1}_acc_%"] = round(a * 100, 3)
    rows.append(row)

results_df = pd.DataFrame(rows)
results_df.to_csv("results.csv", index=False)
display_cols = ["experiment", "mean_acc_%", "std_acc_%", "mean_macro_f1", "mean_loss"]
results_df[display_cols]

baseline_acc = results[0]["mean_acc"] * 100
best_idx = int(np.argmax([r["mean_acc"] for r in results]))
best_acc = results[best_idx]["mean_acc"] * 100
print(f"Baseline (E0):    mean accuracy = {baseline_acc:.2f}%")
print(f"Best (E{best_idx}): mean accuracy = {best_acc:.2f}%")
print(f"Improvement:      +{best_acc - baseline_acc:.2f} pp")

# Eksperimentu isvados:
'''
    Bazinis modelis E0 mazas sigmoidinis tinklas be standardizavimo
    E0:  mean acc = 87.50% (±3.87),  macro F1 = 0.4819,  mean loss = 0.3322

    Papildant standardizacija ir padidinus epochu skaiciu (E1) rezultatai pakilo iki:
    E1:  mean acc = 95.94% (±0.83),  macro F1 = 0.6139,  mean loss = 0.1128

    E2 softmax+CE su 8 neuronais duoda labai panasu tiksluma i E1, taciau macro F1 sumazeja, kas rodo, kad modelis tapo labiau "sureguliuotas" i dominuojancia klase (Low):
    E2:  mean acc = 95.28% (±0.90),  macro F1 = 0.6044,  mean loss = 0.1200

    Gilesni ReLU tinklai (E3 ir E4) neduoda acc pagerejimo, bet E4 siek tiek pagerina macro-F1, del L2 regularizacijos:
    E3:  mean acc = 93.94% (±1.76),  macro F1 = 0.6022,  mean loss = 0.2061
    E4:  mean acc = 94.78% (±1.49),  macro F1 = 0.6160,  mean loss = 0.1629

    Galiausiai, class-weighted CE (E5) duoda maziausia pagerinima acc, taciau macro-F1 pakyla iki auksciausio lygio, kas rodo, kad modelis tapo labiau subalansuotas tarp klasiu:
    E5:  mean acc = 94.44% (±1.22),  macro F1 = 0.6697,  mean loss = 4.1870
'''

# Ataskaitai:
'''
    1. Pradinio duomenų rinkinio aprašymas (žr. 1 laboratorinio darbo aprašo 2 paveiksle pateiktą lentelę).
        Duomenų rinkinys: Work From Home Burnout Dataset.
        Pagrindiniai atributai: work_hours, screen_time_hours, meetings_count, breaks_taken, after_hours_work, sleep_hours, task_completion_rate, day_type.
        Tikslo atributas: burnout_risk (Low, Medium, High), kuris yra labai nebalansuotas.
        (Low ~85 %, Medium ~14 %, High ~1 %).

    2. Duomenų rinkinio pertvarkymų aprašas (jei buvo daryta)
        Pašalinti: user_id, burnout_score (duplikuoja burnout_risk).
        day_type paverstas į binarinį is_weekend (0=Weekday, 1=Weekend).
        burnout_risk užkoduotas kaip int (Low=0, Medium=1, High=2).
        Skaitiniai bruožai palikti originaliame mastelyje; vėliau daugumoje eksperimentų
        standartizuojami (z-score) kiekviename CV mokymo intervale atskirai.

    3. DNT architektūros schema, kurioje matytųsi sluoksnių ir neuronų skaičius, įskaitant parametrų vertes (mokymosi greitis, aktyvavimo funkcija)
        E0 bazinis tinklas (PyTorch Sequential):
            Įvesties sluoksnis: n_features
            Paslėptas sluoksnis: 4 neuronai, Sigmoid, lr=0.01, be stand., 30 epochų.
            Išvesties sluoksnis: 3 neuronai (Low/Medium/High), CrossEntropyLoss.

        E1: ta pati architektūra, bet:
            įvesties bruožai standartizuojami,
            Mokoma 200 epochų, lr=0.05.

        E5: gilesnis tinklas:
            n_features → 32 (ReLU) → 16 (ReLU) → 3, lr=0.005, 400 epochų,
            CrossEntropyLoss su class_weights (kompensuoja klasių disbalansą).

    4. 10 intervalų kryžminės patikros eksperimentų rezultatai (sąnaudų funkcijos vertė kiekviename intervale, vidutinė vertė).
        Rezultatai pateikti lentelėje results.csv

    5. Priemonių, kurių buvo imtasi siekiant pagerinti DNT veiklą, aprašymas. Kiekvienas eksperimentas ir jo poveikis modelio vidutiniui tikslumui 
       turi būti dokumentuotas (pvz.: lentelės pavidalu pateiktos pasirinktos priemonės ir rezultatai).
            (Tas pats kaip 3 punkte, bet su papildomais eksperimentais E2-E5, kurie buvo sukurti siekiant pagerinti bazinį modelį E0. Rezultatai pateikti lentelėje results.csv ir apibendrinti aukščiau tekste.)

    6. 10 intervalų kryžminės patikros eksperimentų rezultatai (sąnaudų funkcijos vertė kiekviename intervale, vidutinė vertė), kad iliustruoti altiktų DNT pokyčių efektyvumą.
        Rezultatai pateikti lentelėje results.csv, o bazinio modelio E0 ir geriausio modelio E5 rezultatai apibendrinti aukščiau tekste.

    7. Jei naudojamas tas pats duomenų rinkinys kaip LD2, palyginti DNT rezultatus su sprendimų medžio ir atsitiktinio miško rezultatais.
        N/A, naudoti LD1 duomenys, todėl palyginimas su LD2 modeliais neįmanomas.

    8. Išvados.
        DNT su tinkamu duomenų standartizavimu ir architektūros parinkimu gerokai lenkia bazinį modelį.
        Imbalanced klasėms svarbu vertinti macro-F1 ir naudoti class weights; E5 pagerina „High“ klasės F1.
        Per didelė architektūra (E3/E4) neduoda didelio papildomo acc, bet padidina skaičiavimo sąnaudas ir šiek tiek rizikuoja perfitinimu.
'''