import pandas as pd
import time
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier   
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score

# -----------------------------------------
# 1. DUOMENŲ RINKINYS
# -----------------------------------------

# Dataset: https://archive.ics.uci.edu/dataset/19/car+evaluation

'''
    Ataskaitai:

    Duomenu rinkinys apima 1728 irasus, trukstamu reiksmiu nera.
    Rinkinys skirtas ivertinti ar automobilis yra tinkamas pirkimui.

    Atributai:
        1. buying (Automobilio pirkinio kaina): vhigh, high, med, low
        2. maint (Automobilio prieziuros kaina): vhigh, high, med, low
        3. doors (Durys): 2, 3, 4, 5more (5 ir daugiau)
        4. persons (Keleiviu talpa): 2, 4, more (daugiau nei 4)
        5. lug_boot (Bagažo skyriaus dydis): small, med, big
        6. safety (Saugumas): low, med, high
        7. class (Automobilio vertinimas): unacc (netinkamas), acc (tinkamas), good (geras), vgood (labai geras)

    Pasiskirstymas:
        unacc: 1210  ~70%
        acc  : 384   ~22%
        good : 69    ~4%
        vgood: 65    ~4%

    Klase yra disbalansuota, daugiausia netinkamu automobiliu, todel butent sita
    klase bus geriausiai atpazystama.
'''

df = pd.read_csv('car.data', header=None, names=[
    'buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class'
])

print('Duomenų pirmos eilutės:\n')
print(df.head())

print('\nKlasių pasiskirstymas:\n')
print(df['class'].value_counts())

print('\nTrūkstamos reikšmės:\n')
print(df.isnull().sum())

# -----------------------------------------
# 2. PROGNOZUOJAMAS ATRIBUTAS
# -----------------------------------------
# Prognozuojamas atributas: class

# -----------------------------------------
# 3. TRAIN / TEST SKAIDYMAS
# -----------------------------------------

X = df.drop('class', axis=1)
y = df['class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print('\nApmokymo aibės klasių pasiskirstymas:\n')
print(y_train.value_counts())

print('\nTestavimo aibės klasių pasiskirstymas:\n')
print(y_test.value_counts())

# -----------------------------------------
# 4. ĮVESTYS IR IŠVESTYS
# -----------------------------------------

print('\nX_train pirmos eilutės:\n')
print(X_train.head())

print('\ny_train pirmos reikšmės:\n')
print(y_train.head())

# -----------------------------------------
# KATEGORINIŲ REIKŠMIŲ KODAVIMAS
# -----------------------------------------

enc = OrdinalEncoder()
X_train_enc = enc.fit_transform(X_train)
X_test_enc = enc.transform(X_test)

classes = sorted(y.unique())

# -----------------------------------------
# 5. SPRENDIMŲ MEDŽIO SUDARYMAS
# -----------------------------------------

model = DecisionTreeClassifier(
    criterion='entropy',
    max_depth=5,
    random_state=42
)

start = time.perf_counter()
model.fit(X_train_enc, y_train)
build_time = time.perf_counter() - start

print(f'\nMedžio sudarymo laikas: {build_time:.6f} s')

# -----------------------------------------
# 6. GRAFINIS ATVAIZDAVIMAS
# -----------------------------------------

plt.figure(figsize=(24, 10))
plot_tree(
    model,
    feature_names=X.columns,
    class_names=classes,
    filled=True,
    rounded=True,
    fontsize=8
)
plt.title('Sprendimų medis')
plt.tight_layout()
plt.savefig('car_tree_preview.png', dpi=150)
plt.show()

# -----------------------------------------
# 7. TESTAVIMAS IR TIKSLUMAS
# -----------------------------------------

y_train_pred = model.predict(X_train_enc)
y_test_pred = model.predict(X_test_enc)

train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)

print('\nApmokymo accuracy:', round(train_acc, 4))
print('Testavimo accuracy:', round(test_acc, 4))

print('\nTestavimo classification report:\n')
print(classification_report(y_test, y_test_pred, zero_division=0))

cm = confusion_matrix(y_test, y_test_pred, labels=classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
disp.plot(cmap='Greens')
plt.title('Testavimo susimaišymo matrica')
plt.tight_layout()
plt.show()

# -----------------------------------------
# 8. EKSPERIMENTAS SU SKIRTINGAIS GYLIAIS
# -----------------------------------------

# Treniravimas
'''
    Ataskaitai:

    Kai gylis yra 12, modelis pasiekia geriausia tiksluma: treniravime 0.9978, testavime 0.9855.
    Didesnis gylis nei 12 sukelia overfitting'a, kai gylis yra 13, treniravimo tikslumas yra 1.0, bet testavimo tikslumas krenta iki 0.9798.
'''

depths = [5, 7, 9, 11, 12, 13]
depth_results = []

for d in depths:
    model_d = DecisionTreeClassifier(
        criterion='entropy',
        max_depth=d,
        random_state=42
    )

    start = time.perf_counter()
    model_d.fit(X_train_enc, y_train)
    elapsed = time.perf_counter() - start

    y_train_pred_d = model_d.predict(X_train_enc)
    y_test_pred_d = model_d.predict(X_test_enc)

    train_acc_d = accuracy_score(y_train, y_train_pred_d)
    test_acc_d = accuracy_score(y_test, y_test_pred_d)

    depth_results.append({
        'depth': d,
        'build_time': elapsed,
        'train_acc': train_acc_d,
        'test_acc': test_acc_d
    })

    print(f'gylis={d}  laikas={elapsed:.6f}s  apmokymas={train_acc_d:.4f}  testavimas={test_acc_d:.4f}')

depth_df = pd.DataFrame(depth_results)

best_depth = int(depth_df.loc[depth_df['test_acc'].idxmax(), 'depth'])
print(f'\nGeriausias gylis pagal testavimo tikslumą: {best_depth}')

fig, ax1 = plt.subplots(figsize=(8, 4))
ax2 = ax1.twinx()

ax1.plot(depth_df['depth'], depth_df['train_acc'], 'o-', label='Apmokymo tikslumas')
ax1.plot(depth_df['depth'], depth_df['test_acc'], 's-', label='Testavimo tikslumas')
ax2.bar(depth_df['depth'], depth_df['build_time'], alpha=0.25, label='Formavimo laikas (s)')

ax1.set_xlabel('Maksimalus gylis')
ax1.set_ylabel('Tikslumas')
ax2.set_ylabel('Formavimo laikas (s)')
ax1.set_xticks(depths)
ax1.set_ylim(0.5, 1.05)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right')

plt.title('Sprendimų medis — tikslumas ir formavimo laikas pagal gylį')
plt.tight_layout()
plt.savefig('depth_experiment.png', dpi=150)
plt.show()

# -----------------------------------------
# 9. ATSITIKTINIS MIŠKAS (5 MEDŽIAI)
# -----------------------------------------

forest = RandomForestClassifier(
    n_estimators=5,       # 5 medžiai
    max_depth=best_depth, # tavo rastas geriausias gylis (12)
    random_state=42
)

start = time.perf_counter()
forest.fit(X_train_enc, y_train)
forest_time = time.perf_counter() - start

y_pred_forest = forest.predict(X_test_enc)

forest_acc = accuracy_score(y_test, y_pred_forest)

print('\n--- ATSITIKTINIS MIŠKAS ---')
print(f'Miško sudarymo laikas: {forest_time:.6f} s')
print(f'Miško tikslumas: {forest_acc:.4f}')

print('\nClassification report:\n')
print(classification_report(y_test, y_pred_forest, zero_division=0))

cm_forest = confusion_matrix(y_test, y_pred_forest, labels=classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm_forest, display_labels=classes)
disp.plot(cmap='Blues')
plt.title('Random Forest susimaišymo matrica')
plt.tight_layout()
plt.show()