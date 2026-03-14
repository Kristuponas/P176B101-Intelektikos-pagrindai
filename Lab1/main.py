# Link: https://www.kaggle.com/datasets/sonalshinde123/work-from-home-employee-burnout-dataset
# !pip install pandas
# !pip install matplotlib
# !pip install seaborn

import pandas as pd
import sys
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import itertools
import seaborn as sns

sys.stdout.reconfigure(encoding='utf-8')

def correlationDescription(r):
    abs_r = abs(r)
    
    if abs_r < 0.2:
        strength = "labai silpna"
    elif abs_r < 0.4:
        strength = "silpna"
    elif abs_r < 0.6:
        strength = "vidutinė"
    elif abs_r < 0.8:
        strength = "stipri"
    else:
        strength = "labai stipri"
    
    if r > 0:
        direction = "teigiama"
    elif r < 0:
        direction = "neigiama"
    else:
        direction = "nėra"
    
    return f"{strength} {direction} koreliacija"

df = pd.read_csv('work_from_home_burnout_dataset.csv')

# 2. Atlikti duomenų rinkinio kokybės analizę (žr. 2 pav.). Kiekvienam tolydinio tipo atributui
# paskaičiuoti:

print("2. Atlikti duomenų rinkinio kokybės analizę (žr. 2 pav.). Kiekvienam tolydinio tipo atributui paskaičiuoti:")

# - masyvas reikšmių, kurios turi skaitinę reikšmę
num_df = df.select_dtypes(include='number').drop(columns=['user_id', 'after_hours_work', 'breaks_taken'], errors='ignore')

# - bendrą reikšmių skaičių
total_count_num = num_df.count()
total_rows_num = len(num_df)

# - trūkstamų reikšmių procentą
missing_count_num = num_df.isna().sum()
missing_percentage_num = (missing_count_num / total_rows_num) * 100

print("\nBendras reikšmių skaičius kiekvienam atributui:")
print(total_count_num)
print("\nTrūkstamų reikšmių procentas kiekvienam atributui:")
print(missing_percentage_num)

# kardinalumą (kardinalumas matematikoje yra aibės savybė, apibendrinanti baigtinės aibės 
# narių kiekio sąvoką. Paprasčiau tariant kiek yra skirtingų atributo reikšmių. Pavyzdžiui 
# lyties atributo kardinalumas lygus 2 – t.y., lytis gali turėti tik dvi reikšmes)

card_values_num = num_df.nunique()

print("\nKardinalumas kiekvienam atributui:")
print(card_values_num)

# - minimalią (min) ir maksimalią (max) reikšmes

min_values = num_df.min()
max_values = num_df.max()

print("\nMinimalių reikšmių kiekvienam atributui:")
print(min_values)
print("\nMaksimalių reikšmių kiekvienam atributui:")
print(max_values)

# - 1-ąją ir 3- ją kvartilius

q1 = num_df.quantile(0.25).round(2)
q3 = num_df.quantile(0.75).round(2)

print("\nPirmojo kvartilio reikšmės kiekvienam atributui:")
print(q1)
print("\nTrečiojo kvartilio reikšmės kiekvienam atributui:")
print(q3)

# - vidurkį

mean_values = num_df.mean().round(2)

print("\nVidurkis kiekvienam atributui:")
print(mean_values)

# - medianą

median_values = num_df.median().round(2)

print("\nMediana kiekvienam atributui:")
print(median_values)

# - standartinį nuokrypį

std_values = num_df.std().round(2)

print("\nStandartinis nuokrypis kiekvienam atributui:")
print(std_values)

# 3. Kiekvienam kategorinio tipo atributui paskaičiuoti:

print("\n3. Kiekvienam kategorinio tipo atributui paskaičiuoti:")

# - masyvas reikšmių, kurios turi kategorinę reikšmę
cat_df = df.select_dtypes(include=['object', 'category', 'string'])
cat_df["after_hours_work"] = df["after_hours_work"]
cat_df["breaks_taken"] = df["breaks_taken"]

# - bendrą reikšmių skaičių
total_count_cat = cat_df.count()
total_rows_cat = len(cat_df)

# - trūkstamų reikšmių procentą
missing_count_cat = cat_df.isna().sum()
missing_percentage_cat = (missing_count_cat / total_rows_cat) * 100

print("\nBendras reikšmių skaičius kiekvienam atributui:")
print(total_count_cat)
print("\nTrūkstamų reikšmių procentas kiekvienam atributui:")
print(missing_percentage_cat)

# - kardinalumą

card_values_cat = cat_df.nunique()

print("\nKardinalumas kiekvienam atributui:")
print(card_values_cat)

# - pirma moda, dažnumas ir procentas
# - antra moda, dažnumas ir procentas

# Kiekvienam kategoriniui atributui naikintos trūkstamos reikšmės ir skaičiuoti dažniai
# Maždaug toki dictionary gaunam:
#print(freq)
#day_type                    {'Weekday': 876, 'Weekend': 924}
#burnout_risk        {'Low': 1527, 'Medium': 253, 'High': 20}
#after_hours_work                           {0: 1154, 1: 646}
#breaks_taken        {2: 340, 1: 361, 4: 394, 3: 345, 5: 360}

freq = cat_df.apply(lambda x: Counter(x.dropna()))
res = {}

for col, counter in freq.items():
    # Kiekvieno kategorinio atributo reikšmių sk.
    total = sum(counter.values())

    top2 = counter.most_common(2)
    
    first_mode, first_count = top2[0] if len(top2) > 0 else (None, 0)
    second_mode, second_count = top2[1] if len(top2) > 0 else (None, 0)
    
    res[col] = {
        "1_mode": first_mode,
        "1_count": first_count,
        "1_percent": round(first_count / total * 100, 2) if total > 0 else 0,
        "2_mode": second_mode,
        "2_count": second_count,
        "2_percent": round(second_count / total * 100, 2) if total > 0 else 0
    }

for col, stats in res.items():
    print(f"\nAtributas: {col}")
    print(f"1 moda: {stats['1_mode']}, Dažnumas: {stats['1_count']}, Procentas: {stats['1_percent']}%")
    print(f"2 moda: {stats['2_mode']}, Dažnumas: {stats['2_count']}, Procentas: {stats['2_percent']}%")

# 4. Nupaišyti atributų histogramas (rekomenduotinas stulpelių skaičius randamas 
# formule: 1 + 3.22 ∙ 𝑙𝑜𝑔𝑒 𝑛, kur n imties dydis). Ataskaitoje pateikti aprašymus, 
# koks tai pasiskirstymas (pvz., normalusis, vien(a)modalis, eksponentinis ir t.t.) 
# ir kokias išvadas pagal tai galima formuluoti (žr. 2 paskaita, 41-43 skaidres).
'''
n = len(num_df)
col_count = int(1 + 3.22 * np.log(n))

for column in num_df.columns:
    plt.figure(figsize=(8, 5))
    plt.hist(num_df[column].dropna(), bins=col_count, edgecolor='black')
    plt.title(f'{column} atributo histograma')
    plt.xlabel(column)
    plt.ylabel('Dažnis')
    plt.grid(axis='y', alpha=0.75)
    plt.show()

for column in cat_df.columns:
    counts = cat_df[column].value_counts(dropna=False)
    plt.figure(figsize=(8, 5))
    plt.bar(counts.index.astype(str), counts.values)
    plt.title(f'{column} atributo stulpelinė diagrama')
    plt.xlabel(column)
    plt.ylabel('Dažnis')
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.75)
    
    plt.show()
'''

# 7. Nustatyti sąryšius tarp atributų panaudojant vizualizacijos būdus:

'''
Tolydinio tipo atributams: naudojant „scatter plot“ tipo diagramą (žr. 3 paskaita, 5 skaidrė ) pateikti
kelis (2-3) pavyzdžius su stipria tiesine atributų priklausomybe (tiesioginė arba atvirkštinė koreliacija)
bei kelis pavyzdžius su tarpusavyje nekoreliuojančiais (silpnai koreliuojančiais) atributais.
Pakomentuoti rezultatus.
'''

'''
pairs = [
    ("work_hours", "screen_time_hours"),
    ("task_completion_rate", "burnout_score"),
    ("screen_time_hours", "sleep_hours")
]

for x, y in pairs:
    
    data = num_df[[x, y]].dropna()
    r = data[x].corr(data[y])
    
    interpretation = correlationDescription(r)
    
    plt.figure(figsize=(6,5))
    plt.scatter(data[x], data[y], alpha=0.6, color='black', edgecolors='black', s=20)
    plt.title(f"{x} ir {y}\nr = {r:.3f} ({interpretation})")
    plt.xlabel(x)
    plt.ylabel(y)
    plt.grid(alpha=0.3)
    plt.show()
'''

'''
Pateikti SPLOM diagramą (Scatter Plot Matrix) (žr. 3 paskaita, 6 skaidrė).
'''

'''
splom_columns = [
    "work_hours",
    "screen_time_hours",
    "task_completion_rate",
    "burnout_score",
    "sleep_hours"
]

splom_df = df[splom_columns].dropna()

pd.plotting.scatter_matrix(
    splom_df,
    figsize=(12, 12),
    diagonal='hist',
    alpha=0.6
)

plt.show()
'''

'''
Kategorinio tipo atributams: naudojant „bar plot“ tipo diagramą pateikti keletą (2-3) atributų
priklausomybės pavyzdžių ir pakomentuoti rezultatus (žr. 3 paskaita, 7-9 skaidres).
'''

'''
ct = pd.crosstab(df["breaks_taken"], df["day_type"], normalize="index") * 100

ct.plot(kind="bar", figsize=(8,5))
plt.title("day_type pasiskirstymas pagal breaks_taken")
plt.ylabel("Procentai")
plt.xticks(rotation=0)
plt.grid(axis="y", alpha=0.3)
plt.show()

ct = pd.crosstab(df["after_hours_work"], df["burnout_risk"], normalize="index") * 100

ct.plot(kind="bar", figsize=(8,5))
plt.title("burnout_risk pasiskirstymas pagal after_hours_work")
plt.ylabel("Procentai")
plt.xticks(rotation=0)
plt.grid(axis="y", alpha=0.3)
plt.show()
'''

'''
Pateikti keletą (2-3) histogramų (žr. 3 paskaita, 12-14 skaidres) ir „box plot“ diagramų pavyzdžių (žr. 3
paskaita, 15 skaidrę), vaizduojančių sąryšius tarp kategorinio (pavyzdys pateiktas pav.3) ir tolydinio
tipo kintamųjų .
'''

'''
# histogramos
plt.figure(figsize=(8, 5))
for val in df["after_hours_work"].unique():
    subset = df[df["after_hours_work"] == val]
    plt.hist(subset["burnout_score"], bins=15, alpha=0.5, label=f"After hours: {val}")
    
plt.title("Burnout Score pasiskirstymas pagal After Hours Work")
plt.xlabel("Burnout Score")
plt.ylabel("Dažnis")
plt.legend()
plt.grid(axis="y", alpha=0.3)
plt.show()

plt.figure(figsize=(8, 5))
for val in df["day_type"].unique():
    subset = df[df["day_type"] == val]
    plt.hist(subset["work_hours"], bins=15, alpha=0.5, label=f"Day Type: {val}")
    
plt.title("Darbo valandų pasiskirstymas pagal dienos tipą")
plt.xlabel("Work Hours")
plt.ylabel("Dažnis")
plt.legend()
plt.grid(axis="y", alpha=0.3)
plt.show()

#box plotai
plt.figure(figsize=(8, 5))
df.boxplot(column="burnout_score", by="after_hours_work", grid=True)
plt.title("Burnout Score pagal After Hours Work")
plt.suptitle("")
plt.xlabel("After Hours Work")
plt.ylabel("Burnout Score")
plt.grid(axis="y", alpha=0.3)
plt.show()

plt.figure(figsize=(8, 5))
df.boxplot(column="work_hours", by="day_type", grid=True)
plt.title("Darbo valandos pagal dienos tipą")
plt.suptitle("")
plt.xlabel("Day Type")
plt.ylabel("Work Hours")
plt.grid(axis="y", alpha=0.3)
plt.show()
'''

# 8. Paskaičiuoti kovariacĳos ir koreliacĳos reikšmes tarp tolydinio tipo atributų ir grafiškai
# atvaizduoti koreliacĳos matricą (žr. 3 paskaita, 24-34 skaidres). Rezultatus pakomentuoti.

'''
cov_matrix = num_df.cov()
corr = num_df.corr(method='spearman')

plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=45, ha="right")
plt.title("Koreliacijos matrica")
plt.show()

print("Kovariacijos matrica:")
print(cov_matrix)
print("\nKoreliacijos matrica:")
print(corr)
'''

# 9. Atlikti duomenų normalizaciją (rėžiai [0;1] arba [-1;1]) (žr. 3 paskaita, 35-37 skaidres).

def normalize_range(series, low=0, high=1):
    min_val = series.min()
    max_val = series.max()
    return (series - min_val) / (max_val - min_val) * (high - low) + low


normalized_df = num_df.apply(lambda x: normalize_range(x, low=0, high=1))
normalized_df.head() # kad belenkiek neprintintu

print(normalized_df)

normalized_df = num_df.apply(lambda x: normalize_range(x, low=0, high=1))

sns.pairplot(normalized_df)
plt.suptitle("SPLOM - Normalizuoti duomenys", y=1.02)
plt.show()

# 10. Kategorinio tipo kintamuosius paversti į tolydinio tipo kintamuosius.

encoded_cat_df = pd.get_dummies(cat_df, drop_first=False) # drop_first atskiria kategorija i jos kintamuosius

print(encoded_cat_df)