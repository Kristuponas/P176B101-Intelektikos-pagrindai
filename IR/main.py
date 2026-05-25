import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples
from clustergram import Clustergram
import warnings
warnings.filterwarnings('ignore')

# 1. Duomenys
print("=" * 60)
print("1. DUOMENŲ ĮKĖLIMAS IR VALYMAS")
print("=" * 60)

df = pd.read_csv('body.csv')
print(f"Pradinis dydis: {df.shape[0]} eilučių, {df.shape[1]} stulpelių")

continuous_cols = [
    col for col in df.select_dtypes(include=[np.number]).columns
    if df[col].nunique() >= 50  # pašalinti kategorinio pobūdžio skaitmeninius stulpelius
]
categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
print(f"Tolydžiai atributai: {continuous_cols}")
print(f"Kategoriniai (neįtraukiami): {categorical_cols}")

skip = ['sit-ups counts']
for col in continuous_cols:
    if col in skip:
        continue
    before = len(df)
    df.drop(df[df[col] <= 0].index, inplace=True)
    removed = before - len(df)
    if removed > 0:
        print(f"  Pašalinta {removed} neteisingų įrašų iš '{col}'")

print("\nEkstremalių reikšmių šalinimas (IQR metodas):")
total_outliers = 0
for col in continuous_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower) | (df[col] > upper)]
    df.drop(outliers.index, inplace=True)
    if len(outliers) > 0:
        print(f"  '{col}': pašalinta {len(outliers)} ekstremalių reikšmių")
    total_outliers += len(outliers)

print(f"Iš viso pašalinta ekstremalių reikšmių: {total_outliers}")
print(f"Galutinis dydis: {df.shape[0]} eilučių")

X_raw = df[continuous_cols].values
scaler = StandardScaler()
X = scaler.fit_transform(X_raw)
X_df = pd.DataFrame(X, columns=continuous_cols)
print(f"Duomenys standartizuoti. Atributai ({len(continuous_cols)}): {continuous_cols}")

# 2. Pagalbines funkcijos

SIL_SAMPLE = 3000
N_INIT = 5

def kmeans_inertia_silhouette(data, k, random_state=42):
    km = KMeans(n_clusters=k, random_state=random_state, n_init=N_INIT)
    labels = km.fit_predict(data)
    inertia = km.inertia_
    n = len(data)
    if k > 1:
        sample = min(SIL_SAMPLE, n)
        sil = silhouette_score(data, labels, sample_size=sample, random_state=random_state)
    else:
        sil = 0.0
    return inertia, sil, labels, km

def plot_silhouette_standalone(data, labels, k, title="", fname=""):
    """Silueto diagrama – atskiras failas."""
    sil_vals = silhouette_samples(data, labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim([-0.2, 1.0])
    ax.set_ylim([0, len(data) + (k + 1) * 10])
    y_lower = 10
    colors = plt.cm.get_cmap('tab10', k)
    for i in range(k):
        ith_sil = np.sort(sil_vals[labels == i])
        size = ith_sil.shape[0]
        y_upper = y_lower + size
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_sil,
                         facecolor=colors(i), edgecolor=colors(i), alpha=0.7)
        ax.text(-0.05, y_lower + 0.5 * size, str(i))
        y_lower = y_upper + 10
    avg = sil_vals.mean()
    ax.axvline(x=avg, color="red", linestyle="--", linewidth=1.2)
    ax.set_xlabel("Silueto koeficientų reikšmės")
    ax.set_ylabel("Klasteris")
    ax.set_title(title if title else f"Silueto diagrama (k={k})")
    ax.text(avg + 0.01, ax.get_ylim()[1] * 0.95, f"vid.={avg:.3f}",
            color="red", fontsize=9)
    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    return avg

# 3. Dviejų atributų klasterizavimas

print("\n" + "=" * 60)
print("2. DVIEJŲ ATRIBUTŲ KLASTERIZAVIMAS")
print("=" * 60)

K_RANGE = range(2, 9)
all_pairs = list(combinations(continuous_cols, 2))
print(f"Tikrinamos {len(all_pairs)} atributų poros, k nuo 2 iki 8...")

pair_results = []
for col_a, col_b in all_pairs:
    data_pair = X_df[[col_a, col_b]].values
    best_sil = -1
    best_k = 2
    row = {"pair": (col_a, col_b)}
    for k in K_RANGE:
        inertia, sil, _, _ = kmeans_inertia_silhouette(data_pair, k)
        row[f"inertia_k{k}"] = round(inertia, 2)
        row[f"sil_k{k}"] = round(sil, 4)
        if sil > best_sil:
            best_sil = sil
            best_k = k
    row["best_sil"] = best_sil
    row["best_k"] = best_k
    pair_results.append(row)

pair_df = pd.DataFrame(pair_results).sort_values("best_sil", ascending=False).reset_index(drop=True)

print("\nGeriausios atributų poros (top 15 pagal max silueto koef.):")
print(f"{'Nr':>3} | {'Atributų pora':<45} | {'Best k':>6} | {'Max sil':>7}")
print("-" * 70)
for i, row in pair_df.head(15).iterrows():
    a, b = row['pair']
    print(f"{i+1:>3} | {a+' × '+b:<45} | {row['best_k']:>6} | {row['best_sil']:>7.4f}")

table_rows = []
for i, row in pair_df.head(15).iterrows():
    a, b = row['pair']
    r = {"Nr": i+1, "Pora": f"{a} × {b}"}
    for k in K_RANGE:
        r[f"Inercija k={k}"] = row[f"inertia_k{k}"]
        r[f"Siluetas k={k}"] = row[f"sil_k{k}"]
    r["Geriausias k"] = row['best_k']
    r["Max siluetas"] = row['best_sil']
    table_rows.append(r)
table2d = pd.DataFrame(table_rows)
table2d.to_csv("lentele_2d.csv", index=False, encoding='utf-8-sig')
print("\nPilna lentelė išsaugota: lentele_2d.csv")

# Top 3 poros – ATSKIRI grafikai
TOP3 = pair_df.head(3)
print("\nKuriant grafikus top 3 poroms (atskiri failai)...")

for rank, (_, row) in enumerate(TOP3.iterrows(), 1):
    col_a, col_b = row['pair']
    best_k = int(row['best_k'])
    data_pair = X_df[[col_a, col_b]].values
    safe_a = col_a.replace(' ', '_').replace('%','proc')
    safe_b = col_b.replace(' ', '_').replace('%','proc')

    inertia_best, sil_best, labels, km = kmeans_inertia_silhouette(data_pair, best_k)
    colors_map = plt.cm.get_cmap('tab10', best_k)

    # Grafikas 1: Scatter
    fig, ax = plt.subplots(figsize=(7, 6))
    for c in range(best_k):
        mask = labels == c
        ax.scatter(data_pair[mask, 0], data_pair[mask, 1],
                   color=colors_map(c), s=8, alpha=0.5, label=f"Klasteris {c}")
    centers = km.cluster_centers_
    ax.scatter(centers[:, 0], centers[:, 1],
               color='black', s=120, marker='X', zorder=5, label='Centroidai')
    ax.set_xlabel(col_a); ax.set_ylabel(col_b)
    ax.set_title(f"Pora #{rank}: {col_a} × {col_b}\nDuomenų klasterizacija (k={best_k}), Siluetas={sil_best:.4f}")
    ax.legend(fontsize=7, markerscale=1.5)
    plt.tight_layout()
    fname = f"2d_pora{rank}_{safe_a}_{safe_b}_scatter.png"
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Išsaugota: {fname}")

    # Grafikas 2: Inercija
    inertias = [row[f'inertia_k{k}'] for k in K_RANGE]
    silhouettes = [row[f'sil_k{k}'] for k in K_RANGE]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(list(K_RANGE), inertias, 'b-o', linewidth=2, markersize=6)
    ax.axvline(x=best_k, color='red', linestyle='--', alpha=0.7, label=f'Geriausias k={best_k}')
    ax.set_xlabel("Klasterių skaičius k"); ax.set_ylabel("Inercija", color='blue')
    ax.tick_params(axis='y', labelcolor='blue')
    ax2t = ax.twinx()
    ax2t.plot(list(K_RANGE), silhouettes, 'g--s', linewidth=1.5, markersize=5, alpha=0.7)
    ax2t.set_ylabel("Silueto koef.", color='green')
    ax2t.tick_params(axis='y', labelcolor='green')
    ax.set_title(f"Pora #{rank}: {col_a} × {col_b}\nInercija ir silueto koef. vs k")
    ax.legend(fontsize=8); ax.set_xticks(list(K_RANGE)); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fname = f"2d_pora{rank}_{safe_a}_{safe_b}_inercija.png"
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Išsaugota: {fname}")

    # Grafikas 3: Silueto diagrama
    fname = f"2d_pora{rank}_{safe_a}_{safe_b}_siluetas.png"
    plot_silhouette_standalone(data_pair, labels, best_k,
        title=f"Pora #{rank}: {col_a} × {col_b}\nSilueto diagrama (k={best_k})", fname=fname)
    print(f"  Išsaugota: {fname}")

# 4. Trijų atributų klasterizavimas

print("\n" + "=" * 60)
print("3. TRIJŲ ATRIBUTŲ KLASTERIZAVIMAS")
print("=" * 60)

all_triples = list(combinations(continuous_cols, 3))
print(f"Tikrinami {len(all_triples)} atributų trejetai, k nuo 2 iki 8...")

triple_results = []
for cols_3 in all_triples:
    data_3 = X_df[list(cols_3)].values
    best_sil = -1
    best_k = 2
    row = {"triple": cols_3}
    for k in K_RANGE:
        inertia, sil, _, _ = kmeans_inertia_silhouette(data_3, k)
        row[f"inertia_k{k}"] = round(inertia, 2)
        row[f"sil_k{k}"] = round(sil, 4)
        if sil > best_sil:
            best_sil = sil
            best_k = k
    row["best_sil"] = best_sil
    row["best_k"] = best_k
    triple_results.append(row)

triple_df = pd.DataFrame(triple_results).sort_values("best_sil", ascending=False).reset_index(drop=True)

print("\nGeriausi atributų trejetai (top 15):")
print(f"{'Nr':>3} | {'Atributų treijetas':<55} | {'Best k':>6} | {'Max sil':>7}")
print("-" * 80)
for i, row in triple_df.head(15).iterrows():
    a, b, c = row['triple']
    print(f"{i+1:>3} | {a+' × '+b+' × '+c:<55} | {row['best_k']:>6} | {row['best_sil']:>7.4f}")

table_rows3 = []
for i, row in triple_df.head(15).iterrows():
    a, b, c = row['triple']
    r = {"Nr": i+1, "Treijetas": f"{a} × {b} × {c}"}
    for k in K_RANGE:
        r[f"Inercija k={k}"] = row[f"inertia_k{k}"]
        r[f"Siluetas k={k}"] = row[f"sil_k{k}"]
    r["Geriausias k"] = row['best_k']
    r["Max siluetas"] = row['best_sil']
    table_rows3.append(r)
table3d = pd.DataFrame(table_rows3)
table3d.to_csv("lentele_3d.csv", index=False, encoding='utf-8-sig')
print("\nPilna lentelė išsaugota: lentele_3d.csv")

# Top 3 trejetai – ATSKIRI grafikai
TOP3_T = triple_df.head(3)
print("\nKuriant 3D grafikus top 3 trejetams (atskiri failai)...")

for rank, (_, row) in enumerate(TOP3_T.iterrows(), 1):
    col_a, col_b, col_c = row['triple']
    best_k = int(row['best_k'])
    data_3 = X_df[[col_a, col_b, col_c]].values
    safe_a = col_a.replace(' ', '_').replace('%','proc')
    safe_b = col_b.replace(' ', '_').replace('%','proc')
    colors_map = plt.cm.get_cmap('tab10', best_k)

    inertia_best, sil_best, labels, km = kmeans_inertia_silhouette(data_3, best_k)

    # Grafikas 1: 3D scatter
    fig = plt.figure(figsize=(8, 7))
    ax1 = fig.add_subplot(111, projection='3d')
    for c in range(best_k):
        mask = labels == c
        ax1.scatter(data_3[mask, 0], data_3[mask, 1], data_3[mask, 2],
                    color=colors_map(c), s=5, alpha=0.4, label=f"K{c}")
    ax1.set_xlabel(col_a[:10]); ax1.set_ylabel(col_b[:10]); ax1.set_zlabel(col_c[:10])
    ax1.set_title(f"Treijetas #{rank}: {col_a} × {col_b} × {col_c}\n3D klasterizacija (k={best_k}), Siluetas={sil_best:.4f}")
    ax1.legend(fontsize=7)
    plt.tight_layout()
    fname = f"3d_treijetas{rank}_{safe_a}_{safe_b}_scatter.png"
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Išsaugota: {fname}")

    # Grafikas 2: Inercija
    inertias = [row[f'inertia_k{k}'] for k in K_RANGE]
    silhouettes = [row[f'sil_k{k}'] for k in K_RANGE]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(list(K_RANGE), inertias, 'b-o', linewidth=2, markersize=6)
    ax.axvline(x=best_k, color='red', linestyle='--', alpha=0.7, label=f'k={best_k}')
    ax.set_xlabel("Klasterių skaičius k"); ax.set_ylabel("Inercija", color='blue')
    ax.tick_params(axis='y', labelcolor='blue')
    ax2t = ax.twinx()
    ax2t.plot(list(K_RANGE), silhouettes, 'g--s', linewidth=1.5, markersize=5, alpha=0.7)
    ax2t.set_ylabel("Silueto koef.", color='green')
    ax2t.tick_params(axis='y', labelcolor='green')
    ax.set_title(f"Treijetas #{rank}: {col_a} × {col_b} × {col_c}\nInercija ir siluetas vs k")
    ax.legend(fontsize=8); ax.set_xticks(list(K_RANGE)); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fname = f"3d_treijetas{rank}_{safe_a}_{safe_b}_inercija.png"
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Išsaugota: {fname}")

    # Grafikas 3: Silueto diagrama
    fname = f"3d_treijetas{rank}_{safe_a}_{safe_b}_siluetas.png"
    plot_silhouette_standalone(data_3, labels, best_k,
        title=f"Treijetas #{rank}: {col_a} × {col_b} × {col_c}\nSilueto diagrama (k={best_k})", fname=fname)
    print(f"  Išsaugota: {fname}")

# 5. M-dimensijų klasterizavimas

print("\n" + "=" * 60)
print("4. M-DIMENSIJŲ KLASTERIZAVIMAS")
print("=" * 60)

m_results = []
all_labels_by_k = {}
print("Skaičiuojama m-dimensijų inercija ir silueto koef. kiekvienam k...")
for k in K_RANGE:
    inertia, sil, labels, km = kmeans_inertia_silhouette(X, k)
    m_results.append({"k": k, "Inercija": round(inertia, 2), "Siluetas": round(sil, 4)})
    all_labels_by_k[k] = labels
    print(f"  k={k}: inercija={inertia:.2f}, siluetas={sil:.4f}")

m_df = pd.DataFrame(m_results)
print("\nM-dimensijų klasterizavimo lentelė:")
print(m_df.to_string(index=False))
m_df.to_csv("lentele_mdim.csv", index=False, encoding='utf-8-sig')
print("Išsaugota: lentele_mdim.csv")

best_k_m = int(m_df.loc[m_df['Siluetas'].idxmax(), 'k'])
top3_k_m = m_df.nlargest(3, 'Siluetas')['k'].tolist()
print(f"Geriausias k pagal siluetą: {best_k_m}")
print(f"Top 3 k: {top3_k_m}")

# Grafikas 1: Inercija (atskiras)
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(m_df['k'], m_df['Inercija'], 'b-o', linewidth=2, markersize=8)
ax.axvline(x=best_k_m, color='red', linestyle='--', alpha=0.7, label=f'Best k={best_k_m}')
ax.set_xlabel("Klasterių skaičius k"); ax.set_ylabel("Inercija", color='blue')
ax.tick_params(axis='y', labelcolor='blue')
ax2t = ax.twinx()
ax2t.plot(m_df['k'], m_df['Siluetas'], 'g--s', linewidth=2, markersize=7, alpha=0.8)
ax2t.set_ylabel("Silueto koef.", color='green')
ax2t.tick_params(axis='y', labelcolor='green')
ax.set_title(f"M-dimensijų ({len(continuous_cols)}D) klasterizavimas\nInercija ir silueto koef. vs k (visi atributai)")
ax.set_xticks(list(K_RANGE)); ax.grid(True, alpha=0.3); ax.legend()
plt.tight_layout()
plt.savefig("mdim_inercija.png", dpi=150, bbox_inches='tight')
plt.close()
print("Išsaugota: mdim_inercija.png")

# Grafikas 2: Vidutiniai silueto koef. (atskiras)
fig, ax = plt.subplots(figsize=(8, 4))
for kidx, k in enumerate(sorted(top3_k_m)):
    labels_k = all_labels_by_k[k]
    sil_vals = silhouette_samples(X, labels_k)
    avg = sil_vals.mean()
    ax.barh(k, avg, color=plt.cm.tab10(kidx), alpha=0.8)
    ax.text(avg + 0.005, k, f"k={k}: {avg:.4f}", va='center', fontsize=9)
ax.set_xlabel("Vidutinis silueto koef."); ax.set_ylabel("k reikšmė")
ax.set_title("M-dimensijų klasterizavimas\nVidutiniai silueto koef. (top 3 k)")
ax.set_yticks(sorted(top3_k_m)); ax.grid(True, axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig("mdim_silueto_top3_bar.png", dpi=150, bbox_inches='tight')
plt.close()
print("Išsaugota: mdim_silueto_top3_bar.png")

# Grafikai 3-5: TRYS atskiros silueto diagramos geriausiam k
for k in sorted(top3_k_m):
    labels_k = all_labels_by_k[k]
    fname = f"mdim_siluetas_k{k}.png"
    plot_silhouette_standalone(X, labels_k, k,
        title=f"M-dimensijų klasterizavimas\nSilueto diagrama (k={k})", fname=fname)
    print(f"Išsaugota: {fname}")

# 6. Klasterograma
print("\n" + "=" * 60)
print("5. KLASTEROGRAMA")
print("=" * 60)

try:
    cg = Clustergram(range(2, 9), n_init=3, random_state=42, verbose=False)
    cg.fit(X)
    fig, ax = plt.subplots(figsize=(12, 7))
    cg.plot(ax=ax)
    ax.set_title("Klasterograma – K-vidurkiai (body.csv)", fontsize=13, fontweight='bold')
    ax.set_xlabel("Klasterių skaičius k")
    plt.tight_layout()
    plt.savefig("klasterograma.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Išsaugota: klasterograma.png")
except Exception as e:
    print(f"Klasterogramos klaida: {e}")

# 7. Suvestinė
print("\n" + "=" * 60)
print("SUVESTINĖ")
print("=" * 60)
print(f"\nDuomenų rinkinys po valymo: {df.shape[0]} eilučių")
print(f"Atributų skaičius (m): {len(continuous_cols)}")
print(f"\n2D – geriausias: {TOP3.iloc[0]['pair']}")
print(f"  Max siluetas: {TOP3.iloc[0]['best_sil']:.4f}, k={int(TOP3.iloc[0]['best_k'])}")
print(f"\n3D – geriausias: {TOP3_T.iloc[0]['triple']}")
print(f"  Max siluetas: {TOP3_T.iloc[0]['best_sil']:.4f}, k={int(TOP3_T.iloc[0]['best_k'])}")
print(f"\nM-dim – geriausias k: {best_k_m}")
print(f"  Siluetas: {m_df.loc[m_df['Siluetas'].idxmax(), 'Siluetas']:.4f}")