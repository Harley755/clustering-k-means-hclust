"""
=============================================================================
TP2 — Clustering K-means / Hclust
IFRI — Master 1 Génie Logiciel
Script principal : main_clustering.py
=============================================================================

Point d'entrée unique du TP2. Lance dans l'ordre :
    1. Génération de la dataset
    2. Pipeline K-means complet
    3. Pipeline Clustering Hiérarchique complet
    4. Analyse comparative K-means vs Hclust
    5. Génération du rapport de synthèse

Usage :
    python main_clustering.py
=============================================================================
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics       import adjusted_rand_score

# ─── Ajout du dossier courant au PATH pour les imports relatifs ─────────────
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR    = SCRIPTS_DIR
sys.path.insert(0, SCRIPTS_DIR)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
DATA_PATH = os.path.join(BASE_DIR, "data", "clients_marketifri.csv")
FIG_DIR   = os.path.join(BASE_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

PALETTE = ["#E63946", "#457B9D", "#2A9D8F", "#E9C46A", "#F4A261"]

plt.rcParams.update({
    "figure.dpi": 150, "font.family": "DejaVu Sans",
    "axes.spines.top": False, "axes.spines.right": False,
})


def separator(titre: str):
    print("\n" + "█" * 60)
    print(f"  {titre}")
    print("█" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 1 — GÉNÉRATION DE LA DATASET
# ─────────────────────────────────────────────────────────────────────────────
def etape1_generer_dataset():
    separator("ÉTAPE 1 — GÉNÉRATION DE LA DATASET")
    if not os.path.exists(DATA_PATH):
        import generate_dataset
        print("  Dataset générée.")
    else:
        print(f"  Dataset déjà présente : {DATA_PATH}")
        sz = os.path.getsize(DATA_PATH) / 1024
        print(f"  Taille : {sz:.1f} Ko")


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 2 — PIPELINE K-MEANS
# ─────────────────────────────────────────────────────────────────────────────
def etape2_kmeans():
    separator("ÉTAPE 2 — PIPELINE K-MEANS")
    import kmeans_clustering as km_mod
    t0 = time.time()
    km, X_scaled, labels_km, df_out = km_mod.main()
    duree_km = time.time() - t0
    print(f"  ⏱  Durée K-means : {duree_km:.2f} s")
    return km, X_scaled, labels_km, df_out, duree_km


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 3 — PIPELINE HCLUST
# ─────────────────────────────────────────────────────────────────────────────
def etape3_hclust():
    separator("ÉTAPE 3 — PIPELINE CLUSTERING HIÉRARCHIQUE")
    import hclust_clustering as hc_mod
    t0 = time.time()
    modele_hc, X_scaled, labels_hc, metriques_hc = hc_mod.main()
    duree_hc = time.time() - t0
    print(f"  ⏱  Durée Hclust : {duree_hc:.2f} s")
    return modele_hc, X_scaled, labels_hc, metriques_hc, duree_hc


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 4 — VISUALISATION COMPARATIVE CÔTE-À-CÔTE
# ─────────────────────────────────────────────────────────────────────────────
def etape4_comparaison(X_scaled, labels_km, labels_hc,
                        metriques_km, metriques_hc,
                        duree_km, duree_hc):
    separator("ÉTAPE 4 — ANALYSE COMPARATIVE")

    import hclust_clustering as hc_mod
    hc_mod.comparer_methodes(metriques_km, metriques_hc)

    # ── PCA côte-à-côte ────────────────────────────────────────────────────
    pca  = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X_scaled)
    var  = pca.explained_variance_ratio_

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    titres = [f"K-Means (k=5)", f"Clustering Hiérarchique Ward (k=5)"]

    for ax, labels, titre in zip(axes, [labels_km, labels_hc], titres):
        for i, cl in enumerate(np.unique(labels)):
            mask = labels == cl
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                       c=PALETTE[i], alpha=0.5, s=14, label=f"Cluster {cl}")
        ax.set_title(titre, fontsize=12, fontweight="bold")
        ax.set_xlabel(f"PC1 ({var[0]*100:.1f} %)")
        ax.set_ylabel(f"PC2 ({var[1]*100:.1f} %)")
        ax.legend(loc="upper right", fontsize=8)

    fig.suptitle("Comparaison des partitions : K-Means vs Clustering Hiérarchique",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    chemin = os.path.join(FIG_DIR, "comparaison_kmeans_hclust.png")
    plt.savefig(chemin, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Figure comparative → {chemin}")

    # ── Accord entre les deux partitions ──────────────────────────────────
    ari_inter = adjusted_rand_score(labels_km, labels_hc)
    print(f"\n  Accord K-Means / Hclust (ARI inter-méthodes) : {ari_inter:.4f}")
    if ari_inter > 0.8:
        interpretation = "Très fort accord → structures similaires détectées"
    elif ari_inter > 0.6:
        interpretation = "Bon accord → partitions globalement cohérentes"
    elif ari_inter > 0.4:
        interpretation = "Accord modéré → quelques divergences"
    else:
        interpretation = "Faible accord → les méthodes divergent significativement"
    print(f"  Interprétation : {interpretation}")

    # ── Barplot comparatif des métriques ──────────────────────────────────
    metriques_comp = ["Silhouette Score", "Adjusted Rand Index (ARI)"]
    vals_km  = [metriques_km[m] for m in metriques_comp]
    vals_hc  = [metriques_hc[m] for m in metriques_comp]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, nom, vkm, vhc in zip(axes, metriques_comp, vals_km, vals_hc):
        bars = ax.bar(["K-Means", "Hclust"], [vkm, vhc],
                      color=[PALETTE[0], PALETTE[1]], alpha=0.85,
                      edgecolor="white", width=0.5)
        ax.set_title(nom, fontsize=11, fontweight="bold")
        ax.set_ylim(0, max(vkm, vhc) * 1.2)
        for bar, val in zip(bars, [vkm, vhc]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{val:.4f}", ha="center", fontsize=10)

    plt.suptitle("Comparaison des performances — K-Means vs Hclust",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    chemin2 = os.path.join(FIG_DIR, "comparaison_metriques.png")
    plt.savefig(chemin2, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Barplot métriques → {chemin2}")

    # ── Temps de calcul ────────────────────────────────────────────────────
    print(f"\n  ┌─────────────────────────────────┐")
    print(f"  │  Temps de calcul (n=2000)       │")
    print(f"  │  K-Means   : {duree_km:6.2f} s           │")
    print(f"  │  Hclust    : {duree_hc:6.2f} s           │")
    if duree_hc > 0:
        ratio = duree_hc / duree_km if duree_km > 0 else float("inf")
        print(f"  │  Ratio     : {ratio:6.1f}x (Hclust/KM) │")
    print(f"  └─────────────────────────────────┘")

    return ari_inter


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 5 — RAPPORT DE SYNTHÈSE TERMINAL
# ─────────────────────────────────────────────────────────────────────────────
def etape5_rapport_terminal(metriques_km, metriques_hc, ari_inter, duree_km, duree_hc):
    separator("ÉTAPE 5 — RAPPORT DE SYNTHÈSE")

    print("""
  ╔══════════════════════════════════════════════════════════╗
  ║     TP2 — Clustering K-Means / Hclust — Synthèse        ║
  ║     IFRI | Master 1 Génie Logiciel                       ║
  ╚══════════════════════════════════════════════════════════╝

  DATASET : MarketIFRI — 2 000 clients, 10 variables, 5 profils

  ┌──────────────────────┬──────────────┬──────────────┐
  │ Métrique             │   K-Means    │    Hclust    │
  ├──────────────────────┼──────────────┼──────────────┤""")

    for m in metriques_km:
        v_km = metriques_km[m]
        v_hc = metriques_hc[m]
        label = m[:20]
        print(f"  │ {label:<20} │ {v_km:>12.4f} │ {v_hc:>12.4f} │")

    print(f"""  └──────────────────────┴──────────────┴──────────────┘

  Accord inter-méthodes (ARI) : {ari_inter:.4f}
  Temps K-Means  : {duree_km:.2f} s
  Temps Hclust   : {duree_hc:.2f} s

  FIGURES GÉNÉRÉES :
    figures/kmeans_elbow_silhouette.png
    figures/kmeans_pca_clusters.png
    figures/kmeans_heatmap_profils.png
    figures/kmeans_boxplots.png
    figures/hclust_dendrogramme.png
    figures/hclust_sauts_fusion.png
    figures/hclust_pca_clusters.png
    figures/hclust_heatmap_profils.png
    figures/comparaison_kmeans_hclust.png
    figures/comparaison_metriques.png

  DONNÉES EXPORTÉES :
    data/resultats_kmeans.csv
    data/resultats_hclust.csv

  ✅ TP2 entièrement exécuté. Bonne rédaction du rapport !
    """)


# ─────────────────────────────────────────────────────────────────────────────
# POINT D'ENTRÉE
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "▓" * 60)
    print("  TP2 — Clustering K-Means / Hclust")
    print("  IFRI | Master 1 Génie Logiciel")
    print("▓" * 60)

    t_global = time.time()

    etape1_generer_dataset()

    km, X_scaled_km, labels_km, df_km, duree_km = etape2_kmeans()

    # Recalcul métriques K-means pour comparaison
    from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
    df_raw_km = pd.read_csv(DATA_PATH)
    labels_reels = df_raw_km["profil_reel"]
    metriques_km = {
        "Silhouette Score":          silhouette_score(X_scaled_km, labels_km),
        "Calinski-Harabász Index":   calinski_harabasz_score(X_scaled_km, labels_km),
        "Davies-Bouldin Index":      davies_bouldin_score(X_scaled_km, labels_km),
        "Adjusted Rand Index (ARI)": adjusted_rand_score(
            LabelEncoder().fit_transform(labels_reels), labels_km),
    }

    modele_hc, X_scaled_hc, labels_hc, metriques_hc, duree_hc = etape3_hclust()

    ari_inter = etape4_comparaison(
        X_scaled_km, labels_km, labels_hc,
        metriques_km, metriques_hc,
        duree_km, duree_hc,
    )

    etape5_rapport_terminal(metriques_km, metriques_hc, ari_inter, duree_km, duree_hc)

    print(f"  ⏱  Durée totale : {time.time() - t_global:.1f} s\n")
