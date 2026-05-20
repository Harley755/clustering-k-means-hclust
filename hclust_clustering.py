"""
=============================================================================
TP2 — Clustering K-means / Hclust
IFRI — Master 1 Génie Logiciel
Script : Implémentation Hierarchical Clustering (hclust_clustering.py)
=============================================================================

Ce module implémente le Clustering Hiérarchique Agglomératif (CAH) sur la
même dataset de segmentation clients "MarketIFRI".

Pipeline :
    1. Chargement et prétraitement identique au module K-means
    2. Construction de la matrice de liens (linkage matrix)
    3. Tracé du dendrogramme tronqué
    4. Détermination du nombre de clusters par coupure du dendrogramme
    5. Attribution des labels via scipy.cluster.hierarchy.fcluster
    6. Visualisations avancées (PCA, heatmap, dendrogramme coloré)
    7. Évaluation avec les mêmes métriques que K-means
    8. Comparaison formelle K-means vs Hclust

Méthode de liaison (linkage) choisie :
    "ward" — minimise la variance intra-cluster lors de chaque fusion.
    Produit des clusters bien équilibrés et compacts.
    C'est la méthode recommandée en clustering exploratoire (Murtagh & Legendre, 2014).
=============================================================================
"""

# ─────────────────────────────────────────────────────────────────────────────
# 1. IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import os
import warnings
import numpy  as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing   import StandardScaler, LabelEncoder
from sklearn.decomposition   import PCA
from sklearn.metrics         import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_rand_score,
)
from sklearn.cluster         import AgglomerativeClustering

from scipy.cluster.hierarchy import (
    linkage,
    dendrogram,
    fcluster,
    cophenet,
)
from scipy.spatial.distance  import pdist

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 2. CONFIGURATION GLOBALE
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = os.path.join(BASE_DIR, "data",    "clients_marketifri.csv")
FIG_DIR     = os.path.join(BASE_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

PALETTE      = ["#E63946", "#457B9D", "#2A9D8F", "#E9C46A", "#F4A261"]
CMAP_HEAT    = "Blues"
RANDOM_STATE = 42
LINKAGE_METHOD = "ward"   # Méthode de liaison Ward — justifiée dans le rapport

plt.rcParams.update({
    "figure.dpi":        150,
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.titlesize":    13,
    "axes.labelsize":    11,
})


# ─────────────────────────────────────────────────────────────────────────────
# 3. CHARGEMENT ET PRÉTRAITEMENT (identique au module K-means)
# ─────────────────────────────────────────────────────────────────────────────
def charger_et_pretraiter(chemin: str):
    """
    Chargement, nettoyage, encodage et standardisation de la dataset.
    Voir kmeans_clustering.py pour les détails complets.
    """
    print("═" * 60)
    print("  CHARGEMENT ET PRÉTRAITEMENT")
    print("═" * 60)

    df = pd.read_csv(chemin)
    print(f"  ✓ Dataset chargée : {df.shape[0]} × {df.shape[1]}")

    # Imputation médiane
    for col in ["revenu_annuel", "panier_moyen", "anciennete_mois", "ratio_promo"]:
        df[col].fillna(df[col].median(), inplace=True)

    # Encodage catégoriel
    df["cat_enc"]   = LabelEncoder().fit_transform(df["categorie_preferee"])
    df["canal_enc"] = LabelEncoder().fit_transform(df["canal_principal"])

    features = [
        "age", "revenu_annuel", "score_depenses", "nb_achats_annuel",
        "panier_moyen", "anciennete_mois", "ratio_promo",
        "score_fidelite", "cat_enc", "canal_enc",
    ]

    # Imputation complète avant scaling
    for col in features:
        if hasattr(df[col], 'isnull') and df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)

    X        = df[features].values
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = np.nan_to_num(X_scaled, nan=0.0)  # sécurité

    print(f"  ✓ {len(features)} features — Standardisation Z-score appliquée\n")
    return df, X_scaled, features, df["profil_reel"]


# ─────────────────────────────────────────────────────────────────────────────
# 4. SOUS-ÉCHANTILLONNAGE POUR LA CAH
# ─────────────────────────────────────────────────────────────────────────────
def sous_echantillonner(X_scaled: np.ndarray, n_max: int = 500, seed: int = 42):
    """
    La CAH a une complexité O(n² log n) en mémoire et O(n³) naïvement.
    Pour un dendrogramme lisible et un calcul rapide, on sous-échantillonne.
    Sur l'échantillon réduit : calcul du dendrogramme + coefficients de liaison.
    Le modèle AgglomerativeClustering final est ensuite ajusté sur la totalité.

    Returns : indices de l'échantillon, X_sample
    """
    rng     = np.random.RandomState(seed)
    indices = rng.choice(len(X_scaled), size=min(n_max, len(X_scaled)), replace=False)
    indices = np.sort(indices)
    return indices, X_scaled[indices]


# ─────────────────────────────────────────────────────────────────────────────
# 5. CONSTRUCTION DE LA MATRICE DE LIENS ET COEFFICIENT COPHÉNÉTIQUE
# ─────────────────────────────────────────────────────────────────────────────
def construire_linkage(X_sample: np.ndarray, methode: str = LINKAGE_METHOD):
    """
    Calcule la matrice de liens Z (dendrogramme) et le coefficient cophénétique.

    Coefficient cophénétique c ∈ [0, 1] :
        - c > 0.75 : bonne représentation de la structure
        - c > 0.90 : excellente
    """
    print("═" * 60)
    print(f"  CONSTRUCTION MATRICE DE LIENS (méthode : {methode.upper()})")
    print("═" * 60)

    Z = linkage(X_sample, method=methode, metric="euclidean")

    # Coefficient cophénétique
    c, coph_dists = cophenet(Z, pdist(X_sample))
    print(f"  ✓ Coefficient cophénétique : {c:.4f}")
    if c > 0.90:
        qualite = "Excellente"
    elif c > 0.75:
        qualite = "Bonne"
    elif c > 0.60:
        qualite = "Acceptable"
    else:
        qualite = "Faible"
    print(f"    → Qualité de représentation : {qualite}\n")

    return Z, c


# ─────────────────────────────────────────────────────────────────────────────
# 6. TRACÉ DU DENDROGRAMME
# ─────────────────────────────────────────────────────────────────────────────
def tracer_dendrogramme(Z: np.ndarray, n_clusters: int, titre: str = "Dendrogramme"):
    """
    Trace un dendrogramme tronqué (les p dernières fusions) avec mise en
    évidence de la coupure au niveau n_clusters.
    """
    # Couleurs des branches selon les clusters finaux
    couleurs_clusters = PALETTE[:n_clusters]

    fig, ax = plt.subplots(figsize=(14, 6))

    # Dendrogramme tronqué (30 dernières fusions pour lisibilité)
    dend = dendrogram(
        Z,
        truncate_mode    = "lastp",
        p                = 30,
        leaf_rotation    = 90,
        leaf_font_size   = 8,
        show_contracted  = True,
        ax               = ax,
        color_threshold  = 0,            # Couleur unique d'abord
        above_threshold_color = "#AAAAAA",
    )

    # Détermination de la hauteur de coupure pour n_clusters clusters
    hauteurs  = Z[:, 2]
    n         = len(Z)
    # La coupure se fait entre la (n-n_clusters)-ième et (n-n_clusters+1)-ième fusion
    if n_clusters < n:
        seuil = (hauteurs[n - n_clusters] + hauteurs[n - n_clusters + 1]) / 2
    else:
        seuil = hauteurs[-1] / 2

    ax.axhline(y=seuil, color="#E63946", linestyle="--", linewidth=2,
               label=f"Coupure → {n_clusters} clusters (h ≈ {seuil:.2f})")

    ax.set_title(titre, fontsize=13, fontweight="bold")
    ax.set_xlabel("Nœuds / Observations regroupées")
    ax.set_ylabel("Distance de Ward")
    ax.legend(fontsize=9)

    plt.tight_layout()
    chemin = os.path.join(FIG_DIR, "hclust_dendrogramme.png")
    plt.savefig(chemin, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Dendrogramme sauvegardé → {chemin}")

    return seuil


# ─────────────────────────────────────────────────────────────────────────────
# 7. SÉLECTION DU NOMBRE DE CLUSTERS — ANALYSE DES SAUTS
# ─────────────────────────────────────────────────────────────────────────────
def analyser_sauts_fusion(Z: np.ndarray, k_max: int = 12):
    """
    Analyse la courbe des hauteurs de fusion pour identifier les sauts majeurs.
    Un grand saut indique qu'on fusionne des groupes très distants → bonne coupure.

    Méthode des accélérations (Tibshirani & Walther, 2001) :
        acceleration = diff(diff(hauteurs))
        k* = argmax(acceleration) + 2
    """
    print("═" * 60)
    print("  ANALYSE DES SAUTS DE FUSION")
    print("═" * 60)

    last       = Z[-k_max:, 2]            # k_max dernières hauteurs
    acceleration = np.diff(np.diff(last)) # Accélération (2ème différence)

    k_optimal  = acceleration[::-1].argmax() + 2
    print(f"  ✓ k optimal par analyse des accélérations : k = {k_optimal}")

    # Figure : courbe des hauteurs de fusion
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    idx_range = range(len(Z) - k_max, len(Z))
    ks        = range(k_max, 0, -1)

    ax1 = axes[0]
    ax1.plot(list(ks), last[::-1], marker="o", color="#457B9D",
             linewidth=2, markersize=7, markerfacecolor="white", markeredgewidth=2)
    ax1.axvline(k_optimal, color="#E63946", linestyle="--", label=f"k* = {k_optimal}")
    ax1.set_xlabel("Nombre de clusters k")
    ax1.set_ylabel("Hauteur de fusion (distance de Ward)")
    ax1.set_title("Hauteurs des k dernières fusions")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    ax2 = axes[1]
    ax2.plot(range(2, len(acceleration) + 2), acceleration[::-1],
             marker="s", color="#2A9D8F", linewidth=2, markersize=6)
    ax2.axvline(k_optimal, color="#E63946", linestyle="--", label=f"k* = {k_optimal}")
    ax2.set_xlabel("Nombre de clusters k")
    ax2.set_ylabel("Accélération (Δ² hauteur)")
    ax2.set_title("Méthode de l'accélération")
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)

    plt.suptitle("Détermination du k optimal — Clustering Hiérarchique",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    chemin = os.path.join(FIG_DIR, "hclust_sauts_fusion.png")
    plt.savefig(chemin, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Figure sauts de fusion → {chemin}\n")

    return k_optimal


# ─────────────────────────────────────────────────────────────────────────────
# 8. CLUSTERING HIÉRARCHIQUE SUR L'ENSEMBLE COMPLET
# ─────────────────────────────────────────────────────────────────────────────
def appliquer_clustering_hclust(X_scaled: np.ndarray, n_clusters: int):
    """
    Applique AgglomerativeClustering de sklearn sur la dataset complète.
    sklearn optimise l'agglomération et gère de grands datasets plus efficacement
    que scipy.cluster.hierarchy.fcluster sur des données volumineuses.

    Paramètres :
        linkage  = "ward"      → minimise la variance intra-cluster
        metric   = "euclidean" → distance euclidienne
        n_clusters = k         → nombre de groupes finals
    """
    print("═" * 60)
    print(f"  CLUSTERING HIÉRARCHIQUE (k = {n_clusters}, linkage = ward)")
    print("═" * 60)

    modele = AgglomerativeClustering(
        n_clusters = n_clusters,
        linkage    = LINKAGE_METHOD,
        metric     = "euclidean",
    )
    labels = modele.fit_predict(X_scaled)

    print(f"  ✓ Clustering terminé sur {len(labels)} observations")
    uniques, counts = np.unique(labels, return_counts=True)
    for cl, cnt in zip(uniques, counts):
        print(f"    Cluster {cl} : {cnt} clients ({100*cnt/len(labels):.1f} %)")
    print()

    return modele, labels


# ─────────────────────────────────────────────────────────────────────────────
# 9. ÉVALUATION DU MODÈLE
# ─────────────────────────────────────────────────────────────────────────────
def evaluer_hclust(X_scaled: np.ndarray, labels: np.ndarray, labels_reels: pd.Series):
    """Calcule les métriques internes et l'ARI pour le clustering hiérarchique."""
    print("═" * 60)
    print("  MÉTRIQUES D'ÉVALUATION — HCLUST")
    print("═" * 60)

    sil = silhouette_score(X_scaled, labels)
    ch  = calinski_harabasz_score(X_scaled, labels)
    db  = davies_bouldin_score(X_scaled, labels)
    ari = adjusted_rand_score(LabelEncoder().fit_transform(labels_reels), labels)

    metriques = {
        "Silhouette Score":          sil,
        "Calinski-Harabász Index":   ch,
        "Davies-Bouldin Index":      db,
        "Adjusted Rand Index (ARI)": ari,
    }
    for nom, val in metriques.items():
        print(f"  {nom:35s} : {val:.4f}")
    print()
    return metriques


# ─────────────────────────────────────────────────────────────────────────────
# 10. VISUALISATION PCA (réutilisable)
# ─────────────────────────────────────────────────────────────────────────────
def visualiser_pca_hclust(X_scaled: np.ndarray, labels: np.ndarray):
    """Projection PCA 2D des clusters hiérarchiques."""
    pca  = PCA(n_components=2, random_state=RANDOM_STATE)
    X_2d = pca.fit_transform(X_scaled)
    var_exp = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(10, 7))

    n_cl = len(np.unique(labels))
    for i, cl in enumerate(np.unique(labels)):
        mask = labels == cl
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   c=PALETTE[i % len(PALETTE)], alpha=0.55, s=18,
                   label=f"Cluster {cl}", edgecolors="none")

    ax.set_xlabel(f"PC1 ({var_exp[0]*100:.1f} %)")
    ax.set_ylabel(f"PC2 ({var_exp[1]*100:.1f} %)")
    ax.set_title("Projection PCA — Clustering Hiérarchique Agglomératif (k=5)",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="upper right")
    ax.text(0.02, 0.97,
            f"Variance expliquée totale : {sum(var_exp)*100:.1f} %",
            transform=ax.transAxes, fontsize=8.5, color="gray", va="top")

    plt.tight_layout()
    chemin = os.path.join(FIG_DIR, "hclust_pca_clusters.png")
    plt.savefig(chemin, bbox_inches="tight")
    plt.close()
    print(f"  ✓ PCA Hclust sauvegardée → {chemin}")


# ─────────────────────────────────────────────────────────────────────────────
# 11. HEATMAP DES PROFILS
# ─────────────────────────────────────────────────────────────────────────────
def heatmap_profils_hclust(df_raw: pd.DataFrame, labels: np.ndarray, features: list):
    """Heatmap des moyennes par cluster pour le clustering hiérarchique."""
    df_cl = df_raw.copy()
    df_cl["cluster"] = labels

    feats_viz = [f for f in features if f not in ("cat_enc", "canal_enc")]
    labels_viz = {
        "age": "Âge", "revenu_annuel": "Revenu\n(€)",
        "score_depenses": "Score\ndépenses", "nb_achats_annuel": "Achats/an",
        "panier_moyen": "Panier\nmoyen (€)", "anciennete_mois": "Ancienneté\n(mois)",
        "ratio_promo": "Ratio\npromo", "score_fidelite": "Fidélité",
    }

    pivot      = df_cl.groupby("cluster")[feats_viz].mean()
    pivot_norm = (pivot - pivot.min()) / (pivot.max() - pivot.min())
    pivot_norm.columns = [labels_viz.get(c, c) for c in pivot_norm.columns]

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.heatmap(pivot_norm.T, annot=pivot.T.round(1).values,
                fmt="g", cmap=CMAP_HEAT,
                linewidths=0.5, linecolor="white",
                ax=ax, cbar_kws={"label": "Valeur normalisée (0–1)"})
    ax.set_xticklabels([f"Cluster {i}" for i in pivot.index], rotation=0)
    ax.set_title("Heatmap des profils moyens — Clustering Hiérarchique",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    chemin = os.path.join(FIG_DIR, "hclust_heatmap_profils.png")
    plt.savefig(chemin, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Heatmap Hclust → {chemin}")


# ─────────────────────────────────────────────────────────────────────────────
# 12. TABLEAU COMPARATIF K-MEANS vs HCLUST
# ─────────────────────────────────────────────────────────────────────────────
def comparer_methodes(metriques_km: dict, metriques_hclust: dict):
    """Affiche un tableau de comparaison formaté entre K-means et Hclust."""
    print("\n" + "═" * 70)
    print("  TABLEAU COMPARATIF — K-MEANS vs CLUSTERING HIÉRARCHIQUE")
    print("═" * 70)
    print(f"  {'Métrique':<40} {'K-Means':>12} {'Hclust':>12}")
    print("  " + "─" * 66)

    for cle in metriques_km:
        v_km  = metriques_km.get(cle,      float("nan"))
        v_hcl = metriques_hclust.get(cle,  float("nan"))
        gagnant = " ◀" if (
            (cle == "Davies-Bouldin Index" and v_km < v_hcl) or
            (cle != "Davies-Bouldin Index" and v_km > v_hcl)
        ) else ""
        gagnant_h = " ◀" if (
            (cle == "Davies-Bouldin Index" and v_hcl < v_km) or
            (cle != "Davies-Bouldin Index" and v_hcl > v_km)
        ) else ""
        print(f"  {cle:<40} {v_km:>12.4f}{gagnant:<3} {v_hcl:>12.4f}{gagnant_h}")

    print("  " + "─" * 66)
    print("  ◀ indique la meilleure valeur pour la métrique donnée\n")


# ─────────────────────────────────────────────────────────────────────────────
# 13. PIPELINE PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "█" * 60)
    print("  TP2 — CLUSTERING HIÉRARCHIQUE | IFRI — Master 1 GL")
    print("█" * 60 + "\n")

    # ── Chargement ───────────────────────────────────────────────────────────
    df_raw, X_scaled, features, labels_reels = charger_et_pretraiter(DATA_PATH)

    # ── Sous-échantillonnage pour le dendrogramme ─────────────────────────
    idx_sample, X_sample = sous_echantillonner(X_scaled, n_max=500)
    print(f"  ✓ Sous-échantillon pour dendrogramme : {len(X_sample)} observations\n")

    # ── Matrice de liens et coefficient cophénétique ──────────────────────
    Z, coeff_coph = construire_linkage(X_sample, methode=LINKAGE_METHOD)

    # ── Analyse des sauts + choix du k ────────────────────────────────────
    k_auto = analyser_sauts_fusion(Z, k_max=12)
    K      = 5   # Cohérence avec la connaissance métier et K-means
    print(f"  ⚙  Choix final : K = {K}\n")

    # ── Dendrogramme ──────────────────────────────────────────────────────
    print("═" * 60)
    print("  DENDROGRAMME")
    print("═" * 60)
    seuil = tracer_dendrogramme(Z, n_clusters=K,
                                 titre=f"Dendrogramme tronqué — Ward | k={K}")
    print()

    # ── Clustering sur dataset complète ───────────────────────────────────
    modele_hclust, labels_hclust = appliquer_clustering_hclust(X_scaled, n_clusters=K)

    # ── Évaluation ────────────────────────────────────────────────────────
    metriques_hclust = evaluer_hclust(X_scaled, labels_hclust, labels_reels)

    # ── Visualisations ────────────────────────────────────────────────────
    print("═" * 60)
    print("  VISUALISATIONS HCLUST")
    print("═" * 60)
    visualiser_pca_hclust(X_scaled, labels_hclust)
    heatmap_profils_hclust(df_raw, labels_hclust, features)

    # ── Export ────────────────────────────────────────────────────────────
    df_out = df_raw.copy()
    df_out["cluster_hclust"] = labels_hclust
    sortie = os.path.join(BASE_DIR, "data", "resultats_hclust.csv")
    df_out.to_csv(sortie, index=False, encoding="utf-8-sig")
    print(f"  ✓ Résultats Hclust exportés → {sortie}")

    # ── Résumé profils ────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  RÉSUMÉ DES PROFILS — HCLUST")
    print("═" * 60)
    feats_r = ["age", "revenu_annuel", "score_depenses",
               "panier_moyen", "score_fidelite", "ratio_promo"]
    resume = df_out.groupby("cluster_hclust")[feats_r].mean().round(1)
    resume.index = [f"Cluster {i}" for i in resume.index]
    print(resume.to_string())

    print("\n  ✅ Pipeline Hclust terminé avec succès.")
    return modele_hclust, X_scaled, labels_hclust, metriques_hclust


if __name__ == "__main__":
    main()
