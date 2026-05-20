"""
=============================================================================
TP2 — Clustering K-means / Hclust
IFRI — Master 1 Génie Logiciel
Script : Implémentation K-means (kmeans_clustering.py)
=============================================================================

Ce module implémente le clustering K-means sur la dataset de segmentation
clients "MarketIFRI". Il couvre l'intégralité du pipeline :
    1. Chargement et prétraitement des données
    2. Détermination du nombre optimal de clusters (Elbow + Silhouette)
    3. Entraînement du modèle K-means final
    4. Visualisations (PCA 2D, Elbow, Silhouette, Heatmap profils)
    5. Analyse et export des résultats
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
matplotlib.use("Agg")  # Backend non-interactif pour environnements sans display
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.preprocessing   import StandardScaler, LabelEncoder
from sklearn.cluster         import KMeans
from sklearn.decomposition   import PCA
from sklearn.metrics         import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_rand_score,
)

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 2. CONFIGURATION GLOBALE
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = os.path.join(BASE_DIR, "data",    "clients_marketifri.csv")
FIG_DIR     = os.path.join(BASE_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# Palette esthétique cohérente pour toutes les figures
PALETTE     = ["#E63946", "#457B9D", "#2A9D8F", "#E9C46A", "#F4A261"]
CMAP_HEAT   = "YlOrRd"
RANDOM_STATE = 42

plt.rcParams.update({
    "figure.dpi":      150,
    "font.family":     "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize":  13,
    "axes.labelsize":  11,
    "legend.fontsize": 9,
})


# ─────────────────────────────────────────────────────────────────────────────
# 3. CHARGEMENT ET PRÉTRAITEMENT
# ─────────────────────────────────────────────────────────────────────────────
def charger_et_pretraiter(chemin: str):
    """
    Charge la dataset CSV, applique le nettoyage et la standardisation.

    Étapes :
        - Chargement CSV
        - Imputation des valeurs manquantes (médiane pour les numériques)
        - Encodage ordinal des variables catégorielles
        - Standardisation (StandardScaler → μ=0, σ=1)
        - Conservation des labels réels pour évaluation externe

    Returns
    -------
    df_raw   : DataFrame original (non transformé)
    X_scaled : np.ndarray standardisé prêt pour le clustering
    features : liste des colonnes utilisées
    labels_reels : pd.Series des profils réels (pour ARI)
    """
    print("═" * 60)
    print("  CHARGEMENT ET PRÉTRAITEMENT")
    print("═" * 60)

    df = pd.read_csv(chemin)
    print(f"  ✓ Dataset chargée : {df.shape[0]} lignes × {df.shape[1]} colonnes")

    # ── Valeurs manquantes ──────────────────────────────────────────────────
    nb_manquants = df.isnull().sum()
    print(f"\n  Valeurs manquantes détectées :")
    print(nb_manquants[nb_manquants > 0].to_string())

    cols_num = ["revenu_annuel", "panier_moyen", "anciennete_mois", "ratio_promo"]
    for col in cols_num:
        mediane = df[col].median()
        df[col].fillna(mediane, inplace=True)
        print(f"    → Imputation médiane sur '{col}' : {mediane:.2f}")

    # ── Encodage des variables catégorielles ────────────────────────────────
    le_cat   = LabelEncoder()
    le_canal = LabelEncoder()
    df["cat_enc"]   = le_cat.fit_transform(df["categorie_preferee"])
    df["canal_enc"] = le_canal.fit_transform(df["canal_principal"])

    # ── Sélection des features pour le clustering ───────────────────────────
    features = [
        "age",
        "revenu_annuel",
        "score_depenses",
        "nb_achats_annuel",
        "panier_moyen",
        "anciennete_mois",
        "ratio_promo",
        "score_fidelite",
        "cat_enc",
        "canal_enc",
    ]

    # ── Imputation supplémentaire sur toutes les colonnes numériques ─────────
    for col in features:
        if df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)

    X = df[features].values

    # ── Standardisation Z-score ─────────────────────────────────────────────
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Sécurité : remplacement des NaN résiduels par 0 (ne devrait pas arriver)
    X_scaled = np.nan_to_num(X_scaled, nan=0.0)

    labels_reels = df["profil_reel"]

    print(f"\n  ✓ Features sélectionnées ({len(features)}) : {features}")
    print(f"  ✓ Standardisation appliquée — μ≈0, σ≈1")
    print(f"  ✓ Valeurs manquantes résiduelles : {np.isnan(X_scaled).sum()}")
    print()

    return df, X_scaled, features, labels_reels, scaler


# ─────────────────────────────────────────────────────────────────────────────
# 4. MÉTHODE ELBOW + SILHOUETTE — CHOIX DU K OPTIMAL
# ─────────────────────────────────────────────────────────────────────────────
def determiner_k_optimal(X_scaled: np.ndarray, k_max: int = 12):
    """
    Calcule l'inertie (Elbow) et le Silhouette Score pour k ∈ [2, k_max].
    Génère une figure double-panneau et retourne le k recommandé.

    Critère de choix automatique :
        k* = argmax(Silhouette Score) parmi les k ∈ [2, k_max]
    """
    print("═" * 60)
    print("  DÉTERMINATION DU K OPTIMAL")
    print("═" * 60)

    inerties   = []
    silhouettes = []
    k_range    = range(2, k_max + 1)

    for k in k_range:
        km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=RANDOM_STATE)
        km.fit(X_scaled)
        inerties.append(km.inertia_)
        sil = silhouette_score(X_scaled, km.labels_, sample_size=500, random_state=RANDOM_STATE)
        silhouettes.append(sil)
        print(f"  k={k:2d} | Inertie = {km.inertia_:10.1f} | Silhouette = {sil:.4f}")

    k_optimal = list(k_range)[np.argmax(silhouettes)]
    print(f"\n  ✓ K optimal recommandé : k = {k_optimal} (Silhouette max = {max(silhouettes):.4f})")

    # ── Figure Elbow + Silhouette ────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Détermination du nombre optimal de clusters K", fontsize=14, fontweight="bold", y=1.02)

    # Panneau gauche : Elbow
    ax1 = axes[0]
    ax1.plot(list(k_range), inerties, marker="o", color="#E63946", linewidth=2, markersize=7, markerfacecolor="white", markeredgewidth=2)
    ax1.axvline(k_optimal, color="#457B9D", linestyle="--", linewidth=1.5, label=f"k* = {k_optimal}")
    ax1.set_xlabel("Nombre de clusters k")
    ax1.set_ylabel("Inertie (Within-Cluster SSE)")
    ax1.set_title("Méthode du Coude (Elbow Method)")
    ax1.set_xticks(list(k_range))
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    # Panneau droit : Silhouette
    ax2 = axes[1]
    bars = ax2.bar(list(k_range), silhouettes, color=[PALETTE[i % len(PALETTE)] for i in range(len(k_range))], alpha=0.85, edgecolor="white")
    ax2.axvline(k_optimal, color="#2A9D8F", linestyle="--", linewidth=2, label=f"k* = {k_optimal}")
    ax2.set_xlabel("Nombre de clusters k")
    ax2.set_ylabel("Silhouette Score")
    ax2.set_title("Silhouette Score par nombre de clusters")
    ax2.set_xticks(list(k_range))
    ax2.set_ylim(0, max(silhouettes) * 1.15)
    ax2.legend()

    # Annotation max
    idx_max = np.argmax(silhouettes)
    ax2.annotate(f"  max = {max(silhouettes):.3f}",
                 xy=(list(k_range)[idx_max], max(silhouettes)),
                 xytext=(list(k_range)[idx_max] + 0.4, max(silhouettes) + 0.005),
                 fontsize=9, color="#2A9D8F")

    plt.tight_layout()
    chemin_fig = os.path.join(FIG_DIR, "kmeans_elbow_silhouette.png")
    plt.savefig(chemin_fig, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Figure sauvegardée → {chemin_fig}")
    print()

    return k_optimal, inerties, silhouettes


# ─────────────────────────────────────────────────────────────────────────────
# 5. ENTRAÎNEMENT DU MODÈLE K-MEANS FINAL
# ─────────────────────────────────────────────────────────────────────────────
def entrainer_kmeans(X_scaled: np.ndarray, k: int):
    """
    Entraîne le modèle K-means final avec k clusters.

    Paramètres :
        init     = "k-means++"  → initialisation intelligente (Arthur & Vassilvitskii, 2007)
        n_init   = 20           → 20 initialisations aléatoires, on garde la meilleure
        max_iter = 500          → jusqu'à 500 itérations par exécution
        tol      = 1e-4         → seuil de convergence sur le déplacement des centroides

    Returns : modèle KMeans entraîné
    """
    print("═" * 60)
    print(f"  ENTRAÎNEMENT K-MEANS (k = {k})")
    print("═" * 60)

    km = KMeans(
        n_clusters  = k,
        init        = "k-means++",
        n_init      = 20,
        max_iter    = 500,
        tol         = 1e-4,
        random_state = RANDOM_STATE,
    )
    km.fit(X_scaled)

    print(f"  ✓ Convergence en {km.n_iter_} itérations")
    print(f"  ✓ Inertie finale : {km.inertia_:.2f}")
    print(f"  ✓ Distribution des clusters :")
    uniques, counts = np.unique(km.labels_, return_counts=True)
    for cl, cnt in zip(uniques, counts):
        print(f"    Cluster {cl} : {cnt} clients ({100*cnt/len(km.labels_):.1f} %)")
    print()

    return km


# ─────────────────────────────────────────────────────────────────────────────
# 6. ÉVALUATION DES MÉTRIQUES
# ─────────────────────────────────────────────────────────────────────────────
def evaluer_modele(X_scaled: np.ndarray, labels_pred: np.ndarray, labels_reels: pd.Series):
    """
    Calcule et affiche les métriques internes et externes du clustering.

    Métriques internes (ne nécessitent pas les labels réels) :
        - Silhouette Score          : cohésion et séparation ([-1, 1], ↑ mieux)
        - Calinski-Harabász Index   : rapport variance inter/intra (↑ mieux)
        - Davies-Bouldin Index      : similarité inter-clusters (↓ mieux)

    Métrique externe (nécessite les labels réels) :
        - Adjusted Rand Index (ARI) : accord avec la vérité terrain ([-1,1], 1=parfait)
    """
    print("═" * 60)
    print("  MÉTRIQUES D'ÉVALUATION")
    print("═" * 60)

    sil  = silhouette_score(X_scaled, labels_pred)
    ch   = calinski_harabasz_score(X_scaled, labels_pred)
    db   = davies_bouldin_score(X_scaled, labels_pred)

    le   = LabelEncoder()
    y    = le.fit_transform(labels_reels)
    ari  = adjusted_rand_score(y, labels_pred)

    metriques = {
        "Silhouette Score":         sil,
        "Calinski-Harabász Index":  ch,
        "Davies-Bouldin Index":     db,
        "Adjusted Rand Index (ARI)": ari,
    }

    for nom, val in metriques.items():
        print(f"  {nom:35s} : {val:.4f}")
    print()

    return metriques


# ─────────────────────────────────────────────────────────────────────────────
# 7. VISUALISATION PCA 2D
# ─────────────────────────────────────────────────────────────────────────────
def visualiser_pca(X_scaled: np.ndarray, labels: np.ndarray, centroides_pca: np.ndarray,
                   titre: str, nom_fichier: str):
    """
    Projette les données en 2D via PCA et trace le nuage de points coloré par cluster.
    Les centroides sont superposés avec un marqueur distinctif.
    """
    pca  = PCA(n_components=2, random_state=RANDOM_STATE)
    X_2d = pca.fit_transform(X_scaled)
    var_exp = pca.explained_variance_ratio_

    # Transformation des centroides dans l'espace PCA
    pca_centroides = pca.transform(centroides_pca)

    fig, ax = plt.subplots(figsize=(10, 7))

    n_clusters = len(np.unique(labels))
    couleurs   = PALETTE[:n_clusters]

    for i, cl in enumerate(np.unique(labels)):
        mask = labels == cl
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   c=couleurs[i], alpha=0.55, s=18, label=f"Cluster {cl}",
                   edgecolors="none")

    # Centroides
    for i, (cx, cy) in enumerate(pca_centroides):
        ax.scatter(cx, cy, c=couleurs[i], s=280, marker="*",
                   edgecolors="black", linewidths=1.2, zorder=5)
        ax.annotate(f"C{i}", (cx, cy), textcoords="offset points",
                    xytext=(8, 4), fontsize=9, fontweight="bold")

    ax.set_xlabel(f"Composante principale 1 ({var_exp[0]*100:.1f} % variance)")
    ax.set_ylabel(f"Composante principale 2 ({var_exp[1]*100:.1f} % variance)")
    ax.set_title(titre, fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="upper right", framealpha=0.9)

    # Annotation variance totale
    ax.text(0.02, 0.97, f"Variance expliquée totale : {sum(var_exp)*100:.1f} %",
            transform=ax.transAxes, fontsize=8.5, color="gray", va="top")

    plt.tight_layout()
    chemin = os.path.join(FIG_DIR, nom_fichier)
    plt.savefig(chemin, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Figure PCA sauvegardée → {chemin}")

    return pca, X_2d


# ─────────────────────────────────────────────────────────────────────────────
# 8. HEATMAP DES PROFILS DE CLUSTERS
# ─────────────────────────────────────────────────────────────────────────────
def heatmap_profils(df_raw: pd.DataFrame, labels: np.ndarray, features: list):
    """
    Calcule les moyennes de chaque feature par cluster (sur données non-standardisées)
    et les représente sous forme d'une heatmap normalisée pour faciliter la comparaison.
    """
    df_clusters = df_raw.copy()
    df_clusters["cluster"] = labels

    # Sélection des features numériques lisibles (exclure les encodées)
    feats_viz = [f for f in features if f not in ("cat_enc", "canal_enc")]
    labels_viz = {
        "age":              "Âge",
        "revenu_annuel":    "Revenu\nannuel (€)",
        "score_depenses":   "Score\ndépenses",
        "nb_achats_annuel": "Nbre d'achats\n/an",
        "panier_moyen":     "Panier\nmoyen (€)",
        "anciennete_mois":  "Ancienneté\n(mois)",
        "ratio_promo":      "Ratio\npromotions",
        "score_fidelite":   "Score\nfidélité",
    }

    pivot = df_clusters.groupby("cluster")[feats_viz].mean()

    # Normalisation min-max par colonne pour la heatmap
    pivot_norm = (pivot - pivot.min()) / (pivot.max() - pivot.min())
    pivot_norm.columns = [labels_viz.get(c, c) for c in pivot_norm.columns]

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.heatmap(pivot_norm.T, annot=pivot.T.round(1).values,
                fmt="g", cmap=CMAP_HEAT,
                linewidths=0.5, linecolor="white",
                ax=ax, cbar_kws={"label": "Valeur normalisée (0–1)"})

    ax.set_xlabel("Cluster", fontsize=11)
    ax.set_ylabel("")
    ax.set_title("Heatmap des profils moyens par cluster (K-means)", fontsize=13, fontweight="bold")
    ax.set_xticklabels([f"Cluster {i}" for i in pivot.index], rotation=0)

    plt.tight_layout()
    chemin = os.path.join(FIG_DIR, "kmeans_heatmap_profils.png")
    plt.savefig(chemin, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Heatmap sauvegardée → {chemin}")


# ─────────────────────────────────────────────────────────────────────────────
# 9. BOXPLOTS COMPARATIFS
# ─────────────────────────────────────────────────────────────────────────────
def boxplots_clusters(df_raw: pd.DataFrame, labels: np.ndarray):
    """
    Génère 4 boxplots côte-à-côte pour les variables les plus discriminantes.
    """
    df_cl = df_raw.copy()
    df_cl["Cluster"] = [f"Cluster {l}" for l in labels]

    variables_cles = [
        ("revenu_annuel",    "Revenu annuel (€)"),
        ("panier_moyen",     "Panier moyen (€)"),
        ("score_fidelite",   "Score de fidélité"),
        ("ratio_promo",      "Ratio promotions"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.flatten()

    palette_cluster = {f"Cluster {i}": PALETTE[i] for i in range(5)}

    for idx, (col, titre) in enumerate(variables_cles):
        sns.boxplot(data=df_cl, x="Cluster", y=col,
                    palette=palette_cluster, ax=axes[idx],
                    linewidth=1.2, fliersize=3, width=0.55)
        axes[idx].set_title(titre, fontsize=11, fontweight="bold")
        axes[idx].set_xlabel("")
        axes[idx].set_ylabel("")
        axes[idx].tick_params(axis="x", rotation=15)

    fig.suptitle("Distribution des variables clés par cluster (K-means)",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    chemin = os.path.join(FIG_DIR, "kmeans_boxplots.png")
    plt.savefig(chemin, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Boxplots sauvegardés → {chemin}")


# ─────────────────────────────────────────────────────────────────────────────
# 10. EXPORT DES RÉSULTATS
# ─────────────────────────────────────────────────────────────────────────────
def exporter_resultats(df_raw: pd.DataFrame, labels: np.ndarray):
    """Ajoute les labels de cluster au DataFrame et exporte en CSV."""
    df_out = df_raw.copy()
    df_out["cluster_kmeans"] = labels

    chemin = os.path.join(BASE_DIR, "data", "resultats_kmeans.csv")
    df_out.to_csv(chemin, index=False, encoding="utf-8-sig")
    print(f"  ✓ Résultats exportés → {chemin}")
    return df_out


# ─────────────────────────────────────────────────────────────────────────────
# 11. PIPELINE PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "█" * 60)
    print("  TP2 — CLUSTERING K-MEANS | IFRI — Master 1 GL")
    print("█" * 60 + "\n")

    # ── Étape 1 : Chargement et prétraitement ───────────────────────────────
    df_raw, X_scaled, features, labels_reels, scaler = charger_et_pretraiter(DATA_PATH)

    # ── Étape 2 : Choix du k optimal ────────────────────────────────────────
    k_optimal, inerties, silhouettes = determiner_k_optimal(X_scaled, k_max=10)

    # Pour ce TP on fixe k=5 (en accord avec les 5 profils métier connus)
    K = 5
    print(f"  ⚙  Choix final : K = {K} (cohérent avec les profils métier)\n")

    # ── Étape 3 : Entraînement K-means ──────────────────────────────────────
    km = entrainer_kmeans(X_scaled, K)
    labels_pred = km.labels_

    # ── Étape 4 : Évaluation ────────────────────────────────────────────────
    print("═" * 60)
    print("  ÉVALUATION K-MEANS")
    print("═" * 60)
    metriques = evaluer_modele(X_scaled, labels_pred, labels_reels)

    # ── Étape 5 : Visualisations ─────────────────────────────────────────────
    print("═" * 60)
    print("  VISUALISATIONS")
    print("═" * 60)

    pca, X_2d = visualiser_pca(
        X_scaled, labels_pred, km.cluster_centers_,
        titre      = "Projection PCA des clusters K-means (k=5)",
        nom_fichier= "kmeans_pca_clusters.png",
    )

    heatmap_profils(df_raw, labels_pred, features)
    boxplots_clusters(df_raw, labels_pred)

    # ── Étape 6 : Export ─────────────────────────────────────────────────────
    df_out = exporter_resultats(df_raw, labels_pred)

    # ── Étape 7 : Résumé des profils ─────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  RÉSUMÉ DES PROFILS DE CLUSTERS")
    print("═" * 60)
    feats_resume = ["age", "revenu_annuel", "score_depenses",
                    "panier_moyen", "score_fidelite", "ratio_promo"]
    resume = df_out.groupby("cluster_kmeans")[feats_resume].mean().round(1)
    resume.index = [f"Cluster {i}" for i in resume.index]
    print(resume.to_string())

    print("\n  ✅ Pipeline K-means terminé avec succès.")
    print("     Figures générées dans :", FIG_DIR)
    print()

    return km, X_scaled, labels_pred, df_out


if __name__ == "__main__":
    main()
