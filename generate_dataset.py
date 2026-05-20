"""
=============================================================================
TP2 — Clustering K-means / Hclust
IFRI — Master 1 Génie Logiciel
Script : Génération de la dataset de segmentation clients
=============================================================================

Ce script génère une dataset synthétique mais réaliste de segmentation
clients pour une enseigne de grande distribution fictive ("MarketIFRI").

La dataset simule 2000 clients avec 10 variables pertinentes permettant
d'identifier des profils comportementaux distincts.

Variables générées :
    - client_id          : Identifiant unique
    - age                : Âge du client (18–75 ans)
    - revenu_annuel      : Revenu annuel estimé (€)
    - score_depenses     : Score de propension à la dépense (0–100)
    - nb_achats_annuel   : Nombre d'achats dans l'année
    - panier_moyen       : Valeur moyenne d'un panier (€)
    - anciennete_mois    : Ancienneté client (mois)
    - ratio_promo        : Part des achats en promotion (0–1)
    - score_fidelite     : Score de fidélité calculé (0–100)
    - categorie_preferee : Catégorie d'achat dominante (encodée)
    - canal_principal    : Canal d'achat principal (encodé)

Profils simulés (5 segments) :
    1. "Jeunes économes"        : jeunes, faible revenu, achats prudents
    2. "Familles actives"       : âge moyen, revenu moyen, gros paniers
    3. "Seniors fidèles"        : âge élevé, revenu moyen, très fidèles
    4. "Aisés premium"          : revenu élevé, gros paniers, peu de promos
    5. "Chasseurs de promo"     : tous âges, revenu faible/moyen, max promo
=============================================================================
"""

import numpy as np
import pandas as pd
import os

# ---------------------------------------------------------------------------
# Configuration de la seed pour reproductibilité
# ---------------------------------------------------------------------------
np.random.seed(42)

N_CLIENTS  = 2000
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "clients_marketifri.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Définition des 5 profils (proportions et paramètres statistiques)
# ---------------------------------------------------------------------------
profils = {
    "Jeunes_Economes": {
        "proportion": 0.20,
        "age_mu": 24, "age_sigma": 4,
        "revenu_mu": 18000, "revenu_sigma": 4000,
        "score_dep_mu": 35, "score_dep_sigma": 10,
        "nb_achats_mu": 20, "nb_achats_sigma": 5,
        "panier_mu": 22, "panier_sigma": 6,
        "anciennete_mu": 14, "anciennete_sigma": 8,
        "ratio_promo_mu": 0.45, "ratio_promo_sigma": 0.12,
        "score_fidelite_mu": 30, "score_fidelite_sigma": 10,
        "cat_pref": [0, 1, 2],      # Fast-food, Loisirs, Électronique
        "canal": [1, 2],            # En ligne, Application mobile
    },
    "Familles_Actives": {
        "proportion": 0.25,
        "age_mu": 38, "age_sigma": 6,
        "revenu_mu": 42000, "revenu_sigma": 8000,
        "score_dep_mu": 60, "score_dep_sigma": 12,
        "nb_achats_mu": 48, "nb_achats_sigma": 10,
        "panier_mu": 85, "panier_sigma": 20,
        "anciennete_mu": 36, "anciennete_sigma": 18,
        "ratio_promo_mu": 0.30, "ratio_promo_sigma": 0.10,
        "score_fidelite_mu": 62, "score_fidelite_sigma": 12,
        "cat_pref": [3, 4, 5],      # Alimentaire, Hygiène, Maison
        "canal": [0, 1],            # En magasin, En ligne
    },
    "Seniors_Fideles": {
        "proportion": 0.20,
        "age_mu": 62, "age_sigma": 7,
        "revenu_mu": 35000, "revenu_sigma": 7000,
        "score_dep_mu": 50, "score_dep_sigma": 10,
        "nb_achats_mu": 60, "nb_achats_sigma": 12,
        "panier_mu": 55, "panier_sigma": 15,
        "anciennete_mu": 72, "anciennete_sigma": 24,
        "ratio_promo_mu": 0.20, "ratio_promo_sigma": 0.08,
        "score_fidelite_mu": 85, "score_fidelite_sigma": 8,
        "cat_pref": [3, 5, 6],      # Alimentaire, Maison, Santé
        "canal": [0],               # En magasin uniquement
    },
    "Aisis_Premium": {
        "proportion": 0.15,
        "age_mu": 44, "age_sigma": 8,
        "revenu_mu": 85000, "revenu_sigma": 18000,
        "score_dep_mu": 80, "score_dep_sigma": 10,
        "nb_achats_mu": 35, "nb_achats_sigma": 8,
        "panier_mu": 200, "panier_sigma": 60,
        "anciennete_mu": 48, "anciennete_sigma": 20,
        "ratio_promo_mu": 0.08, "ratio_promo_sigma": 0.05,
        "score_fidelite_mu": 72, "score_fidelite_sigma": 14,
        "cat_pref": [7, 8, 2],      # Luxe, Gastronomie, Électronique
        "canal": [0, 1],            # En magasin, En ligne
    },
    "Chasseurs_Promo": {
        "proportion": 0.20,
        "age_mu": 35, "age_sigma": 12,
        "revenu_mu": 25000, "revenu_sigma": 7000,
        "score_dep_mu": 55, "score_dep_sigma": 15,
        "nb_achats_mu": 55, "nb_achats_sigma": 15,
        "panier_mu": 38, "panier_sigma": 12,
        "anciennete_mu": 28, "anciennete_sigma": 16,
        "ratio_promo_mu": 0.72, "ratio_promo_sigma": 0.12,
        "score_fidelite_mu": 45, "score_fidelite_sigma": 15,
        "cat_pref": [0, 3, 4],      # Fast-food, Alimentaire, Hygiène
        "canal": [1, 2],            # En ligne, Application mobile
    },
}

# Encodages
categories_label = {
    0: "Fast-food",
    1: "Loisirs",
    2: "Electronique",
    3: "Alimentaire",
    4: "Hygiene",
    5: "Maison",
    6: "Sante",
    7: "Luxe",
    8: "Gastronomie",
}

canaux_label = {
    0: "Magasin",
    1: "En_ligne",
    2: "Application",
}

# ---------------------------------------------------------------------------
# Génération des données
# ---------------------------------------------------------------------------
records = []
client_id = 1

for profil_name, p in profils.items():
    n = int(N_CLIENTS * p["proportion"])

    ages            = np.random.normal(p["age_mu"],            p["age_sigma"],            n).clip(18, 75).astype(int)
    revenus         = np.random.normal(p["revenu_mu"],         p["revenu_sigma"],         n).clip(10000, 150000).astype(int)
    scores_dep      = np.random.normal(p["score_dep_mu"],      p["score_dep_sigma"],      n).clip(0, 100).astype(int)
    nb_achats       = np.random.normal(p["nb_achats_mu"],      p["nb_achats_sigma"],      n).clip(1, 120).astype(int)
    paniers         = np.random.normal(p["panier_mu"],         p["panier_sigma"],         n).clip(5, 500).round(2)
    anciennetes     = np.random.normal(p["anciennete_mu"],     p["anciennete_sigma"],     n).clip(1, 120).astype(int)
    ratios_promo    = np.random.normal(p["ratio_promo_mu"],    p["ratio_promo_sigma"],    n).clip(0, 1).round(3)
    scores_fidelite = np.random.normal(p["score_fidelite_mu"],p["score_fidelite_sigma"], n).clip(0, 100).astype(int)
    cats            = np.random.choice(p["cat_pref"],  n)
    canaux          = np.random.choice(p["canal"],     n)

    for i in range(n):
        records.append({
            "client_id":          f"C{client_id:05d}",
            "age":                ages[i],
            "revenu_annuel":      revenus[i],
            "score_depenses":     scores_dep[i],
            "nb_achats_annuel":   nb_achats[i],
            "panier_moyen":       paniers[i],
            "anciennete_mois":    anciennetes[i],
            "ratio_promo":        ratios_promo[i],
            "score_fidelite":     scores_fidelite[i],
            "categorie_preferee": categories_label[cats[i]],
            "canal_principal":    canaux_label[canaux[i]],
            "profil_reel":        profil_name,   # Label de vérité terrain (pour évaluation)
        })
        client_id += 1

# ---------------------------------------------------------------------------
# Assemblage et ajout de bruit / valeurs manquantes réalistes (~2%)
# ---------------------------------------------------------------------------
df = pd.DataFrame(records)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Injection aléatoire de NaN sur colonnes numériques (2 % par colonne)
cols_numeriques = ["revenu_annuel", "panier_moyen", "anciennete_mois", "ratio_promo"]
for col in cols_numeriques:
    mask = np.random.rand(len(df)) < 0.02
    df.loc[mask, col] = np.nan

# ---------------------------------------------------------------------------
# Sauvegarde
# ---------------------------------------------------------------------------
df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

print(f"✅ Dataset générée avec succès !")
print(f"   Fichier : {OUTPUT_FILE}")
print(f"   Dimensions : {df.shape[0]} lignes × {df.shape[1]} colonnes")
print(f"   Taille : {os.path.getsize(OUTPUT_FILE) / 1024:.1f} Ko")
print(f"\n   Répartition des profils :")
print(df["profil_reel"].value_counts().to_string())
print(f"\n   Aperçu des 5 premières lignes :")
print(df.head().to_string())
print(f"\n   Statistiques descriptives :")
print(df.describe().to_string())
