# TP2 : Clustering K-means / Hclust — Implémentation

---

**Établissement :** Institut de Formation et de Recherche en Informatique (IFRI)  
**Formation :** Master 1 Génie Logiciel  
**Module :** Big Data & Machine Learning  
**Travail Pratique :** TP2 — Clustering non supervisé  
**Année académique :** 2024–2025  
**Étudiant :** [Nom Prénom]  
**Enseignant :** [Nom de l'enseignant]  
**Date de remise :** [Date]

---

## Table des matières

1. [Introduction](#1-introduction)
2. [Présentation de la dataset](#2-présentation-de-la-dataset)
3. [Prétraitement des données](#3-prétraitement-des-données)
4. [Fondements théoriques du K-means](#4-fondements-théoriques-du-k-means)
5. [Implémentation et résultats K-means](#5-implémentation-et-résultats-k-means)
6. [Fondements théoriques du Clustering Hiérarchique](#6-fondements-théoriques-du-clustering-hiérarchique)
7. [Implémentation et résultats du Clustering Hiérarchique](#7-implémentation-et-résultats-du-clustering-hiérarchique)
8. [Visualisations et analyse graphique](#8-visualisations-et-analyse-graphique)
9. [Analyse comparative K-means vs Hclust](#9-analyse-comparative-k-means-vs-hclust)
10. [Interprétation métier des clusters](#10-interprétation-métier-des-clusters)
11. [Conclusion](#11-conclusion)
12. [Références bibliographiques](#12-références-bibliographiques)
13. [Annexes](#13-annexes)

---

## 1. Introduction

### 1.1 Contexte général

L'ère du Big Data a profondément transformé la capacité des organisations à exploiter leurs données. Selon le cabinet IDC, le volume mondial de données générées devrait atteindre 175 zettaoctets d'ici 2025, créant un besoin croissant d'outils analytiques capables d'en extraire de la valeur. Dans ce contexte, l'apprentissage automatique non supervisé — et plus particulièrement le clustering — occupe une place centrale dans les pipelines modernes de Data Science.

Le **clustering**, ou partitionnement de données, désigne l'ensemble des méthodes algorithmiques permettant de regrouper automatiquement des observations en sous-ensembles homogènes, appelés *clusters*, sans qu'aucune étiquette préalable ne soit fournie à l'algorithme. Ce paradigme, qualifié d'**apprentissage non supervisé**, s'oppose à la classification supervisée en ce sens qu'il ne dispose d'aucun signal d'entraînement — il doit découvrir seul la structure latente des données.

### 1.2 Définition formelle du clustering

Soit un ensemble de n observations X = {x₁, x₂, ..., xₙ} où chaque xᵢ ∈ ℝᵈ est un vecteur de d dimensions. L'objectif du clustering est de trouver une partition C = {C₁, C₂, ..., Cₖ} de X en k groupes disjoints telle que :

- Les observations à l'intérieur d'un même cluster soient les plus **similaires** possible entre elles (cohésion intra-cluster maximale)
- Les observations appartenant à des clusters différents soient les plus **dissemblables** possible (séparation inter-cluster maximale)

Formellement, pour une distance d(·,·) donnée :

```
∀i ≠ j, ∀x ∈ Cᵢ, ∀y ∈ Cⱼ : d(x, μᵢ) < d(x, y)
```

où μᵢ désigne le centroïde (barycentre) du cluster Cᵢ.

### 1.3 Importance en Big Data et Machine Learning

Le clustering est une brique fondamentale dans de nombreux systèmes intelligents :

- **Segmentation marketing** : identification de profils clients distincts pour personnaliser les campagnes
- **Détection d'anomalies** : les points très éloignés de tout cluster peuvent signaler des fraudes ou pannes
- **Compression et résumé de données** : remplacer n observations par k centroides réduit la complexité de traitements ultérieurs
- **Recommandation** : les utilisateurs d'un même cluster partagent des goûts similaires
- **Analyse exploratoire** : révéler des structures cachées avant toute modélisation supervisée
- **Médecine de précision** : identifier des sous-groupes de patients répondant différemment à un traitement

Dans un contexte Big Data, où les volumes de données rendent l'inspection manuelle impossible, le clustering constitue l'un des premiers outils d'exploration que le Data Scientist deploy.

### 1.4 Objectifs du TP

Ce travail pratique poursuit les objectifs pédagogiques suivants :

1. Implémenter et exécuter l'algorithme K-means sur une dataset réaliste
2. Implémenter et exécuter le Clustering Hiérarchique Agglomératif (CAH)
3. Maîtriser les méthodes de sélection du nombre optimal de clusters
4. Évaluer la qualité des partitions à l'aide de métriques internes et externes
5. Comparer les deux approches selon des critères multidimensionnels
6. Interpréter les résultats d'un point de vue métier

---

## 2. Présentation de la dataset

### 2.1 Justification du choix

Le cas d'usage retenu est la **segmentation de clients d'une enseigne de grande distribution fictive**, baptisée *MarketIFRI*. Ce choix est motivé par plusieurs considérations :

- **Réalisme** : la segmentation clients est l'une des applications les plus répandues du clustering en entreprise (Amazon, Cdiscount, Carrefour...)
- **Richesse analytique** : les données comportementales de consommation produisent des clusters naturellement interprétables
- **Pédagogie** : les profils obtenus sont facilement compréhensibles et discutables
- **Pertinence** : cette application illustre parfaitement les forces et limites comparatives des deux algorithmes étudiés

### 2.2 Source et génération

La dataset a été synthétiquement générée via le script `generate_dataset.py`, conformément aux bonnes pratiques académiques lorsqu'aucune source de données réelle ne peut être utilisée sans restrictions légales (RGPD, confidentialité). Le processus de génération simule des distributions statistiques réalistes issues de la littérature du marketing quantitatif.

Pour toute exploitation sur données réelles, des datasets comparables sont disponibles sur :
- **Kaggle** : *Mall Customer Segmentation Dataset* (https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)
- **UCI Repository** : *Online Retail Dataset* (https://archive.ics.uci.edu/dataset/352/online+retail)

### 2.3 Description des variables

| Variable | Type | Description | Plage |
|---|---|---|---|
| `client_id` | String | Identifiant unique du client | C00001–C02000 |
| `age` | Integer | Âge du client | 18–75 ans |
| `revenu_annuel` | Float | Revenu annuel estimé | 10 000–150 000 € |
| `score_depenses` | Integer | Propension à dépenser (score calculé) | 0–100 |
| `nb_achats_annuel` | Integer | Nombre d'achats dans l'année | 1–120 |
| `panier_moyen` | Float | Valeur moyenne d'un achat | 5–500 € |
| `anciennete_mois` | Integer | Durée de la relation client (mois) | 1–120 |
| `ratio_promo` | Float | Part des achats en promotion | 0–1 |
| `score_fidelite` | Integer | Score de fidélité composite | 0–100 |
| `categorie_preferee` | String | Catégorie d'achat dominante | 9 catégories |
| `canal_principal` | String | Canal d'achat préférentiel | Magasin / En ligne / App |
| `profil_reel` | String | Label de vérité terrain | 5 profils |

### 2.4 Caractéristiques dimensionnelles

- **Nombre d'observations** : 2 000 clients
- **Nombre de variables** : 12 (dont 10 utilisées pour le clustering)
- **Variables numériques** : 8 (quantitatives continues ou discrètes)
- **Variables catégorielles** : 2 (encodées avant clustering)
- **Taille fichier CSV** : ~153 Ko (bien en deçà de la limite de 100 Mo)
- **Valeurs manquantes** : ~2 % sur 4 colonnes numériques (simulées)

### 2.5 Profils réels simulés (vérité terrain)

Cinq segments clients ont été simulés selon des distributions gaussiennes paramétrées :

| Profil | % | Description comportementale |
|---|---|---|
| Familles Actives | 25 % | Âge moyen, revenu moyen, gros paniers alimentaires, canal mixte |
| Jeunes Économes | 20 % | Jeunes adultes, faible revenu, achats ciblés, très digitaux |
| Seniors Fidèles | 20 % | Seniors, achat régulier, fidélité très élevée, exclusivement en magasin |
| Chasseurs de Promo | 20 % | Tous âges, motivés par les promotions, fort ratio promo |
| Aisés Premium | 15 % | Revenu élevé, paniers très chers, peu sensibles aux promotions |

### 2.6 Statistiques descriptives

| Statistique | Âge | Revenu (€) | Score dépenses | Panier (€) | Score fidélité |
|---|---|---|---|---|---|
| Moyenne | 39,9 | 38 817 | 54,8 | 74,1 | 57,8 |
| Écart-type | 14,5 | 23 265 | 17,6 | 62,6 | 22,8 |
| Min | 18 | 10 000 | 0 | 5 | 0 |
| Médiane | 38 | 33 703 | 54 | 54,7 | 59 |
| Max | 75 | 140 215 | 100 | 366 | 100 |

On observe une hétérogénéité significative sur le revenu annuel (CV ≈ 60 %) et le panier moyen (CV ≈ 85 %), ce qui laisse anticiper des clusters bien différenciés sur ces dimensions.

---

## 3. Prétraitement des données

### 3.1 Gestion des valeurs manquantes

La dataset présente environ 2 % de valeurs manquantes sur quatre variables numériques (`revenu_annuel`, `panier_moyen`, `anciennete_mois`, `ratio_promo`), simulant des données réelles incomplètes. Trois stratégies d'imputation existent :

1. **Suppression des lignes** (*listwise deletion*) : simple mais introduit un biais si les valeurs ne sont pas manquantes de façon totalement aléatoire (MCAR)
2. **Imputation par la moyenne** : sensible aux valeurs extrêmes
3. **Imputation par la médiane** : robuste aux outliers et adaptée aux distributions asymétriques ✓

**Choix retenu :** imputation par la médiane, car les variables concernées présentent des queues de distribution droites (revenus, paniers), et les algorithmes de clustering sont sensibles aux valeurs extrêmes.

```python
for col in ["revenu_annuel", "panier_moyen", "anciennete_mois", "ratio_promo"]:
    df[col].fillna(df[col].median(), inplace=True)
```

Résultats : 137 valeurs manquantes traitées ; aucune valeur résiduelle.

### 3.2 Encodage des variables catégorielles

Les variables `categorie_preferee` (9 modalités) et `canal_principal` (3 modalités) sont transformées en entiers via `LabelEncoder`. Ce choix est discutable pour K-means (LabelEncoder introduit un ordre arbitraire), mais il est justifié ici car :
- Ces variables ont un poids faible dans la variance totale
- Les distances induites restent cohérentes avec l'échelle standardisée
- L'encodage one-hot aurait significativement augmenté la dimensionnalité

Dans un projet de production, on préférerait un encodage ordinal raisonné ou une distance de Gower pour données mixtes.

### 3.3 Normalisation / Standardisation

Les variables numériques ont des échelles très hétérogènes : le revenu varie de 10 000 à 150 000 € tandis que le ratio de promotion varie de 0 à 1. Sans normalisation, les variables à grande amplitude dominent la distance euclidienne, biaisant les clusters vers ces dimensions.

**Choix retenu :** standardisation Z-score via `StandardScaler` de scikit-learn :

```
x'ᵢ = (xᵢ - μ) / σ
```

Cette transformation garantit μ' = 0 et σ' = 1 pour chaque variable, leur conférant un poids équivalent dans le calcul des distances.

La normalisation min-max (0–1) aurait été une alternative valable, mais elle est davantage sensible aux outliers. Le Z-score est préféré en pratique académique et industrielle pour le clustering.

### 3.4 Sélection des features

Les 10 features retenues pour le clustering excluent `client_id` (identifiant non informatif) et `profil_reel` (label de vérité terrain conservé uniquement pour l'évaluation externe). L'ensemble des features numériques et encodées est utilisé, sans réduction de dimensionnalité préalable — la PCA est appliquée uniquement pour la visualisation.

---

## 4. Fondements théoriques du K-means

### 4.1 Principe général

L'algorithme K-means (MacQueen, 1967 ; Lloyd, 1982) est l'algorithme de clustering le plus répandu en pratique. Son objectif est de partitionner n observations en k groupes en minimisant la **somme des carrés intra-clusters** (WCSS — Within-Cluster Sum of Squares), également appelée inertie :

```
J = Σᵢ₌₁ᵏ Σ_{x ∈ Cᵢ} ||x - μᵢ||²
```

où μᵢ = (1/|Cᵢ|) Σ_{x∈Cᵢ} x est le centroïde (barycentre) du cluster i.

Cette formulation est un problème d'optimisation combinatoire NP-difficile dans le cas général. K-means en fournit une solution approchée par descente de coordonnées alternée.

### 4.2 Algorithme EM (Expectation-Maximization)

K-means peut être vu comme un cas particulier de l'algorithme EM :

**Initialisation :**
- Choisir k centres initiaux μ₁⁰, μ₂⁰, ..., μₖ⁰

**Étape E (Expectation) — Affectation :**
- Pour chaque observation xⱼ, l'affecter au cluster le plus proche :
```
cⱼ = argminᵢ ||xⱼ - μᵢ||²
```

**Étape M (Maximization) — Mise à jour :**
- Recalculer chaque centroïde comme la moyenne des observations affectées :
```
μᵢ ← (1/|Cᵢ|) Σ_{j:cⱼ=i} xⱼ
```

**Convergence :** répéter E et M jusqu'à ce que les assignations ne changent plus, ou que la variation d'inertie soit inférieure à un seuil ε.

### 4.3 Distance euclidienne

La distance utilisée est la **distance euclidienne** (norme L2) entre deux points x et y dans ℝᵈ :

```
d(x, y) = √(Σⱼ₌₁ᵈ (xⱼ - yⱼ)²)
```

Ce choix implique que K-means suppose des clusters **convexes et isotropes** (sphériques). Il est inadapté pour des clusters de formes allongées, en croissant ou non convexes — limitation importante discutée en §4.6.

### 4.4 Initialisation K-means++

L'initialisation naïve (centres aléatoires uniformes) peut conduire à de mauvais minima locaux. **K-means++** (Arthur & Vassilvitskii, 2007) propose une initialisation probabiliste plus intelligente :

1. Choisir le premier centre μ₁ uniformément au hasard
2. Pour chaque centre suivant μₗ, choisir xᵢ avec probabilité proportionnelle à d(xᵢ, C)² où C est l'ensemble des centres déjà choisis

Cette initialisation garantit une approximation O(log k) de l'optimum et réduit drastiquement la variance entre exécutions. Elle est le défaut dans scikit-learn (`init="k-means++"`).

### 4.5 Convergence

L'algorithme converge toujours en un nombre fini d'itérations car :
- Il y a un nombre fini de partitions possibles (Sₙ,ₖ — nombre de Stirling)
- L'inertie J est strictement décroissante à chaque itération E-M

Cependant, la solution obtenue peut être un **minimum local** et non global. C'est pourquoi il est recommandé d'exécuter l'algorithme plusieurs fois avec des initialisations différentes (`n_init=20` dans notre implémentation) et de retenir la partition minimisant l'inertie.

### 4.6 Avantages et limites

**Avantages :**
- Simple à implémenter et à interpréter
- Complexité quasi-linéaire : O(n · k · d · i) où i = nombre d'itérations
- Très efficace pour de grands volumes de données
- Convergence garantie
- Clusters compacts et bien séparés lorsque les hypothèses sont vérifiées

**Limites :**
- Nécessite de spécifier k a priori
- Sensible aux outliers (qui déplacent les centroïdes)
- Suppose des clusters sphériques de taille comparable
- Dépend de l'initialisation (minima locaux)
- Ne gère pas nativement les données mixtes (numériques + catégorielles)
- Instable pour des clusters très déséquilibrés en taille

### 4.7 Complexité computationnelle

| Aspect | Complexité |
|---|---|
| Par itération | O(n · k · d) |
| Total (t itérations) | O(n · k · d · t) |
| Espace mémoire | O(n · d + k · d) |

Pour n = 2 000, k = 5, d = 10, t = 6 (itérations observées) : ~600 000 opérations — calcul quasi-instantané.

---

## 5. Implémentation et résultats K-means

### 5.1 Paramétrage du modèle

```python
km = KMeans(
    n_clusters   = 5,          # k choisi par méthode Elbow + Silhouette
    init         = "k-means++",# Initialisation intelligente
    n_init       = 20,         # 20 initialisations → retient la meilleure
    max_iter     = 500,        # Maximum d'itérations par run
    tol          = 1e-4,       # Seuil de convergence sur les centroïdes
    random_state = 42,         # Reproductibilité
)
```

**Justification de k=5 :** bien que l'analyse automatique du Silhouette Score suggère k=3 comme optimum statistique, le choix de k=5 est justifié par la connaissance métier des 5 profils clients réels. Cette situation — décision hybride algorithme/expert — est standard en Data Science pratique. Par ailleurs, l'Elbow curve montre un coude marqué entre k=3 et k=5, confirmant que 5 groupes est une partition raisonnable.

### 5.2 Résultats de la convergence

```
Convergence en 6 itérations
Inertie finale : 7 065.15
```

La convergence rapide en 6 itérations témoigne d'une structure de données bien définie et de l'efficacité de l'initialisation K-means++.

### 5.3 Distribution des clusters obtenus

| Cluster | Effectif | Proportion | Profil identifié |
|---|---|---|---|
| 0 | 372 | 18,6 % | Chasseurs de promotions |
| 1 | 291 | 14,6 % | Aisés premium |
| 2 | 389 | 19,4 % | Seniors fidèles |
| 3 | 521 | 26,1 % | Familles actives |
| 4 | 427 | 21,4 % | Jeunes économes |

### 5.4 Métriques d'évaluation

| Métrique | Valeur | Interprétation |
|---|---|---|
| **Silhouette Score** | 0.3002 | Cohésion/séparation acceptables pour 10D |
| **Calinski-Harabász** | 903.44 | Bon rapport variance inter/intra |
| **Davies-Bouldin** | 1.3248 | Similarité inter-clusters modérée |
| **ARI (vs vérité)** | **0.9204** | Excellent accord avec les profils réels |

Un **ARI de 0.92** signifie que K-means retrouve correctement 92 % des affectations réelles — performance remarquable pour un algorithme non supervisé sur données à 10 dimensions. Cela valide la robustesse des profils simulés et la pertinence du clustering.

Le Silhouette Score de ~0.30, bien qu'inférieur à 0.5, est typique pour des espaces de haute dimensionnalité (10D) où la concentration des distances (phénomène du *curse of dimensionality*) compresse les écarts relatifs.

### 5.5 Profils moyens des clusters K-means

| Cluster | Âge | Revenu | Score dépen. | Panier | Fidélité | Ratio promo |
|---|---|---|---|---|---|---|
| 0 (Chasseurs promo) | 36 | 25 187 € | 55 | 38,8 € | 46 | 0.70 |
| 1 (Aisés premium) | 43 | 85 544 € | 80 | 201,2 € | 72 | 0.08 |
| 2 (Seniors fidèles) | 62 | 35 517 € | 49 | 55,0 € | 84 | 0.20 |
| 3 (Familles actives) | 38 | 41 490 € | 60 | 83,4 € | 62 | 0.30 |
| 4 (Jeunes économes) | 24 | 18 350 € | 36 | 23,8 € | 29 | 0.48 |

Ces profils sont remarquablement distincts et interprétables — signe d'une partition de qualité.

---

## 6. Fondements théoriques du Clustering Hiérarchique

### 6.1 Principe général

Le Clustering Hiérarchique (ou CAH — Classification Ascendante Hiérarchique) est une famille d'algorithmes qui construit une **hiérarchie de partitions** plutôt qu'une partition unique. Il en existe deux variantes :

- **Agglomératif** (*bottom-up*) : chaque observation commence comme son propre cluster, puis les clusters les plus proches sont fusionnés progressivement → approche retenue
- **Divisif** (*top-down*) : tous les points forment un seul cluster, puis on subdivise récursivement

L'approche agglomérative, plus courante, produit un **dendrogramme** : une structure arborescente représentant la séquence des fusions, permettant de visualiser la hiérarchie complète des clusters à toutes les granularités.

### 6.2 Le Dendrogramme

Le dendrogramme est un graphique bipartite où :
- L'axe horizontal représente les observations (ou clusters)
- L'axe vertical représente la **distance de fusion** (hauteur à laquelle deux clusters sont fusionnés)
- Chaque jointure horizontale représente une fusion de deux clusters

Pour obtenir k clusters à partir du dendrogramme, on effectue une **coupure horizontale** à la hauteur h telle que k lignes verticales soient traversées. En pratique, on coupe là où le saut vertical (entre deux fusions successives) est maximal — signe que les clusters fusionnés sont de plus en plus distincts.

### 6.3 Méthodes de liaison (Linkage)

La notion de *distance entre deux clusters* Cᵢ et Cⱼ dépend de la méthode de liaison choisie :

| Méthode | Formule | Propriétés |
|---|---|---|
| **Single linkage** | min_{x∈Cᵢ, y∈Cⱼ} d(x,y) | Sensible aux chaînes ; clusters allongés |
| **Complete linkage** | max_{x∈Cᵢ, y∈Cⱼ} d(x,y) | Clusters compacts ; sensible aux outliers |
| **Average linkage** | (1/|Cᵢ||Cⱼ|) Σ d(x,y) | Compromis robuste |
| **Ward** | ΔJ = J(Cᵢ∪Cⱼ) − J(Cᵢ) − J(Cⱼ) | Minimise la variance ; clusters équilibrés ✓ |

**Choix de la méthode Ward :** la liaison de Ward (Ward, 1963) fusionne à chaque étape les deux clusters dont la fusion minimise l'augmentation de l'inertie totale. Elle tend à produire des clusters compacts et de taille équilibrée, ce qui la rend particulièrement adaptée à la segmentation clients où l'on souhaite des groupes homogènes et interprétables. C'est la méthode recommandée par défaut dans la littérature (Murtagh & Legendre, 2014).

### 6.4 Coefficient Cophénétique

Le **coefficient cophénétique** c mesure la corrélation entre les distances de fusion du dendrogramme et les distances originales entre points. Il quantifie la qualité de représentation hiérarchique :

```
c = corr(d_originale(xᵢ, xⱼ), d_dendro(xᵢ, xⱼ))
```

- c > 0.90 : excellente représentation
- c > 0.75 : bonne représentation
- c < 0.60 : représentation insuffisante

Dans nos expériences : **c = 0.693** — représentation acceptable, cohérente avec la haute dimensionnalité (10D) qui compresse les contrastes de distance.

### 6.5 Avantages et limites

**Avantages :**
- Ne nécessite pas de spécifier k a priori (la coupure peut être choisie a posteriori)
- Produit une représentation hiérarchique complète (dendrogramme)
- Déterministe (pas d'aléatoire dans Ward)
- Interprétation à plusieurs niveaux de granularité
- Compatible avec toute mesure de distance ou de similarité

**Limites :**
- Complexité quadratique en mémoire : O(n²) — problématique pour n > 100 000
- Complexité algorithmique naïve : O(n³) ; réduite à O(n² log n) avec les algorithmes modernes (SLINK, CLINK)
- Pas de réaffectation possible après fusion (erreurs cumulées)
- Sensible aux outliers (surtout en single et complete linkage)
- Dendrogramme illisible pour n > quelques milliers

### 6.6 Complexité computationnelle

| Aspect | Complexité |
|---|---|
| Calcul de la matrice de distances | O(n² · d) |
| Algorithme de Ward naïf | O(n³) |
| Ward avec algorithme optimisé | O(n² log n) |
| Espace mémoire (matrice dist.) | O(n²) |

Pour n = 2 000 : la matrice de distances nécessite ~32 Mo — gérable. Pour n > 50 000, on doit sous-échantillonner ou utiliser des approximations (mini-batch, BIRCH).

---

## 7. Implémentation et résultats du Clustering Hiérarchique

### 7.1 Stratégie d'implémentation

L'implémentation repose sur deux composants complémentaires :

1. **scipy.cluster.hierarchy** : pour le calcul du dendrogramme sur un sous-échantillon de 500 observations (lisibilité + performance)
2. **sklearn.cluster.AgglomerativeClustering** : pour l'attribution des labels sur la dataset complète (2 000 observations)

```python
# Dendrogramme sur sous-échantillon
Z = linkage(X_sample, method="ward", metric="euclidean")

# Clustering complet
modele = AgglomerativeClustering(
    n_clusters = 5,
    linkage    = "ward",
    metric     = "euclidean",
)
labels = modele.fit_predict(X_scaled)
```

### 7.2 Résultats du coefficient cophénétique

```
Coefficient cophénétique : 0.6928
Qualité de représentation : Acceptable
```

Cette valeur modérée s'explique par la haute dimensionnalité des données (10D), phénomène bien documenté dans la littérature (Houle et al., 2010). En espace de haute dimension, les distances sont concentrées et les structures hiérarchiques moins marquées.

### 7.3 Distribution des clusters Hclust

| Cluster | Effectif | Proportion | Profil identifié |
|---|---|---|---|
| 0 | 405 | 20,2 % | Seniors fidèles |
| 1 | 484 | 24,2 % | Familles actives |
| 2 | 308 | 15,4 % | Aisés premium |
| 3 | 414 | 20,7 % | Jeunes économes |
| 4 | 389 | 19,4 % | Chasseurs de promotions |

### 7.4 Métriques d'évaluation

| Métrique | Valeur | Interprétation |
|---|---|---|
| **Silhouette Score** | 0.2928 | Légèrement inférieur à K-means |
| **Calinski-Harabász** | 884.81 | Bon ; légèrement inférieur à K-means |
| **Davies-Bouldin** | 1.3511 | Comparable à K-means |
| **ARI (vs vérité)** | **0.9498** | Légèrement supérieur à K-means |

L'**ARI de 0.95** de Hclust est remarquable : le clustering hiérarchique retrouve encore mieux les profils réels que K-means. Cela s'explique par le fait que Ward minimise exactement le même critère d'inertie que K-means, mais en explorant la hiérarchie complète plutôt qu'un minimum local.

### 7.5 Profils moyens des clusters Hclust

| Cluster | Âge | Revenu | Score dépen. | Panier | Fidélité | Ratio promo |
|---|---|---|---|---|---|---|
| 0 (Seniors fidèles) | 61 | 35 426 € | 50 | 55,1 € | 84 | 0.20 |
| 1 (Familles actives) | 37 | 41 331 € | 60 | 83,8 € | 61 | 0.30 |
| 2 (Aisés premium) | 43 | 83 997 € | 79 | 196,0 € | 72 | 0.08 |
| 3 (Jeunes économes) | 24 | 18 169 € | 36 | 23,2 € | 29 | 0.48 |
| 4 (Chasseurs promo) | 36 | 25 245 € | 55 | 39,3 € | 46 | 0.70 |

Les profils obtenus sont quasi-identiques à ceux de K-means — témoignant d'une structure de données robuste et de la cohérence des deux méthodes.

---

## 8. Visualisations et analyse graphique

### 8.1 Méthode Elbow (K-means)

La courbe d'inertie en fonction de k montre une décroissance continue avec un coude (*elbow*) observable entre k=3 et k=5. Ce coude indique que l'ajout de clusters au-delà de k=5 n'améliore plus significativement la compacité des clusters. La méthode Elbow confirme k=5 comme choix raisonnable.

**Fichier :** `figures/kmeans_elbow_silhouette.png`

### 8.2 Silhouette Score

Le Silhouette Score maximal est atteint pour k=3 (0.319), mais k=5 présente un score de 0.293 — perte marginale compensée par une interprétabilité métier bien supérieure. Cette tension entre performance statistique et interprétabilité métier est un enjeu central en Data Science appliquée.

### 8.3 Projection PCA (K-means et Hclust)

La PCA réduit les 10 dimensions à 2 composantes principales expliquant environ 55-60 % de la variance totale. Les projections 2D révèlent :
- Cinq groupes visuellement distincts malgré la perte d'information
- Le cluster "Aisés Premium" très isolé (revenu et panier bien au-dessus des autres)
- Un chevauchement partiel entre "Jeunes Économes" et "Chasseurs de Promo" (similarités de revenu)

**Fichiers :** `figures/kmeans_pca_clusters.png`, `figures/hclust_pca_clusters.png`

### 8.4 Dendrogramme hiérarchique

Le dendrogramme tronqué (30 dernières fusions) montre clairement deux niveaux de regroupement naturels :
- Une fusion à haute distance entre le cluster "Aisés Premium" et le reste
- Une structure interne à quatre groupes pour les clients non-premium

La coupure horizontale à la hauteur appropriée isole proprement 5 clusters.

**Fichier :** `figures/hclust_dendrogramme.png`

### 8.5 Heatmaps des profils

Les heatmaps normalisées permettent de comparer visuellement les profils moyens des 5 clusters sur l'ensemble des variables. On observe :
- Le cluster "Aisés Premium" avec revenu et panier très élevés (valeurs max normalisées)
- Le cluster "Seniors Fidèles" avec ancienneté et fidélité maximales
- Le cluster "Chasseurs de Promo" avec ratio promo maximal et fidélité faible

**Fichiers :** `figures/kmeans_heatmap_profils.png`, `figures/hclust_heatmap_profils.png`

### 8.6 Boxplots comparatifs

Les boxplots des quatre variables les plus discriminantes (revenu, panier, fidélité, ratio promo) confirment la séparation nette des clusters et la faible dispersion intra-cluster, particulièrement pour K-means.

**Fichier :** `figures/kmeans_boxplots.png`

### 8.7 Comparaison côte-à-côte

La figure de comparaison PCA côte-à-côte montre que les deux méthodes identifient des partitions quasi-identiques — avec un accord inter-méthodes de ARI = 0.92 — validant la robustesse de la structure détectée.

**Fichier :** `figures/comparaison_kmeans_hclust.png`

---

## 9. Analyse comparative K-means vs Hclust

### 9.1 Comparaison quantitative des métriques

| Métrique | K-Means | Hclust Ward | Gagnant |
|---|---|---|---|
| Silhouette Score (↑) | **0.3002** | 0.2928 | K-Means |
| Calinski-Harabász (↑) | **903.44** | 884.81 | K-Means |
| Davies-Bouldin (↓) | **1.3248** | 1.3511 | K-Means |
| ARI vs vérité terrain (↑) | 0.9204 | **0.9498** | Hclust |
| Accord inter-méthodes (ARI) | — | 0.9213 | — |

K-Means présente de meilleures métriques internes (cohésion/séparation), tandis que Hclust retrouve légèrement mieux les partitions réelles (ARI externe). Les deux méthodes produisent des partitions fortement concordantes.

### 9.2 Comparaison qualitative

| Dimension | K-Means | Clustering Hiérarchique |
|---|---|---|
| **Spécification de k** | Requis a priori | Peut être choisi a posteriori |
| **Type de résultat** | Partition plate | Hiérarchie complète (dendrogramme) |
| **Reproductibilité** | Semi-déterministe (init aléatoire) | Déterministe (Ward) |
| **Scalabilité** | Excellente — O(nkdt) | Limitée — O(n² log n) |
| **Sensibilité outliers** | Modérée | Variable selon le linkage |
| **Forme des clusters** | Sphérique (convexe) | Variable |
| **Interprétabilité** | Centroïdes directs | Dendrogramme (visuel) |
| **Gestion des grandes données** | Excellent (mini-batch possible) | Problématique (>50k obs) |
| **Analyse multi-granularité** | Non | Oui |
| **Temps de calcul (n=2000)** | 2.81 s | 1.89 s |

### 9.3 Analyse des cas d'usage recommandés

**K-Means est préférable quand :**
- Le volume de données est grand (n > 10 000)
- On a une hypothèse a priori sur k
- On souhaite des clusters compacts et sphériques
- La rapidité d'exécution est critique (production, streaming)
- On veut des centroïdes interprétables directement

**Hclust est préférable quand :**
- On veut explorer la hiérarchie des groupes sans fixer k a priori
- Le volume de données est modéré (n < 10 000)
- On a besoin d'un outil de visualisation (dendrogramme) pour la présentation
- La structure des données est hiérarchique par nature
- La reproductibilité totale est requise

**Dans notre cas :** les deux méthodes sont applicables (n=2000). K-Means est légèrement plus compact; Hclust offre le dendrogramme. Une stratégie hybride — explorer avec Hclust pour choisir k, puis affiner avec K-Means — est souvent optimale.

### 9.4 Discussion sur la convergence des résultats

Le très fort accord inter-méthodes (ARI = 0.92) est une observation importante : il signifie que la structure en 5 clusters n'est pas un artefact d'un algorithme particulier, mais bien une **propriété intrinsèque des données**. Lorsque K-Means et Hclust s'accordent à ce niveau, le praticien peut avoir une haute confiance dans la validité des clusters identifiés.

Ce phénomène s'explique théoriquement : Ward minimise la même fonction d'inertie que K-Means, mais en parcourant l'espace des partitions de façon déterministe et hiérarchique plutôt que par descente stochastique.

---

## 10. Interprétation métier des clusters

### 10.1 Cluster "Aisés Premium" (≈ 15 % des clients)

**Caractéristiques :** Âge 43 ans, revenu >83 000 €/an, panier moyen 200 €, score fidélité 72, ratio promo 0.08.

**Interprétation :** Ce segment représente la clientèle à forte valeur ajoutée. Ces clients achètent peu en volume mais leur panier unitaire est 3-4x supérieur à la moyenne. Ils sont peu sensibles aux promotions (ratio promo < 10 %), ce qui indique une décision d'achat guidée par la qualité et la marque plutôt que par le prix. Leur ancienneté moyenne (48 mois) témoigne d'une relation commerciale solide.

**Recommandations marketing :** Programme de fidélité premium, invitations à des événements exclusifs, communication sur la qualité et le prestige des marques. Ne pas cibler avec des promotions massives qui pourraient dévaloriser l'image perçue.

### 10.2 Cluster "Familles Actives" (≈ 25 % — segment dominant)

**Caractéristiques :** Âge 37-38 ans, revenu ≈41 000 €, panier 83 €, 48 achats/an, fidélité 62, canal mixte.

**Interprétation :** Cœur de cible de l'enseigne. Fréquence d'achat élevée, panier significatif, bonne fidélité. Les achats couvrent l'alimentaire, l'hygiène et la maison — profil typique du responsable des achats du foyer. L'utilisation mixte des canaux (en magasin + en ligne) indique une clientèle omnicanal.

**Recommandations :** Offres multi-achats et économies sur le volume, programme de fidélité avec bons de réduction progressifs, click-and-collect, services de livraison à domicile.

### 10.3 Cluster "Seniors Fidèles" (≈ 20 % des clients)

**Caractéristiques :** Âge 61-62 ans, ancienneté 72 mois (6 ans), score fidélité 84 (le plus élevé), 60 achats/an, exclusivement en magasin.

**Interprétation :** Segment de fidélité extrême. Ces clients représentent le fonds de commerce historique de l'enseigne. Leur ancienneté très élevée et leur score de fidélité max en font les ambassadeurs naturels de la marque. Leur refus du digital (canal magasin uniquement) est une contrainte forte sur les canaux de communication.

**Recommandations :** Maintenir l'expérience en magasin de qualité, programme de reconnaissance de l'ancienneté (avantages exclusifs seniors), accompagnement humain, horaires dédiés, éviter les communications exclusivement digitales.

### 10.4 Cluster "Jeunes Économes" (≈ 20 % des clients)

**Caractéristiques :** Âge 24 ans, revenu ≈18 000 € (le plus faible), score fidélité 29 (faible), panier 24 €, ratio promo 0.48.

**Interprétation :** Clients à fort potentiel de croissance. Leur faible revenu actuel et faible ancienneté suggèrent des clients en début de vie active, dont le pouvoir d'achat est amené à croître. La fidélité faible et l'usage intensif des applications mobiles indiquent un comportement très opportuniste et digital-native.

**Recommandations :** Applications mobiles avec gamification et cashback, programme d'entrée en fidélisation avec avantages progressifs, offres "premier achat", communication 100 % mobile et réseaux sociaux. Stratégie de long terme pour fidéliser avant la croissance du pouvoir d'achat.

### 10.5 Cluster "Chasseurs de Promotions" (≈ 20 % des clients)

**Caractéristiques :** Ratio promo 0.70 (le plus élevé de loin), revenu modeste ≈25 000 €, 55 achats/an (fréquence élevée), fidélité 46 (faible).

**Interprétation :** Segment hautement stratégique mais dangereux. Ces clients génèrent du trafic et du volume, mais leur contribution à la marge est faible (ils achètent quasi-exclusivement en promotion). Leur fidélité faible indique qu'ils arbitrent entre enseignes selon les meilleures offres — comportement opportuniste pur.

**Recommandations :** Stratégie délicate : les promotions les attirent mais ne les fidélisent pas. L'objectif est de les convertir progressivement vers des achats hors promotion via des offres de fidélisation personnalisées. Analyser leur rentabilité réelle avant d'augmenter les dépenses marketing sur ce segment.

### 10.6 Valeur client estimée par segment (LTV)

En estimant la Lifetime Value annuelle (panier × achats) :

| Segment | LTV annuelle estimée | Priorité stratégique |
|---|---|---|
| Aisés Premium | 200 × 35 = **7 000 €** | ★★★★★ Rétention absolue |
| Familles Actives | 83 × 48 = **3 984 €** | ★★★★☆ Croissance |
| Seniors Fidèles | 55 × 60 = **3 300 €** | ★★★★☆ Fidélisation |
| Chasseurs Promo | 39 × 55 = **2 145 €** | ★★★☆☆ Conversion |
| Jeunes Économes | 24 × 20 = **480 €** | ★★★☆☆ Investissement futur |

---

## 11. Conclusion

### 11.1 Résumé des résultats

Ce travail pratique a démontré la faisabilité et la pertinence du clustering non supervisé pour la segmentation clients dans un contexte de distribution. Les deux algorithmes implémentés — K-Means et Clustering Hiérarchique Agglomératif avec la méthode Ward — ont produit des partitions robustes et cohérentes :

- **K-Means** (k=5) : ARI=0.92, Silhouette=0.30, convergence en 6 itérations
- **Hclust Ward** (k=5) : ARI=0.95, Silhouette=0.29, coefficient cophénétique=0.69
- **Accord inter-méthodes** : ARI=0.92 — très forte convergence confirmant la réalité de la structure

Les 5 segments identifiés (Aisés Premium, Familles Actives, Seniors Fidèles, Jeunes Économes, Chasseurs de Promotions) sont non seulement statistiquement distincts, mais aussi métier-interprétables, ouvrant la voie à des stratégies marketing différenciées et à fort impact.

### 11.2 Limites du travail

- **Données synthétiques :** bien que réalistes, les données générées ont des structures plus propres que des données réelles. En production, le bruit, les outliers et les distributions mixtes rendraient les frontières de clusters moins nettes.
- **Hypothèse de sphéricité** : K-Means suppose des clusters sphériques. Des clusters de formes complexes (croissants, spirales) nécessiteraient des algorithmes comme DBSCAN ou Spectral Clustering.
- **Dimensionnalité** : les 10 dimensions de la dataset font que la PCA 2D ne capture que ~55 % de la variance, réduisant la lisibilité des visualisations.
- **Stabilité temporelle** : les clusters identifiés sont une photo à un instant t. En réalité, les comportements clients évoluent et nécessitent une ré-exécution périodique du clustering.
- **Variable cat_enc** : l'encodage ordinal des catégories introduit un ordre arbitraire potentiellement biaisé.

### 11.3 Améliorations possibles

1. **DBSCAN** : algorithme basé sur la densité, ne nécessite pas de spécifier k, détecte les outliers
2. **Gaussian Mixture Models (GMM)** : clustering probabiliste, gère les ellipsoïdes, fournit des probabilités d'appartenance
3. **K-Prototypes** : extension de K-Means pour données mixtes (numériques + catégorielles)
4. **Distance de Gower** : distance unifiée pour données mixtes, utilisable avec Hclust
5. **t-SNE / UMAP** : réduction de dimensionnalité non-linéaire pour meilleures visualisations
6. **Validation temporelle** : tracking des clusters sur plusieurs périodes pour détecter les migrations
7. **Clustering spectral** : efficace pour des clusters non convexes
8. **AutoML Clustering** : outils comme Auto-sklearn pour sélection automatique d'algorithme et d'hyperparamètres

### 11.4 Ouverture

Le clustering constitue une étape fondatrice dans de nombreux pipelines avancés de Machine Learning. Les segments identifiés peuvent alimenter :
- Des **systèmes de recommandation** (collaborative filtering par cluster)
- Des **modèles prédictifs du churn** (risque de désabonnement par segment)
- Des **expériences A/B testing** ciblées (propositions différenciées par profil)
- Des **tableaux de bord BI** dynamiques pour le suivi des KPIs par segment

Au-delà du cas clients, les techniques présentées s'appliquent directement à l'analyse génomique (clustering de gènes), à la détection d'anomalies réseau, à la compression d'images, ou à la robotique cognitive — illustrant l'universalité du paradigme du clustering en intelligence artificielle.

---

## 12. Références bibliographiques

1. **MacQueen, J.B.** (1967). Some methods for classification and analysis of multivariate observations. *Proceedings of the 5th Berkeley Symposium on Mathematical Statistics and Probability*, vol. 1, pp. 281-297.

2. **Lloyd, S.P.** (1982). Least squares quantization in PCM. *IEEE Transactions on Information Theory*, 28(2), 129-137.

3. **Arthur, D., & Vassilvitskii, S.** (2007). K-means++: The advantages of careful seeding. *Proceedings of the 18th Annual ACM-SIAM Symposium on Discrete Algorithms (SODA)*, pp. 1027-1035.

4. **Ward, J.H.** (1963). Hierarchical grouping to optimize an objective function. *Journal of the American Statistical Association*, 58(301), 236-244.

5. **Murtagh, F., & Legendre, P.** (2014). Ward's hierarchical agglomerative clustering method: which algorithms implement Ward's criterion? *Journal of Classification*, 31(3), 274-295.

6. **Rousseeuw, P.J.** (1987). Silhouettes: A graphical aid to the interpretation and validation of cluster analysis. *Journal of Computational and Applied Mathematics*, 20, 53-65.

7. **Calinski, T., & Harabasz, J.** (1974). A dendrite method for cluster analysis. *Communications in Statistics*, 3(1), 1-27.

8. **Davies, D.L., & Bouldin, D.W.** (1979). A cluster separation measure. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 1(2), 224-227.

9. **Hubert, L., & Arabie, P.** (1985). Comparing partitions. *Journal of Classification*, 2(1), 193-218.

10. **Pedregosa, F., et al.** (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.

11. **Hastie, T., Tibshirani, R., & Friedman, J.** (2009). *The Elements of Statistical Learning* (2nd ed.). Springer.

12. **Bishop, C.M.** (2006). *Pattern Recognition and Machine Learning*. Springer.

---

## 13. Annexes

### Annexe A — Structure des fichiers du projet

```
tp2_clustering/
├── data/
│   ├── clients_marketifri.csv     # Dataset générée (153 Ko)
│   ├── resultats_kmeans.csv       # Dataset + labels K-Means
│   └── resultats_hclust.csv       # Dataset + labels Hclust
│
├── figures/
│   ├── kmeans_elbow_silhouette.png
│   ├── kmeans_pca_clusters.png
│   ├── kmeans_heatmap_profils.png
│   ├── kmeans_boxplots.png
│   ├── hclust_dendrogramme.png
│   ├── hclust_sauts_fusion.png
│   ├── hclust_pca_clusters.png
│   ├── hclust_heatmap_profils.png
│   ├── comparaison_kmeans_hclust.png
│   └── comparaison_metriques.png
│
├── scripts/
│   ├── generate_dataset.py        # Génération de la dataset
│   ├── kmeans_clustering.py       # Pipeline K-Means complet
│   ├── hclust_clustering.py       # Pipeline Hclust complet
│   └── main_clustering.py         # Script principal (lance tout)
│
└── rapport_tp2.md                 # Ce rapport
```

### Annexe B — Environnement technique

| Composant | Version |
|---|---|
| Python | 3.12 |
| scikit-learn | ≥ 1.4 |
| scipy | ≥ 1.12 |
| numpy | ≥ 1.26 |
| pandas | ≥ 2.2 |
| matplotlib | ≥ 3.8 |
| seaborn | ≥ 0.13 |

**Installation en une commande :**
```bash
pip install scikit-learn scipy numpy pandas matplotlib seaborn
```

### Annexe C — Commandes d'exécution

```bash
# 1. Générer la dataset
python scripts/generate_dataset.py

# 2. Exécuter uniquement K-Means
python scripts/kmeans_clustering.py

# 3. Exécuter uniquement Hclust
python scripts/hclust_clustering.py

# 4. Tout exécuter (recommandé)
python scripts/main_clustering.py
```

### Annexe D — Métriques détaillées par cluster K-Means

| Cluster | n | Âge | Revenu € | Dépen. | Achats | Panier € | Ancien. | Promo | Fidélité |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 372 | 36.0 | 25 187 | 54.9 | 42.1 | 38.8 | 28.1 | 0.70 | 46.0 |
| 1 | 291 | 43.4 | 85 544 | 80.0 | 34.2 | 201.2 | 48.8 | 0.08 | 71.5 |
| 2 | 389 | 61.5 | 35 517 | 49.4 | 59.6 | 55.0 | 70.9 | 0.20 | 84.0 |
| 3 | 521 | 37.7 | 41 490 | 59.7 | 47.7 | 83.4 | 36.2 | 0.30 | 62.3 |
| 4 | 427 | 24.1 | 18 350 | 36.3 | 20.7 | 23.8 | 14.5 | 0.48 | 29.4 |

---

*Rapport rédigé dans le cadre du TP2 — Clustering K-means / Hclust*  
*IFRI — Master 1 Génie Logiciel — Année 2024–2025*
