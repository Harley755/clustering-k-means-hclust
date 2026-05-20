# TP2 : Clustering K-means / Hclust

**Établissement :** Institut de Formation et de Recherche en Informatique (IFRI)  
**Formation :** Master 1 Génie Logiciel  
**Module :** Big Data & Machine Learning  
**Année académique :** 2024–2025

---

## 📋 Table des matières

1. [Description du projet](#description-du-projet)
2. [Structure du projet](#structure-du-projet)
3. [Prérequis et installation](#prérequis-et-installation)
4. [Exécution du projet](#exécution-du-projet)
5. [Description des scripts](#description-des-scripts)
6. [Dataset](#dataset)
7. [Résultats et visualisations](#résultats-et-visualisations)
8. [Méthodologie](#méthodologie)
9. [Interprétation des résultats](#interprétation-des-résultats)
10. [Comparaison K-means vs Hclust](#comparaison-k-means-vs-hclust)

---

## 📖 Description du projet

Ce projet de clustering non supervisé implémente et compare deux algorithmes de clustering sur une dataset synthétique de segmentation clients :

- **K-means** : Algorithme de partitionnement itératif basé sur la minimisation de l'inertie intra-cluster
- **Clustering Hiérarchique (Hclust)** : Classification Ascendante Hiérarchique (CAH) avec méthode de liaison Ward

L'objectif est d'identifier 5 profils clients distincts pour une enseigne de grande distribution fictive "MarketIFRI".

### Objectifs pédagogiques

- Implémenter K-means et le clustering hiérarchique
- Maîtriser les méthodes de sélection du nombre optimal de clusters (Elbow, Silhouette)
- Évaluer la qualité des partitions avec des métriques internes et externes
- Comparer les deux approches selon des critères multidimensionnels
- Interpréter les résultats d'un point de vue métier

---

## 📁 Structure du projet

```
tp2/
├── generate_dataset.py          # Génération de la dataset synthétique
├── kmeans_clustering.py         # Implémentation K-means
├── hclust_clustering.py         # Implémentation Clustering Hiérarchique
├── main_clustering.py           # Point d'entrée principal
├── rapport_tp2.md               # Rapport académique détaillé
└── README.md                    # Ce fichier

data/                          #   Données générées
│   ├── clients_marketifri.csv   # Dataset originale (2000 clients)
│   ├── resultats_kmeans.csv     # Résultats K-means avec labels
│   └── resultats_hclust.csv     # Résultats Hclust avec labels
figures/                       #   Visualisations générées
    ├── kmeans_elbow_silhouette.png
    ├── kmeans_pca_clusters.png
    ├── kmeans_heatmap_profils.png
    ├── kmeans_boxplots.png
    ├── hclust_dendrogramme.png
    ├── hclust_sauts_fusion.png
    ├── hclust_pca_clusters.png
    ├── hclust_heatmap_profils.png
    ├── comparaison_kmeans_hclust.png
    └── comparaison_metriques.png
```

---

## 🔧 Prérequis et installation

### Système d'exploitation

- Linux (testé)
- macOS (compatible)
- Windows (compatible avec ajustements de chemins)

### Python

- **Python 3.8+** recommandé

### Dépendances

Installer les packages Python requis :

```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy
```

Ou créer un environnement virtuel :

```bash
python -m venv venv
source venv/bin/activate  # Sur Windows : venv\Scripts\activate
pip install numpy pandas matplotlib seaborn scikit-learn scipy
```

### Versions testées

- numpy >= 1.20.0
- pandas >= 1.3.0
- matplotlib >= 3.3.0
- seaborn >= 0.11.0
- scikit-learn >= 0.24.0
- scipy >= 1.7.0

---

## 🚀 Exécution du projet

### Exécution complète (recommandée)

Exécutez le script principal qui lance tout le pipeline :

```bash
cd /home/harley/Documents/big_data/tp2
python main_clustering.py
```

Ce script exécute automatiquement :
1. Génération de la dataset (si elle n'existe pas)
2. Pipeline K-means complet
3. Pipeline Clustering Hiérarchique complet
4. Analyse comparative K-means vs Hclust
5. Génération du rapport de synthèse

### Exécution individuelle des modules

Vous pouvez également exécuter chaque module séparément :

```bash
# 1. Générer la dataset
python generate_dataset.py

# 2. Exécuter K-means uniquement
python kmeans_clustering.py

# 3. Exécuter le clustering hiérarchique uniquement
python hclust_clustering.py
```

### Exécution depuis le sous-répertoire

Si vous travaillez dans le sous-répertoire `TP2_Clustering_IFRI/tp2_clustering/` :

```bash
cd TP2_Clustering_IFRI/tp2_clustering/scripts
python main_clustering.py
```

---

## 📜 Description des scripts

### 1. generate_dataset.py

Génère une dataset synthétique de 2000 clients avec 10 variables comportementales.

**Fonctionnalités :**
- Simulation de 5 profils clients distincts (Jeunes Économes, Familles Actives, Seniors Fidèles, Aisés Premium, Chasseurs de Promo)
- Génération de variables : âge, revenu annuel, score dépenses, nombre d'achats, panier moyen, ancienneté, ratio promo, score fidélité, catégorie préférée, canal principal
- Injection de valeurs manquantes (~2%) pour simuler des données réelles
- Sauvegarde en CSV dans `data/clients_marketifri.csv`

**Exécution :**
```bash
python generate_dataset.py
```

**Sortie :**
- Fichier CSV : `data/clients_marketifri.csv` (156.4 Ko)
- 2000 lignes × 12 colonnes
- Affichage des statistiques descriptives

---

### 2. kmeans_clustering.py

Implémente l'algorithme K-means avec analyse complète.

**Pipeline :**
1. Chargement et prétraitement des données
2. Détermination du k optimal (méthode Elbow + Silhouette)
3. Entraînement du modèle K-means final (k=5)
4. Évaluation avec métriques internes et externes
5. Visualisations : PCA, Elbow, Silhouette, Heatmap, Boxplots
6. Export des résultats

**Paramètres du modèle :**
- `n_clusters = 5` (choisi par expertise métier)
- `init = "k-means++"` (initialisation intelligente)
- `n_init = 20` (20 initialisations, meilleure retenue)
- `max_iter = 500` (itérations max par run)
- `random_state = 42` (reproductibilité)

**Exécution :**
```bash
python kmeans_clustering.py
```

**Sorties :**
- `data/resultats_kmeans.csv` : Dataset avec labels de clusters
- `figures/kmeans_elbow_silhouette.png` : Courbe Elbow et Silhouette
- `figures/kmeans_pca_clusters.png` : Projection PCA 2D
- `figures/kmeans_heatmap_profils.png` : Heatmap des profils moyens
- `figures/kmeans_boxplots.png` : Boxplots des variables clés

---

### 3. hclust_clustering.py

Implémente le Clustering Hiérarchique Agglomératif (CAH).

**Pipeline :**
1. Chargement et prétraitement identique à K-means
2. Sous-échantillonnage (500 obs) pour le dendrogramme
3. Construction de la matrice de liens (linkage Ward)
4. Calcul du coefficient cophénétique
5. Analyse des sauts de fusion pour déterminer k
6. Tracé du dendrogramme
7. Clustering sur la dataset complète
8. Évaluation et visualisations
9. Comparaison formelle avec K-means

**Paramètres du modèle :**
- `n_clusters = 5`
- `linkage = "ward"` (minimise la variance intra-cluster)
- `metric = "euclidean"`

**Exécution :**
```bash
python hclust_clustering.py
```

**Sorties :**
- `data/resultats_hclust.csv` : Dataset avec labels de clusters
- `figures/hclust_dendrogramme.png` : Dendrogramme tronqué
- `figures/hclust_sauts_fusion.png` : Analyse des sauts de fusion
- `figures/hclust_pca_clusters.png` : Projection PCA 2D
- `figures/hclust_heatmap_profils.png` : Heatmap des profils moyens

---

### 4. main_clustering.py

Point d'entrée unique qui orchestre l'ensemble du pipeline.

**Fonctionnalités :**
- Exécute séquentiellement tous les modules
- Mesure le temps d'exécution de chaque méthode
- Génère des visualisations comparatives
- Affiche un rapport de synthèse terminal

**Exécution :**
```bash
python main_clustering.py
```

**Sorties supplémentaires :**
- `figures/comparaison_kmeans_hclust.png` : PCA côte-à-côte
- `figures/comparaison_metriques.png` : Barplot comparatif des métriques
- Rapport terminal avec tableau comparatif complet

---

## 📊 Dataset

### Source

Dataset synthétique générée par `generate_dataset.py`, simulant des données réelles de segmentation clients pour une enseigne de grande distribution.

### Caractéristiques

- **Nombre d'observations** : 2 000 clients
- **Nombre de variables** : 12 (dont 10 utilisées pour le clustering)
- **Taille du fichier** : 156.4 Ko
- **Valeurs manquantes** : ~2 % sur 4 colonnes numériques

### Variables

| Variable | Type | Description | Plage |
|----------|------|-------------|-------|
| `client_id` | String | Identifiant unique du client | C00001–C02000 |
| `age` | Integer | Âge du client | 18–75 ans |
| `revenu_annuel` | Float | Revenu annuel estimé | 10 000–150 000 € |
| `score_depenses` | Integer | Propension à dépenser | 0–100 |
| `nb_achats_annuel` | Integer | Nombre d'achats dans l'année | 1–120 |
| `panier_moyen` | Float | Valeur moyenne d'un achat | 5–500 € |
| `anciennete_mois` | Integer | Durée de la relation client (mois) | 1–120 |
| `ratio_promo` | Float | Part des achats en promotion | 0–1 |
| `score_fidelite` | Integer | Score de fidélité composite | 0–100 |
| `categorie_preferee` | String | Catégorie d'achat dominante | 9 catégories |
| `canal_principal` | String | Canal d'achat préférentiel | Magasin / En ligne / App |
| `profil_reel` | String | Label de vérité terrain | 5 profils |

### Profils simulés (vérité terrain)

1. **Familles Actives (25%)** : Âge moyen, revenu moyen, gros paniers alimentaires
2. **Jeunes Économes (20%)** : Jeunes adultes, faible revenu, achats ciblés
3. **Seniors Fidèles (20%)** : Seniors, achat régulier, fidélité très élevée
4. **Chasseurs de Promo (20%)** : Motivés par les promotions, fort ratio promo
5. **Aisés Premium (15%)** : Revenu élevé, paniers très chers

---

## Résultats et visualisations

### Fichiers générés

Toutes les visualisations sont sauvegardées dans le répertoire `figures/` :

#### K-means
- `kmeans_elbow_silhouette.png` : Détermination du k optimal
- `kmeans_pca_clusters.png` : Projection 2D des clusters
- `kmeans_heatmap_profils.png` : Heatmap des profils moyens
- `kmeans_boxplots.png` : Distribution des variables par cluster

#### Clustering Hiérarchique
- `hclust_dendrogramme.png` : Dendrogramme avec coupure
- `hclust_sauts_fusion.png` : Analyse des sauts pour choix du k
- `hclust_pca_clusters.png` : Projection 2D des clusters
- `hclust_heatmap_profils.png` : Heatmap des profils moyens

#### Comparaison
- `comparaison_kmeans_hclust.png` : PCA côte-à-côte
- `comparaison_metriques.png` : Barplot des métriques

### Données exportées

- `data/resultats_kmeans.csv` : Dataset originale + colonne `cluster_kmeans`
- `data/resultats_hclust.csv` : Dataset originale + colonne `cluster_hclust`

---

## Méthodologie

### Prétraitement des données

1. **Imputation des valeurs manquantes** : Médiane pour les variables numériques
2. **Encodage des variables catégorielles** : LabelEncoder pour `categorie_preferee` et `canal_principal`
3. **Standardisation** : Z-score via StandardScaler (μ=0, σ=1)

### Sélection du nombre de clusters

#### K-means
- **Méthode Elbow** : Analyse de la décroissance de l'inertie
- **Silhouette Score** : Maximisation du score de cohésion/séparation
- **Choix final** : k=5 (cohérent avec les 5 profils métier)

#### Clustering Hiérarchique
- **Dendrogramme** : Coupure horizontale au niveau des sauts majeurs
- **Méthode de l'accélération** : Analyse de la 2ème différence des hauteurs
- **Coefficient cophénétique** : Évaluation de la qualité de représentation (c=0.69)

### Métriques d'évaluation

#### Métriques internes (sans labels réels)
- **Silhouette Score** : Cohésion et séparation des clusters [-1, 1], ↑ mieux
- **Calinski-Harabász Index** : Rapport variance inter/intra, ↑ mieux
- **Davies-Bouldin Index** : Similarité inter-clusters, ↓ mieux

#### Métrique externe (avec labels réels)
- **Adjusted Rand Index (ARI)** : Accord avec la vérité terrain [-1, 1], 1=parfait

---

## Interprétation des résultats

### Profils identifiés par K-means

| Cluster | Profil | Âge | Revenu | Panier | Fidélité | Ratio promo |
|---------|--------|-----|--------|--------|----------|-------------|
| 0 | Chasseurs de promotions | 36 | 25 187 € | 38,8 € | 46 | 0.70 |
| 1 | Aisés premium | 43 | 85 544 € | 201,2 € | 72 | 0.08 |
| 2 | Seniors fidèles | 62 | 35 517 € | 55,0 € | 84 | 0.20 |
| 3 | Familles actives | 38 | 41 490 € | 83,4 € | 62 | 0.30 |
| 4 | Jeunes économes | 24 | 18 350 € | 23,8 € | 29 | 0.48 |

### Performances K-means

- **Silhouette Score** : 0.3002 (acceptable pour 10D)
- **Calinski-Harabász** : 903.44 (bon rapport variance)
- **Davies-Bouldin** : 1.3248 (similarité modérée)
- **ARI vs vérité** : **0.9204** (excellent accord)

### Performances Hclust

- **Silhouette Score** : 0.2928 (légèrement inférieur à K-means)
- **Calinski-Harabász** : 884.81 (bon)
- **Davies-Bouldin** : 1.3511 (comparable)
- **ARI vs vérité** : **0.9498** (excellent, légèrement supérieur à K-means)

### Recommandations métier

#### Aisés Premium (≈15%)
- Programme de fidélité premium
- Communications sur qualité et prestige
- Éviter promotions massives

#### Familles Actives (≈25%)
- Offres multi-achats et économies sur volume
- Click-and-collect, livraison à domicile
- Bons de réduction progressifs

#### Seniors Fidèles (≈20%)
- Maintenir expérience en magasin de qualité
- Programme reconnaissance ancienneté
- Éviter communications exclusivement digitales

#### Jeunes Économes (≈20%)
- Promotions ciblées et offres de lancement
- Communication digitale (app, réseaux sociaux)
- Programmes de parrainage

#### Chasseurs de Promo (≈20%)
- Campagnes promotionnelles régulières
- Alertes promo personnalisées
- Cross-selling sur produits discount

---

## Comparaison K-means vs Hclust

### Comparaison quantitative

| Métrique | K-Means | Hclust Ward | Gagnant |
|----------|---------|-------------|---------|
| Silhouette Score (↑) | **0.3002** | 0.2928 | K-Means |
| Calinski-Harabász (↑) | **903.44** | 884.81 | K-Means |
| Davies-Bouldin (↓) | **1.3248** | 1.3511 | K-Means |
| ARI vs vérité (↑) | 0.9204 | **0.9498** | Hclust |
| Accord inter-méthodes (ARI) | — | 0.9213 | — |

### Comparaison qualitative

| Dimension | K-Means | Hclust |
|-----------|---------|--------|
| Spécification de k | Requis a priori | Choix a posteriori |
| Type de résultat | Partition plate | Hiérarchie (dendrogramme) |
| Reproductibilité | Semi-déterministe | Déterministe |
| Scalabilité | O(n·k·d·t) - Excellent | O(n² log n) - Limitée |
| Temps calcul (n=2000) | ~2.8 s | ~1.9 s |
| Interprétabilité | Centroïdes directs | Dendrogramme visuel |
| Multi-granularité | Non | Oui |

### Quand utiliser K-means ?

- Grand volume de données (n > 10 000)
- Hypothèse a priori sur k
- Clusters sphériques attendus
- Rapidité critique (production)
- Centroïdes interprétables nécessaires

### Quand utiliser Hclust ?

- Exploration sans fixer k a priori
- Volume modéré (n < 10 000)
- Besoin de visualisation hiérarchique
- Structure hiérarchique naturelle
- Reproductibilité totale requise

### Conclusion

Les deux méthodes donnent des résultats très concordants (ARI inter-méthodes = 0.92), validant la robustesse de la structure en 5 clusters. Pour ce projet, une stratégie hybride est recommandée : explorer avec Hclust pour choisir k, puis affiner avec K-means.

---

## Références

- Arthur, D., & Vassilvitskii, S. (2007). K-means++: The advantages of careful seeding
- MacQueen, J. (1967). Some methods for classification and analysis of multivariate observations
- Ward, J. H. (1963). Hierarchical grouping to optimize an objective function
- Murtagh, F., & Legendre, P. (2014). Ward's hierarchical agglomerative clustering method
- Tibshirani, R., & Walther, G. (2001). Estimating the number of clusters in a data set via the gap statistic

---

## Auteur

**Étudiant :** Brice GOUDALO
**Formation :** Master 1 Génie Logiciel - IFRI  
**Module :** Big Data
**Année :** 2025–2026

---

## Dépannage

### Erreur : ModuleNotFoundError

**Solution :** Installer les dépendances manquantes
```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy
```

### Erreur : FileNotFoundError pour le dataset

**Solution :** Exécuter d'abord `generate_dataset.py`
```bash
python generate_dataset.py
```

**Note :** Les fichiers de sortie sont générés dans les répertoires `data/` et `figures/` du projet

### Problème : Chemins relatifs incorrects

**Solution :** Exécutez les scripts depuis le répertoire racine du projet
```bash
cd /tp2_Brice_GOUDALO
python main_clustering.py
```
