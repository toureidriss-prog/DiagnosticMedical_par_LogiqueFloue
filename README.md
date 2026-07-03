# 🩺 Système Intelligent de Diagnostic Cardiaque Basé sur la Logique Floue

## Présentation

Ce projet est une application de diagnostic médical développée en Python utilisant la **logique floue** pour estimer le niveau de gravité d'une maladie cardiaque à partir de plusieurs paramètres cliniques.

Contrairement aux approches classiques basées sur l'apprentissage automatique, ce système repose sur une **base de connaissances composée de règles floues** automatiquement générées puis sélectionnées selon leur pertinence à l'aide des métriques **Support**, **Confiance** et **Lift**.

L'application offre également une interface interactive développée avec **Streamlit**, permettant de réaliser un diagnostic en temps réel et d'expliquer la règle ayant conduit à la décision.

---

# Fonctionnalités

* Génération automatique de toutes les règles floues candidates.
* Calcul des métriques :

  * Support flou
  * Confiance floue
  * Lift flou
* Sélection automatique des règles les plus pertinentes.
* Construction d'un système d'inférence floue de type Mamdani.
* Diagnostic interactif via une interface Streamlit.
* Explicabilité de la décision grâce à l'affichage de la règle activée.
* Détection des situations où aucune règle de la base de connaissances n'est activée.

---

# Variables utilisées

Le système utilise les variables médicales suivantes :

| Variable    | Description                                       |
| ----------- | ------------------------------------------------- |
| **thalach** | Fréquence cardiaque maximale atteinte             |
| **oldpeak** | Dépression du segment ST induite par l'effort     |
| **ca**      | Nombre de gros vaisseaux colorés par fluoroscopie |
| **thal**    | Résultat du test Thal                             |
| **num**     | Niveau de gravité de la maladie cardiaque         |

---

# Fonctions d'appartenance

Les variables sont représentées par des ensembles flous.

### thalach

* faible
* moyenne
* élevée

### oldpeak

* léger
* risque
* sévère

### ca

* zéro
* un
* deux
* trois

### thal

* normal
* fixe
* réversible

### Diagnostic (num)

* sain
* faible
* modéré
* sévère
* très sévère

---

# Génération des règles

Toutes les combinaisons possibles des termes linguistiques sont générées automatiquement.

Nombre total de règles candidates :

540

Chaque règle est ensuite évaluée selon :

* Support
* Confiance
* Lift

Une règle est conservée uniquement si :

* Support ≥ 0.003
* Confiance ≥ 0.60
* Lift > 1

Les règles retenues sont enregistrées dans le fichier :

```text
rules.csv
```

---

# Structure du projet

```text
.
├── app.py
├── fuzzy_model.py
├── generate_rules.py
├── rules.csv
├── requirements.txt
└── README.md
```

---

# Description des fichiers

### generate_rules.py

Construit automatiquement la base de règles.

Il :

* charge le dataset,
* calcule les degrés d'appartenance,
* génère toutes les règles candidates,
* calcule Support, Confiance et Lift,
* sélectionne les règles pertinentes,
* enregistre les règles retenues dans `rules.csv`.

---

### fuzzy_model.py

Construit le système d'inférence floue.

Il contient notamment :

* la définition des ensembles flous,
* la reconstruction du système à partir de `rules.csv`,
* la fonction de prédiction,
* la fonction d'explicabilité.

---

### app.py

Application développée avec Streamlit.

Elle permet :

* d'afficher les règles retenues,
* de réaliser un diagnostic,
* d'expliquer la décision du système.

---

# Installation

Créer un environnement virtuel (optionnel mais recommandé) :

```bash
python -m venv venv
```

Activation :

Windows

```bash
venv\Scripts\activate
```

Linux / macOS

```bash
source venv/bin/activate
```

Installer les dépendances :

```bash
pip install -r requirements.txt
```

---

# Génération des règles

Exécuter une seule fois :

```bash
python generate_rules.py
```

Le fichier `rules.csv` sera créé automatiquement.

---

# Lancer l'application

```bash
streamlit run app.py
```

Le navigateur ouvrira automatiquement l'interface du système.

---

# Interface

L'application comporte trois sections :

## Dashboard

* nombre de règles retenues ;
* aperçu de la base de connaissances.

## Diagnostic

L'utilisateur saisit :

* thalach ;
* oldpeak ;
* ca ;
* thal.

Le système affiche :

* la classe diagnostiquée ;
* la valeur défuzzifiée.

Si aucune règle n'est activée, un message informe que le système ne peut pas établir de diagnostic à partir de la base de connaissances actuelle.

## Explicabilité

Cette section affiche :

* la règle ayant conduit au diagnostic ;
* son degré d'activation.

Si aucune règle n'est activée, un message indique qu'aucune explication ne peut être fournie.

---

# Technologies utilisées

* Python
* Streamlit
* NumPy
* Pandas
* Scikit-Fuzzy
* scikit-learn
* UCI Machine Learning Repository

---

# Jeu de données

Le projet utilise le **Heart Disease Dataset** provenant de l'UCI Machine Learning Repository.

---

# Auteur

**Idrissa Oumar Touré**

Projet réalisé dans le cadre d'un mémoire de Master portant sur le diagnostic médical par logique floue.
