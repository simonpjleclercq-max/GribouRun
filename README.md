# 🎨 Gribou Run

> Dessine une forme, l'appli la transforme en itinéraire de course sur les rues de Paris.

📄 [Pitch deck (PDF)](../docs/gribou-run-pitch.pdf)

<!-- Ajoute ici une capture d'écran de l'appli et un exemple de tracé : ![Exemple](docs/exemple.png) -->

## Le problème

Courir sur les mêmes parcours finit par lasser. Gribou Run transforme la sortie en jeu : le tracé de la course *dessine* quelque chose (un chien, un cœur, ce que tu veux) sur la carte, comme du « GPS art ».

## Comment ça marche

```
Image (dessin) ─► Contour ─► Points de contrôle ─► Accrochage au réseau routier ─► Itinéraire ─► GPX
```

1. **Auto-calibrage** : seuil de binarisation par la méthode d'**Otsu**, taille de travail et nombre de points calculés automatiquement à partir de l'image
2. **Extraction du contour** principal de la forme (OpenCV) puis **simplification Douglas-Peucker** (le curseur « Lissage » règle le niveau de détail)
3. **Échantillonnage** du contour en points de contrôle, projetés sur la carte de Paris
4. **Accrochage** de chaque point au nœud le plus proche du **graphe routier OpenStreetMap** (OSMnx)
5. **Routage** : plus court chemin (NetworkX) entre points successifs
6. **Nettoyage** : suppression des boucles redondantes (détours évitables)
7. **Sorties** :
   - l'image du résultat (PNG)
   - les **instructions de navigation** pas à pas (« tourner à droite »…), calculées à partir du cap (*bearing*) entre segments
   - la trace **GPX**, importable dans Strava ou une montre GPS

L'interface affiche aussi des indicateurs : nombre de nœuds du circuit, réduction obtenue par le nettoyage, nombre de points de contrôle et échecs de routage.

## Lancer

```bash
pip install -r requirements.txt
python -m streamlit run app.py
```

Dépose une image (trait sombre sur fond clair, forme simple) et clique sur **Générer l'itinéraire**.

⏱️ Le premier lancement télécharge le graphe routier de Paris : compte une à deux minutes. Il est ensuite mis en cache.

Un exemple de trace générée est fourni dans [`examples/gribourun_route.gpx`](examples/gribourun_route.gpx).

## Structure

```
gribou-run/
├── app.py                      # Application Streamlit complète (version finale)
├── requirements.txt
├── examples/
│   └── gribourun_route.gpx     # Exemple d'itinéraire généré
└── prototypes/                 # Étapes précédentes du projet
    ├── GribouRun.py            # Premier script (paramètres à la main)
    ├── GribouRun_Opti.py       # + auto-calibrage des paramètres
    └── gribourun_appnogpx.py   # Première version Streamlit, sans export GPX
```

## Limites connues

- Paris uniquement
- Le graphe utilisé est celui des voies **carrossables** (`network_type="drive"`) : un réseau piéton serait plus adapté à la course
- Les formes simples à contour unique donnent les meilleurs résultats
- Le « Mystery Shape Challenge » (courir une forme secrète et deviner l'animal) est décrit dans le pitch mais **pas encore implémenté**

## Pistes d'amélioration

- Réseau piéton / sentiers, et autres villes
- Contrôle de la distance totale et de la difficulté
- Mystery Shape Challenge et défis entre amis
