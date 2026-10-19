import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from PIL import Image
import osmnx as ox
import networkx as nx
import cv2
from scipy.ndimage import binary_fill_holes
import warnings
warnings.filterwarnings("ignore")

# =========================
# CONFIGURATION
# =========================
IMAGE_PATH        = "dog.jpg"
IMG_SIZE          = 500        # Résolution interne
BINARY_THRESHOLD  = 200        # Seuil : baisse si traits manquants
N_POINTS          = 300        # Nb de points échantillonnés sur le contour permet d'affiner le tracé 
                               # (augmente pour plus de fidélité)

# =========================
# 1. CHARGER L'IMAGE ET EXTRAIRE LE CONTOUR ORDONNÉ
# =========================
print("━━━ Étape 1 : Extraction du contour ordonné ━━━")

img = Image.open(IMAGE_PATH).convert("L")
img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
img_arr = np.array(img)

# Binarisation : trait noir sur fond blanc → on inverse
binary = (img_arr < BINARY_THRESHOLD).astype(np.uint8) * 255

# ⚠️ cv2.findContours attend uint8, trait blanc sur fond noir
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

if not contours:
    raise ValueError("❌ Aucun contour trouvé ! Ajuste BINARY_THRESHOLD.")

# Prendre le contour le plus long (= le trait principal)
contour = max(contours, key=len)
print(f"  Contour trouvé : {len(contour)} points")

# Extraire et normaliser les coordonnées [0, 1]
pts = contour.squeeze()   # shape (N, 2) : colonnes = (x, y) en pixels image
xs_norm = pts[:, 0] / (IMG_SIZE - 1)
ys_norm = 1 - pts[:, 1] / (IMG_SIZE - 1)   # flip vertical (image vs carte)

# Sous-échantillonner N_POINTS régulièrement sur le contour
indices   = np.round(np.linspace(0, len(xs_norm) - 1, N_POINTS)).astype(int)
xs_sample = xs_norm[indices]
ys_sample = ys_norm[indices]
print(f"  Points échantillonnés : {len(xs_sample)}")

# =========================
# 2. CHARGER LE GRAPHE OSM DE PARIS
# =========================
print("━━━ Étape 2 : Téléchargement des rues de Paris ━━━")
print("  (peut prendre 1-2 minutes...)")

G = ox.graph_from_place("Paris, France", network_type="drive")
nodes_gdf, edges_gdf = ox.graph_to_gdfs(G)

# Bounds en WGS84
minlon = nodes_gdf.geometry.x.min()
maxlon = nodes_gdf.geometry.x.max()
minlat = nodes_gdf.geometry.y.min()
maxlat = nodes_gdf.geometry.y.max()

print(f"  Nœuds : {G.number_of_nodes()}, Arêtes : {G.number_of_edges()}")
print(f"  Bounds : lon [{minlon:.4f}, {maxlon:.4f}], lat [{minlat:.4f}, {maxlat:.4f}]")

# =========================
# 3. PROJECTION IMAGE → LON/LAT
# =========================
print("━━━ Étape 3 : Projection image → WGS84 ━━━")

# Marge pour que le dessin ne colle pas aux bords de Paris
MARGIN = 0.05   # 5% de marge de chaque côté
lon_range = (maxlon - minlon) * (1 - 2 * MARGIN)
lat_range = (maxlat - minlat) * (1 - 2 * MARGIN)

lons_sample = minlon + MARGIN * (maxlon - minlon) + xs_sample * lon_range
lats_sample = minlat + MARGIN * (maxlat - minlat) + ys_sample * lat_range

print(f"  Premier point : lon={lons_sample[0]:.4f}, lat={lats_sample[0]:.4f}")

# =========================
# 4. SNAPPING : nœud OSM le plus proche pour chaque point
# =========================
print("━━━ Étape 4 : Snapping sur le réseau routier ━━━")

snapped_nodes = ox.distance.nearest_nodes(G, lons_sample, lats_sample)
print(f"  Nœuds snappés : {len(snapped_nodes)}, uniques : {len(set(snapped_nodes))}")

# =========================
# 5. ROUTING : relier les points consécutifs via le réseau réel
# =========================
print("━━━ Étape 5 : Calcul des itinéraires ━━━")

all_route_nodes = []
failed = 0

for i in range(len(snapped_nodes) - 1):
    u = snapped_nodes[i]
    v = snapped_nodes[i + 1]
    if u == v:
        continue
    try:
        path = nx.shortest_path(G, u, v, weight="length")
        if not all_route_nodes:
            all_route_nodes.extend(path)
        else:
            all_route_nodes.extend(path[1:])
    except nx.NetworkXNoPath:
        failed += 1
    if (i + 1) % 20 == 0:
        print(f"  Routing {i+1}/{len(snapped_nodes)-1} — échecs : {failed}")

# Fermer le circuit (retour au départ)
try:
    closing = nx.shortest_path(G, snapped_nodes[-1], snapped_nodes[0], weight="length")
    all_route_nodes.extend(closing[1:])
    print("  Circuit fermé ✅")
except nx.NetworkXNoPath:
    print("  ⚠️ Impossible de fermer le circuit")

print(f"  ✅ Total nœuds du tracé : {len(all_route_nodes)}, échecs : {failed}")

if len(all_route_nodes) < 2:
    raise RuntimeError("❌ Tracé vide ! Vérifie que le contour est bien détecté.")

# =========================
# 6. EXTRAIRE LES COORDONNÉES DU TRACÉ
# =========================
route_lons = [G.nodes[n]["x"] for n in all_route_nodes]
route_lats = [G.nodes[n]["y"] for n in all_route_nodes]

# =========================
# 7. AFFICHAGE
# =========================
print("━━━ Étape 6 : Affichage ━━━")

fig, axes = plt.subplots(1, 2, figsize=(20, 10), facecolor="#0d0d1a")

# ── Panneau gauche : dessin original + contour détecté ───────────────────────
ax0 = axes[0]
ax0.set_facecolor("#0d0d1a")
ax0.imshow(img_arr, cmap="gray", origin="upper", extent=[0, IMG_SIZE, IMG_SIZE, 0])

# Afficher le contour détecté (en rouge) pour vérification
ax0.plot(pts[indices, 0], pts[indices, 1],
         "r-", linewidth=1.5, alpha=0.7, label="Contour détecté")
ax0.scatter(pts[indices, 0], pts[indices, 1],
            c=np.arange(len(indices)), cmap="plasma", s=20, zorder=5)
ax0.set_title("Dessin + contour ordonné", color="white", fontsize=13, fontweight="bold")
ax0.axis("off")

# ── Panneau droit : carte de Paris ───────────────────────────────────────────
ax1 = axes[1]
ax1.set_facecolor("#0d0d1a")

# Fond : toutes les rues de Paris
edges_gdf.plot(ax=ax1, color="#2a2a4a", linewidth=0.25, alpha=0.7)

# Tracé coloré avec dégradé plasma
points_arr = np.array([route_lons, route_lats]).T.reshape(-1, 1, 2)
segments   = np.concatenate([points_arr[:-1], points_arr[1:]], axis=1)
lc = LineCollection(segments, cmap="plasma",
                    norm=plt.Normalize(0, len(segments)),
                    linewidth=2.0, alpha=0.95, capstyle="round")
lc.set_array(np.arange(len(segments)))
ax1.add_collection(lc)

# Points départ / arrivée
ax1.scatter([route_lons[0]],  [route_lats[0]],
            c="#00ff88", s=100, zorder=6, label="Départ")
ax1.scatter([route_lons[-1]], [route_lats[-1]],
            c="#ff4444", s=100, zorder=6, label="Arrivée")

# Vue centrée sur Paris entier
pad_lon = (maxlon - minlon) * 0.02
pad_lat = (maxlat - minlat) * 0.02
ax1.set_xlim(minlon - pad_lon, maxlon + pad_lon)
ax1.set_ylim(minlat - pad_lat, maxlat + pad_lat)

ax1.set_title(
    f"Circuit sur les rues de Paris  ({len(all_route_nodes)} nœuds)",
    color="white", fontsize=13, fontweight="bold"
)
ax1.axis("off")

# Légende
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color="#2a2a4a",         lw=1.5, label="Rues de Paris"),
    Line2D([0], [0], color=plt.cm.plasma(0.1),lw=2,   label="Début du tracé"),
    Line2D([0], [0], color=plt.cm.plasma(0.9),lw=2,   label="Fin du tracé"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#00ff88",
           markersize=8, label="Départ"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#ff4444",
           markersize=8, label="Arrivée"),
]
ax1.legend(handles=legend_elements, loc="lower left",
           facecolor="#1a1a3a", edgecolor="#444466",
           labelcolor="white", fontsize=9)

plt.suptitle("GribouRun — Coeur à Paris",
             color="white", fontsize=18, fontweight="bold", y=1.01)
plt.tight_layout()

plt.savefig("paris_coeur_euler.png", dpi=150,
            bbox_inches="tight", facecolor=fig.get_facecolor())
print("✅ Image sauvegardée : paris_coeur_euler.png")
plt.show()
