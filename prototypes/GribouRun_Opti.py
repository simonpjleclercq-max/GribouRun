import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.collections import LineCollection
from PIL import Image
import osmnx as ox
import networkx as nx
import cv2
import warnings
warnings.filterwarnings("ignore")

# =============================================================
#  SEUL PARAMÈTRE À TOUCHER : le nom de ton image
# =============================================================
IMAGE_PATH = "dog.jpg"


# ╔═══════════════════════════════════════════════════════════╗
# ║         AUTO-CALIBRATION DES PARAMÈTRES                  ║
# ╚═══════════════════════════════════════════════════════════╝

def auto_calibrate(image_path):
    """
    Calcule automatiquement IMG_SIZE, BINARY_THRESHOLD et N_POINTS
    à partir de l'image seule.
    """
    print("━━━ AUTO-CALIBRATION ━━━")

    # Chargement
    raw = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if raw is None:
        raise FileNotFoundError(f"Image introuvable : {image_path}")
    h0, w0 = raw.shape
    print(f"  Image originale : {w0}×{h0} px")

    # ═══ A. BINARY_THRESHOLD via Otsu ═══
    blur = cv2.GaussianBlur(raw, (5, 5), 0)
    otsu_thresh, _ = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    print(f"  [Otsu] BINARY_THRESHOLD = {int(otsu_thresh)}")

    # ═══ B. IMG_SIZE adaptatif ═══
    TARGET_MIN, TARGET_MAX = 800, 3000
    lo, hi = 200, 1000
    best_size = 400
    best_n    = 0

    for _ in range(10):
        mid = (lo + hi) // 2
        test = cv2.resize(raw, (mid, mid), interpolation=cv2.INTER_AREA)
        _, bw = cv2.threshold(
            cv2.GaussianBlur(test, (5, 5), 0),
            otsu_thresh, 255, cv2.THRESH_BINARY_INV
        )
        cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        n = len(max(cnts, key=len)) if cnts else 0
        if n < TARGET_MIN:
            lo = mid
        elif n > TARGET_MAX:
            hi = mid
        else:
            best_size, best_n = mid, n
            break
        best_size, best_n = mid, n

    best_size = max(256, min(800, best_size))
    print(f"  [Dichotomie] IMG_SIZE = {best_size}  (contour brut ≈ {best_n} pts)")

    # ═══ C. N_POINTS géométrique ═══
    PARIS_WIDTH_KM    = 12.0
    PARIS_HEIGHT_KM   =  9.0
    STREET_SPACING_KM =  0.18

    img_cal = cv2.resize(raw, (best_size, best_size), interpolation=cv2.INTER_AREA)
    _, bw_cal = cv2.threshold(
        cv2.GaussianBlur(img_cal, (5, 5), 0),
        otsu_thresh, 255, cv2.THRESH_BINARY_INV
    )
    cnts_cal, _ = cv2.findContours(bw_cal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour_cal  = max(cnts_cal, key=len)

    pts_cal  = contour_cal.squeeze().astype(float)
    diffs    = np.diff(pts_cal, axis=0)
    arc_px   = float(np.sum(np.linalg.norm(diffs, axis=1)))

    px_per_km   = best_size / max(PARIS_WIDTH_KM, PARIS_HEIGHT_KM)
    arc_km      = arc_px / px_per_km
    n_points_geo = int(arc_km / STREET_SPACING_KM)
    n_points_geo = max(60, min(300, n_points_geo))

    print(f"  [Géométrie] Arc contour ≈ {arc_km:.1f} km  →  N_POINTS = {n_points_geo}")

    params = {
        "IMG_SIZE"         : best_size,
        "BINARY_THRESHOLD" : int(otsu_thresh),
        "N_POINTS"         : n_points_geo,
    }

    rapport = (
        f"╔══════════════════════════════════════╗\n"
        f"║       PARAMÈTRES AUTO-CALIBRÉS       ║\n"
        f"╠══════════════════════════════════════╣\n"
        f"║  IMG_SIZE          = {best_size:<6}          ║\n"
        f"║  BINARY_THRESHOLD  = {int(otsu_thresh):<6}          ║\n"
        f"║  N_POINTS          = {n_points_geo:<6}          ║\n"
        f"╚══════════════════════════════════════╝"
    )
    print(rapport)
    return params, rapport


# ╔═══════════════════════════════════════════════════════════╗
# ║         A) LISSAGE DU CONTOUR (Douglas-Peucker)          ║
# ╚═══════════════════════════════════════════════════════════╝

def smooth_contour(contour_raw, epsilon_ratio=0.008):
    """
    Simplifie le contour avec cv2.approxPolyDP (Douglas-Peucker).
    
    epsilon_ratio : % de la longueur du contour
                    0.005 = très fidèle (beaucoup de points)
                    0.01  = bon équilibre
                    0.02  = simplifié agressif
    """
    arc_length = cv2.arcLength(contour_raw, closed=True)
    epsilon    = epsilon_ratio * arc_length
    smoothed   = cv2.approxPolyDP(contour_raw, epsilon, closed=True)
    print(f"  [Lissage] {len(contour_raw)} pts → {len(smoothed)} pts (ε={epsilon:.1f})")
    return smoothed


# ╔═══════════════════════════════════════════════════════════╗
# ║    B) SUPPRESSION DES BOUCLES REDONDANTES (post-routing) ║
# ╚═══════════════════════════════════════════════════════════╝

def remove_redundant_loops(route_nodes, G, threshold_ratio=2.5, min_saving=0.25):
    """
    Parcourt le tracé et supprime les boucles où on fait un grand
    détour pour revenir proche du point de départ.
    
    threshold_ratio : si route_length > ratio × direct_distance → suspect
    min_saving      : gain minimum requis (ex: 0.25 = 25% plus court)
    """
    print("  [Post-filtrage] Suppression des boucles...")
    
    cleaned = [route_nodes[0]]
    i = 0
    removed_segments = 0
    
    while i < len(route_nodes) - 1:
        best_skip   = 0
        best_path   = None
        best_saving = 0
        
        # Regarder jusqu'à 12 nœuds en avant
        for lookahead in range(2, min(13, len(route_nodes) - i)):
            current = route_nodes[i]
            target  = route_nodes[i + lookahead]
            
            # Distance géographique directe (en mètres)
            try:
                direct_dist = ox.distance.great_circle(
                    G.nodes[current]['y'], G.nodes[current]['x'],
                    G.nodes[target]['y'],  G.nodes[target]['x']
                )
            except:
                continue
            
            # Longueur du chemin routé actuel
            routed_length = 0
            try:
                for j in range(lookahead):
                    u = route_nodes[i + j]
                    v = route_nodes[i + j + 1]
                    routed_length += G[u][v][0]['length']
            except:
                continue
            
            # Détection de boucle : détour > threshold_ratio × distance directe
            if routed_length > threshold_ratio * direct_dist and direct_dist < 400:
                # Essayer un chemin direct
                try:
                    direct_path   = nx.shortest_path(G, current, target, weight='length')
                    direct_length = nx.shortest_path_length(G, current, target, weight='length')
                    
                    # Gain suffisant ?
                    if direct_length < routed_length * (1 - min_saving):
                        saving = routed_length - direct_length
                        if saving > best_saving:
                            best_skip   = lookahead
                            best_path   = direct_path
                            best_saving = saving
                except:
                    pass
        
        # Appliquer le meilleur raccourci trouvé
        if best_skip > 0 and best_path:
            cleaned.extend(best_path[1:])   # Ajouter le chemin direct (sans doublon du 1er nœud)
            i += best_skip
            removed_segments += 1
        else:
            cleaned.append(route_nodes[i + 1])
            i += 1
    
    print(f"  [Post-filtrage] {removed_segments} boucles supprimées  "
          f"({len(route_nodes)} → {len(cleaned)} nœuds)")
    return cleaned


# ╔═══════════════════════════════════════════════════════════╗
# ║              PIPELINE PRINCIPAL                          ║
# ╚═══════════════════════════════════════════════════════════╝

# 1. AUTO-CALIBRATION
params, rapport_calibration = auto_calibrate(IMAGE_PATH)
IMG_SIZE          = params["IMG_SIZE"]
BINARY_THRESHOLD  = params["BINARY_THRESHOLD"]
N_POINTS          = params["N_POINTS"]

# 2. CHARGEMENT ET CONTOUR
print("\n━━━ Étape 2 : Extraction et lissage du contour ━━━")
img_pil  = Image.open(IMAGE_PATH).convert("L")
img_pil  = img_pil.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
img_arr  = np.array(img_pil)

binary   = (img_arr < BINARY_THRESHOLD).astype(np.uint8) * 255
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
if not contours:
    raise ValueError("Aucun contour détecté — vérifie l'image.")

contour_raw  = max(contours, key=len)
print(f"  Contour brut : {len(contour_raw)} points")

# ═══ A) LISSAGE ═══
contour_smooth = smooth_contour(contour_raw, epsilon_ratio=0.008)
pts = contour_smooth.squeeze()

# Normalisation [0,1]
xs_norm  = pts[:, 0] / (IMG_SIZE - 1)
ys_norm  = 1 - pts[:, 1] / (IMG_SIZE - 1)

# Sous-échantillonnage
indices   = np.round(np.linspace(0, len(xs_norm) - 1, N_POINTS)).astype(int)
xs_sample = xs_norm[indices]
ys_sample = ys_norm[indices]

# 3. GRAPHE OSM
print("\n━━━ Étape 3 : Téléchargement des rues de Paris ━━━")
G = ox.graph_from_place("Paris, France", network_type="drive")
nodes_gdf, edges_gdf = ox.graph_to_gdfs(G)
minlon = nodes_gdf.geometry.x.min()
maxlon = nodes_gdf.geometry.x.max()
minlat = nodes_gdf.geometry.y.min()
maxlat = nodes_gdf.geometry.y.max()
print(f"  Nœuds : {G.number_of_nodes()}, Arêtes : {G.number_of_edges()}")

# 4. PROJECTION IMAGE → LON/LAT
MARGIN      = 0.05
lon_range   = (maxlon - minlon) * (1 - 2 * MARGIN)
lat_range   = (maxlat - minlat) * (1 - 2 * MARGIN)
lons_sample = minlon + MARGIN * (maxlon - minlon) + xs_sample * lon_range
lats_sample = minlat + MARGIN * (maxlat - minlat) + ys_sample * lat_range

# 5. SNAPPING
print("\n━━━ Étape 5 : Snapping ━━━")
snapped_nodes = ox.distance.nearest_nodes(G, lons_sample, lats_sample)
print(f"  Nœuds uniques : {len(set(snapped_nodes))}")

# 6. ROUTING INITIAL
print("\n━━━ Étape 6 : Routing initial ━━━")
all_route_nodes = []
failed = 0
for i in range(len(snapped_nodes) - 1):
    u, v = snapped_nodes[i], snapped_nodes[i + 1]
    if u == v:
        continue
    try:
        path = nx.shortest_path(G, u, v, weight="length")
        all_route_nodes.extend(path if not all_route_nodes else path[1:])
    except nx.NetworkXNoPath:
        failed += 1
    if (i + 1) % 20 == 0:
        print(f"  {i+1}/{len(snapped_nodes)-1} — échecs : {failed}")

# Fermeture du circuit
try:
    closing = nx.shortest_path(G, snapped_nodes[-1], snapped_nodes[0], weight="length")
    all_route_nodes.extend(closing[1:])
    print("  Circuit fermé ✅")
except nx.NetworkXNoPath:
    print("  ⚠️  Impossible de fermer le circuit")

initial_nodes = len(all_route_nodes)
print(f"  Total nœuds AVANT filtrage : {initial_nodes}, échecs : {failed}")

# ═══ B) SUPPRESSION DES BOUCLES ═══
print("\n━━━ Étape 7 : Post-filtrage des boucles ━━━")
all_route_nodes = remove_redundant_loops(all_route_nodes, G,
                                          threshold_ratio=2.5,
                                          min_saving=0.25)

print(f"  ✅ Total nœuds APRÈS filtrage : {len(all_route_nodes)} "
      f"({100*(initial_nodes - len(all_route_nodes))/initial_nodes:.1f}% réduit)")

route_lons = [G.nodes[n]["x"] for n in all_route_nodes]
route_lats = [G.nodes[n]["y"] for n in all_route_nodes]

# ╔═══════════════════════════════════════════════════════════╗
# ║                     AFFICHAGE                            ║
# ╚═══════════════════════════════════════════════════════════╝
print("\n━━━ Affichage ━━━")

BG      = "#0a0a14"
CARD_BG = "#12122a"

fig = plt.figure(figsize=(22, 11), facecolor=BG)
gs  = gridspec.GridSpec(
    2, 3,
    figure=fig,
    width_ratios=[1, 1.6, 0.38],
    height_ratios=[1, 1],
    hspace=0.05, wspace=0.06
)

# ── Panneau 1 : dessin original avec contour lissé ──────────
ax0 = fig.add_subplot(gs[:, 0])
ax0.set_facecolor(CARD_BG)
ax0.imshow(img_arr, cmap="gray", origin="upper",
           extent=[0, IMG_SIZE, IMG_SIZE, 0])

# Contour brut (gris)
pts_raw = contour_raw.squeeze()
ax0.plot(pts_raw[:, 0], pts_raw[:, 1],
         color="#555566", linewidth=0.8, alpha=0.4, label="Brut")

# Contour lissé (rose)
ax0.plot(pts[:, 0], pts[:, 1],
         color="#ff6b9d", linewidth=1.5, alpha=0.8, label="Lissé")

# Points échantillonnés (dégradé)
scatter_colors = np.linspace(0, 1, len(indices))
ax0.scatter(pts[indices, 0], pts[indices, 1],
            c=scatter_colors, cmap="plasma", s=20, zorder=5, alpha=0.9)

ax0.set_title("Contour lissé (Douglas-Peucker)", color="#aaaacc",
              fontsize=11, pad=8, fontfamily="monospace")
ax0.legend(loc="upper right", facecolor=CARD_BG, edgecolor="#444",
           labelcolor="white", fontsize=8)
ax0.axis("off")

# ── Panneau 2 : carte de Paris ───────────────────────────────
ax1 = fig.add_subplot(gs[:, 1])
ax1.set_facecolor(BG)
edges_gdf.plot(ax=ax1, color="#1e1e3a", linewidth=0.25, alpha=0.9)

points_arr = np.array([route_lons, route_lats]).T.reshape(-1, 1, 2)
segments   = np.concatenate([points_arr[:-1], points_arr[1:]], axis=1)
lc = LineCollection(segments, cmap="plasma",
                    norm=plt.Normalize(0, len(segments)),
                    linewidth=2.5, alpha=0.95, capstyle="round")
lc.set_array(np.arange(len(segments)))
ax1.add_collection(lc)

ax1.scatter([route_lons[0]],  [route_lats[0]],
            c="#00ffaa", s=100, zorder=7, edgecolors="white", linewidths=1)
ax1.scatter([route_lons[-1]], [route_lats[-1]],
            c="#ff4466", s=100, zorder=7, edgecolors="white", linewidths=1)

pad_lon = (maxlon - minlon) * 0.02
pad_lat = (maxlat - minlat) * 0.02
ax1.set_xlim(minlon - pad_lon, maxlon + pad_lon)
ax1.set_ylim(minlat - pad_lat, maxlat + pad_lat)
ax1.set_title(f"Circuit optimisé  ·  {len(all_route_nodes)} nœuds",
              color="#aaaacc", fontsize=11, pad=8, fontfamily="monospace")
ax1.axis("off")

# ── Panneau 3 : rapport ──────────────────────────────────────
ax2_top = fig.add_subplot(gs[0, 2])
ax2_bot = fig.add_subplot(gs[1, 2])
for ax in (ax2_top, ax2_bot):
    ax.set_facecolor(CARD_BG)
    ax.axis("off")

# Paramètres
param_lines = [
    ("BINARY_THRESHOLD", params["BINARY_THRESHOLD"], "Otsu auto"),
    ("IMG_SIZE",         params["IMG_SIZE"],         "Dichotomie"),
    ("N_POINTS",         params["N_POINTS"],         "Géométrique"),
]
y = 0.92
ax2_top.text(0.5, 0.99, "CALIBRATION", color="#ffffff", fontsize=9,
             fontweight="bold", ha="center", va="top",
             transform=ax2_top.transAxes, fontfamily="monospace")
ax2_top.plot([0.05, 0.95], [0.93, 0.93], color="#334",
             linewidth=0.8, transform=ax2_top.transAxes)

for name, val, expl in param_lines:
    ax2_top.text(0.08, y - 0.05, name, color="#00ffaa", fontsize=7.5,
                 fontweight="bold", transform=ax2_top.transAxes, fontfamily="monospace")
    ax2_top.text(0.08, y - 0.13, f"= {val}", color="#ffffff", fontsize=10,
                 fontweight="bold", transform=ax2_top.transAxes, fontfamily="monospace")
    ax2_top.text(0.08, y - 0.21, expl, color="#8888aa", fontsize=6.5,
                 transform=ax2_top.transAxes, fontfamily="monospace")
    ax2_top.plot([0.05, 0.95], [y - 0.25, y - 0.25], color="#223",
                 linewidth=0.5, transform=ax2_top.transAxes)
    y -= 0.28

# Stats
reduction_pct = 100 * (initial_nodes - len(all_route_nodes)) / initial_nodes
stats = [
    ("Nœuds (avant)",  initial_nodes),
    ("Nœuds (après)",  len(all_route_nodes)),
    ("Réduction",      f"{reduction_pct:.1f}%"),
    ("Échecs routing", failed),
]
ax2_bot.text(0.5, 0.99, "OPTIMISATION", color="#ffffff", fontsize=9,
             fontweight="bold", ha="center", va="top",
             transform=ax2_bot.transAxes, fontfamily="monospace")
ax2_bot.plot([0.05, 0.95], [0.93, 0.93], color="#334",
             linewidth=0.8, transform=ax2_bot.transAxes)
y2 = 0.82
for label, val in stats:
    ax2_bot.text(0.08, y2, label, color="#8888aa", fontsize=7,
                 transform=ax2_bot.transAxes, fontfamily="monospace")
    ax2_bot.text(0.75, y2, str(val), color="#00ffaa", fontsize=8.5,
                 fontweight="bold", ha="right",
                 transform=ax2_bot.transAxes, fontfamily="monospace")
    y2 -= 0.18

# Titre global
fig.text(0.5, 0.98, "GribouRun  ·  v6 Optimisé", color="#ffffff",
         fontsize=16, fontweight="bold", ha="center", va="top",
         fontfamily="monospace")

plt.savefig("paris_gribourun_v6.png", dpi=150,
            bbox_inches="tight", facecolor=BG)
print("✅  Sauvegardé : paris_gribourun_v6.png")
plt.show()
