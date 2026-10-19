#streamlit run gribourun_app_complete.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.collections import LineCollection
from PIL import Image
import osmnx as ox
import networkx as nx
import cv2
import io
import warnings
warnings.filterwarnings("ignore")

# Configuration de la page
st.set_page_config(
    page_title="GribouRun — Paris Street Art",
    page_icon="🎨",
    layout="wide"
)

# ╔═══════════════════════════════════════════════════════════╗
# ║                    FONCTIONS UTILITAIRES                 ║
# ╚═══════════════════════════════════════════════════════════╝

def auto_calibrate(img_array):
    """Auto-calibration des paramètres depuis une image numpy."""
    raw = img_array
    h0, w0 = raw.shape
    
    # A. BINARY_THRESHOLD via Otsu
    blur = cv2.GaussianBlur(raw, (5, 5), 0)
    otsu_thresh, _ = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    
    # B. IMG_SIZE adaptatif
    TARGET_MIN, TARGET_MAX = 800, 3000
    lo, hi = 200, 1000
    best_size = 400
    
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
            best_size = mid
            break
        best_size = mid
    
    best_size = max(256, min(800, best_size))
    
    # C. N_POINTS géométrique
    PARIS_WIDTH_KM = 12.0
    PARIS_HEIGHT_KM = 9.0
    STREET_SPACING_KM = 0.18
    
    img_cal = cv2.resize(raw, (best_size, best_size), interpolation=cv2.INTER_AREA)
    _, bw_cal = cv2.threshold(
        cv2.GaussianBlur(img_cal, (5, 5), 0),
        otsu_thresh, 255, cv2.THRESH_BINARY_INV
    )
    cnts_cal, _ = cv2.findContours(bw_cal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour_cal = max(cnts_cal, key=len)
    
    pts_cal = contour_cal.squeeze().astype(float)
    diffs = np.diff(pts_cal, axis=0)
    arc_px = float(np.sum(np.linalg.norm(diffs, axis=1)))
    
    px_per_km = best_size / max(PARIS_WIDTH_KM, PARIS_HEIGHT_KM)
    arc_km = arc_px / px_per_km
    n_points_geo = int(arc_km / STREET_SPACING_KM)
    n_points_geo = max(60, min(300, n_points_geo))
    
    return {
        "IMG_SIZE": best_size,
        "BINARY_THRESHOLD": int(otsu_thresh),
        "N_POINTS": n_points_geo,
    }


def smooth_contour(contour_raw, epsilon_ratio=0.008):
    """Lissage Douglas-Peucker."""
    arc_length = cv2.arcLength(contour_raw, closed=True)
    epsilon = epsilon_ratio * arc_length
    smoothed = cv2.approxPolyDP(contour_raw, epsilon, closed=True)
    return smoothed


def remove_redundant_loops(route_nodes, G, threshold_ratio=2.5, min_saving=0.25):
    """Suppression des boucles redondantes."""
    cleaned = [route_nodes[0]]
    i = 0
    removed = 0
    
    while i < len(route_nodes) - 1:
        best_skip = 0
        best_path = None
        best_saving = 0
        
        for lookahead in range(2, min(13, len(route_nodes) - i)):
            current = route_nodes[i]
            target = route_nodes[i + lookahead]
            
            try:
                direct_dist = ox.distance.great_circle(
                    G.nodes[current]['y'], G.nodes[current]['x'],
                    G.nodes[target]['y'], G.nodes[target]['x']
                )
            except:
                continue
            
            routed_length = 0
            try:
                for j in range(lookahead):
                    u = route_nodes[i + j]
                    v = route_nodes[i + j + 1]
                    routed_length += G[u][v][0]['length']
            except:
                continue
            
            if routed_length > threshold_ratio * direct_dist and direct_dist < 400:
                try:
                    direct_path = nx.shortest_path(G, current, target, weight='length')
                    direct_length = nx.shortest_path_length(G, current, target, weight='length')
                    
                    if direct_length < routed_length * (1 - min_saving):
                        saving = routed_length - direct_length
                        if saving > best_saving:
                            best_skip = lookahead
                            best_path = direct_path
                            best_saving = saving
                except:
                    pass
        
        if best_skip > 0 and best_path:
            cleaned.extend(best_path[1:])
            i += best_skip
            removed += 1
        else:
            cleaned.append(route_nodes[i + 1])
            i += 1
    
    return cleaned, removed


@st.cache_resource
def load_paris_graph():
    """Charge le graphe de Paris (mise en cache)."""
    G = ox.graph_from_place("Paris, France", network_type="drive")
    return G


def process_image(uploaded_file, epsilon_ratio, n_points_override=None):
    """Pipeline complet de traitement."""
    
    # Charger l'image
    img_pil = Image.open(uploaded_file).convert("L")
    img_array = np.array(img_pil)
    
    # Auto-calibration
    params = auto_calibrate(img_array)
    IMG_SIZE = params["IMG_SIZE"]
    BINARY_THRESHOLD = params["BINARY_THRESHOLD"]
    N_POINTS = n_points_override if n_points_override else params["N_POINTS"]
    
    # Redimensionner
    img_pil = img_pil.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    img_arr = np.array(img_pil)
    
    # Extraction du contour
    binary = (img_arr < BINARY_THRESHOLD).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("❌ Aucun contour détecté dans l'image")
    
    contour_raw = max(contours, key=len)
    contour_smooth = smooth_contour(contour_raw, epsilon_ratio=epsilon_ratio)
    pts = contour_smooth.squeeze()
    
    # Normalisation
    xs_norm = pts[:, 0] / (IMG_SIZE - 1)
    ys_norm = 1 - pts[:, 1] / (IMG_SIZE - 1)
    
    # Sous-échantillonnage
    indices = np.round(np.linspace(0, len(xs_norm) - 1, N_POINTS)).astype(int)
    xs_sample = xs_norm[indices]
    ys_sample = ys_norm[indices]
    
    # Charger le graphe de Paris
    G = load_paris_graph()
    nodes_gdf, edges_gdf = ox.graph_to_gdfs(G)
    minlon = nodes_gdf.geometry.x.min()
    maxlon = nodes_gdf.geometry.x.max()
    minlat = nodes_gdf.geometry.y.min()
    maxlat = nodes_gdf.geometry.y.max()
    
    # Projection
    MARGIN = 0.05
    lon_range = (maxlon - minlon) * (1 - 2 * MARGIN)
    lat_range = (maxlat - minlat) * (1 - 2 * MARGIN)
    lons_sample = minlon + MARGIN * (maxlon - minlon) + xs_sample * lon_range
    lats_sample = minlat + MARGIN * (maxlat - minlat) + ys_sample * lat_range
    
    # Snapping
    snapped_nodes = ox.distance.nearest_nodes(G, lons_sample, lats_sample)
    
    # Routing
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
    
    # Fermeture du circuit
    try:
        closing = nx.shortest_path(G, snapped_nodes[-1], snapped_nodes[0], weight="length")
        all_route_nodes.extend(closing[1:])
    except nx.NetworkXNoPath:
        pass
    
    initial_nodes = len(all_route_nodes)
    
    # Post-filtrage
    all_route_nodes, removed = remove_redundant_loops(all_route_nodes, G)
    
    # Extraction coordonnées
    route_lons = [G.nodes[n]["x"] for n in all_route_nodes]
    route_lats = [G.nodes[n]["y"] for n in all_route_nodes]
    
    return {
        "img_arr": img_arr,
        "contour_raw": contour_raw,
        "pts": pts,
        "indices": indices,
        "route_lons": route_lons,
        "route_lats": route_lats,
        "edges_gdf": edges_gdf,
        "minlon": minlon,
        "maxlon": maxlon,
        "minlat": minlat,
        "maxlat": maxlat,
        "params": params,
        "initial_nodes": initial_nodes,
        "final_nodes": len(all_route_nodes),
        "failed": failed,
    }


def create_visualization(data):
    """Crée la visualisation finale."""
    BG = "#0a0a14"
    CARD_BG = "#12122a"
    
    fig = plt.figure(figsize=(20, 10), facecolor=BG)
    gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1, 1.8], wspace=0.04)
    
    # Panneau gauche : contour
    ax0 = fig.add_subplot(gs[0])
    ax0.set_facecolor(CARD_BG)
    IMG_SIZE = data["img_arr"].shape[0]
    ax0.imshow(data["img_arr"], cmap="gray", origin="upper",
               extent=[0, IMG_SIZE, IMG_SIZE, 0])
    
    pts_raw = data["contour_raw"].squeeze()
    ax0.plot(pts_raw[:, 0], pts_raw[:, 1],
             color="#555566", linewidth=0.8, alpha=0.4)
    ax0.plot(data["pts"][:, 0], data["pts"][:, 1],
             color="#ff6b9d", linewidth=1.5, alpha=0.8)
    
    scatter_colors = np.linspace(0, 1, len(data["indices"]))
    ax0.scatter(data["pts"][data["indices"], 0],
                data["pts"][data["indices"], 1],
                c=scatter_colors, cmap="plasma", s=20, zorder=5, alpha=0.9)
    ax0.set_title("Contour détecté", color="#aaaacc",
                  fontsize=13, pad=10, fontfamily="monospace")
    ax0.axis("off")
    
    # Panneau droit : carte
    ax1 = fig.add_subplot(gs[1])
    ax1.set_facecolor(BG)
    data["edges_gdf"].plot(ax=ax1, color="#1e1e3a", linewidth=0.25, alpha=0.9)
    
    points_arr = np.array([data["route_lons"], data["route_lats"]]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points_arr[:-1], points_arr[1:]], axis=1)
    lc = LineCollection(segments, cmap="plasma",
                        norm=plt.Normalize(0, len(segments)),
                        linewidth=2.5, alpha=0.95, capstyle="round")
    lc.set_array(np.arange(len(segments)))
    ax1.add_collection(lc)
    
    ax1.scatter([data["route_lons"][0]], [data["route_lats"][0]],
                c="#00ffaa", s=100, zorder=7, edgecolors="white", linewidths=1)
    ax1.scatter([data["route_lons"][-1]], [data["route_lats"][-1]],
                c="#ff4466", s=100, zorder=7, edgecolors="white", linewidths=1)
    
    pad_lon = (data["maxlon"] - data["minlon"]) * 0.02
    pad_lat = (data["maxlat"] - data["minlat"]) * 0.02
    ax1.set_xlim(data["minlon"] - pad_lon, data["maxlon"] + pad_lon)
    ax1.set_ylim(data["minlat"] - pad_lat, data["maxlat"] + pad_lat)
    ax1.set_title(f"Circuit sur Paris · {data['final_nodes']} nœuds",
                  color="#aaaacc", fontsize=13, pad=10, fontfamily="monospace")
    ax1.axis("off")
    
    fig.suptitle("GribouRun", color="#ffffff",
                 fontsize=18, fontweight="bold", y=0.98, fontfamily="monospace")
    
    plt.tight_layout()
    
    # Sauvegarder en buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor=BG)
    buf.seek(0)
    plt.close()
    
    return buf


# ╔═══════════════════════════════════════════════════════════╗
# ║                    INTERFACE STREAMLIT                   ║
# ╚═══════════════════════════════════════════════════════════╝

st.title("🎨 GribouRun — Dessine sur les rues de Paris")
st.markdown("Transforme ton dessin en itinéraire sur les rues de Paris !")

# Sidebar pour les paramètres
with st.sidebar:
    st.header("⚙️ Paramètres")
    epsilon = st.slider(
        "Lissage du contour",
        min_value=0.003,
        max_value=0.020,
        value=0.008,
        step=0.001,
        help="Plus bas = plus de détails, plus haut = plus lissé"
    )
    
    n_points_manual = st.number_input(
        "Nombre de points (0 = auto)",
        min_value=0,
        max_value=600,
        value=0,
        step=50,
        help="Laisse à 0 pour calcul automatique"
    )

# Upload de l'image
uploaded_file = st.file_uploader(
    "📤 Téléverse ton dessin (noir sur fond blanc)",
    type=['jpg', 'jpeg', 'png']
)

if uploaded_file is not None:
    # Afficher l'image uploadée
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(uploaded_file, caption="Ton dessin", use_container_width=True)
    
    # Bouton de traitement
    if st.button("🚀 Générer l'itinéraire", type="primary", use_container_width=True):
        with st.spinner("🔄 Traitement en cours... (peut prendre 1-2 minutes)"):
            try:
                # Traitement
                n_override = n_points_manual if n_points_manual > 0 else None
                result = process_image(uploaded_file, epsilon, n_override)
                
                # Stats
                reduction = 100 * (result["initial_nodes"] - result["final_nodes"]) / result["initial_nodes"]
                
                st.success("✅ Traitement terminé !")
                
                # Afficher les stats
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Nœuds du circuit", result["final_nodes"])
                with col2:
                    st.metric("Réduction", f"{reduction:.1f}%")
                with col3:
                    st.metric("Points de contrôle", result["params"]["N_POINTS"])
                with col4:
                    st.metric("Échecs routing", result["failed"])
                
                # Créer et afficher la visualisation
                img_buffer = create_visualization(result)
                st.image(img_buffer, use_container_width=True)
                
                # Bouton de téléchargement
                st.download_button(
                    label="💾 Télécharger le résultat",
                    data=img_buffer,
                    file_name="gribourun_result.png",
                    mime="image/png",
                    use_container_width=True
                )
                
            except Exception as e:
                st.error(f"❌ Erreur : {str(e)}")
                st.exception(e)

else:
    # Instructions
    st.info("👆 Commence par téléverser une image de ton dessin (trait noir sur fond blanc)")
    
    with st.expander("📖 Comment ça marche ?"):
        st.markdown("""
        1. **Upload** ton dessin (simple trait noir sur fond blanc)
        2. L'algorithme détecte automatiquement le contour
        3. Il projette ce contour sur les rues de Paris
        4. Il calcule un itinéraire continu suivant la forme
        5. Tu obtiens une carte avec ton dessin tracé sur les vraies rues !
        
        **Conseils :**
        - Utilise un dessin simple avec un trait bien visible
        - Fond blanc, trait noir (ou très sombre)
        - Évite les dessins trop complexes pour un premier essai
        """)
