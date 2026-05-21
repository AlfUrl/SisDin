"""
Módulo de uso de suelo (land-use) para la rejilla de dispersión atmosférica.

Proporciona:
  - Constantes LUSE_* : tipos de celda de uso de suelo
  - PROPIEDADES_LUSE  : tabla de factores de difusión y fricción por tipo
  - fetch_osm_landuse : consulta Overpass para obtener polígonos OSM de land-use
  - DEFAULT_LANDUSE   : respaldo hardcodeado con zonas conocidas de CU-UANL
  - build_landuse_map : rasteriza los polígonos sobre la rejilla → matriz LUSE_*
  - build_wind_maps   : genera D_map, u_scale_map y v_scale_map a partir del land-use

Integración con simulator.py:
    landuse_map             = build_landuse_map(grid)
    D_map, us_map, vs_map   = build_wind_maps(grid, landuse_map, D_base, u, v)
    # y en _dispersion_step usar D_map en lugar de D escalar.

Equipo 11 – Brigada 003 – Modelado y Simulación de Sistemas Dinámicos
"""
from __future__ import annotations

import numpy as np
import requests
import streamlit as st
from matplotlib.path import Path as MplPath

# ---------------------------------------------------------------------------
# 1. TIPOS DE CELDA DE USO DE SUELO
# ---------------------------------------------------------------------------

LUSE_VACIO          = 0   # terreno no clasificado / asfalto genérico
LUSE_ZONA_ARBOLADA  = 1   # parque, bosque, campus arbolado
LUSE_EDIFICIO_DENSO = 2   # bloques de edificios / cañón urbano
LUSE_ZONA_ABIERTA   = 3   # campo deportivo, estacionamiento, pradera
LUSE_ZONA_AGUA      = 4   # lago, fuente, canal (rugosidad mínima)
LUSE_ZONA_INDUSTRIAL= 5   # zona industrial / Ternium

# ---------------------------------------------------------------------------
# 2. PROPIEDADES AERODINÁMICAS POR TIPO DE CELDA
# ---------------------------------------------------------------------------
# D_factor  : multiplica el coeficiente de difusión turbulenta base.
#             > 1 → más mezcla (p.ej. zona arbolada crea turbulencia mecánica)
#             < 1 → menos mezcla (cañón urbano atrapa el aire)
# wind_factor : escala la velocidad efectiva del viento (u y v).
#             < 1 → viento frenado (arbolado, edificios)
#             > 1 → aceleración (zona abierta, efecto de canalización)
PROPIEDADES_LUSE = {
    LUSE_VACIO:           {"D_factor": 1.00, "wind_factor": 1.00},
    LUSE_ZONA_ARBOLADA:   {"D_factor": 1.70, "wind_factor": 0.40},
    LUSE_EDIFICIO_DENSO:  {"D_factor": 0.55, "wind_factor": 0.20},
    LUSE_ZONA_ABIERTA:    {"D_factor": 1.15, "wind_factor": 1.15},
    LUSE_ZONA_AGUA:       {"D_factor": 0.85, "wind_factor": 1.25},
    LUSE_ZONA_INDUSTRIAL: {"D_factor": 0.90, "wind_factor": 0.80},
}

# ---------------------------------------------------------------------------
# 3. MAPA DE TAGS OSM → TIPO DE CELDA
# ---------------------------------------------------------------------------
# Tags de landuse / natural / leisure que Overpass devuelve para polígonos.
_OSM_TAG_MAP: dict[tuple[str, str], int] = {
    # --- Arbolado ---
    ("landuse", "forest"):         LUSE_ZONA_ARBOLADA,
    ("landuse", "orchard"):        LUSE_ZONA_ARBOLADA,
    ("landuse", "vineyard"):       LUSE_ZONA_ARBOLADA,
    ("natural", "wood"):           LUSE_ZONA_ARBOLADA,
    ("natural", "scrub"):          LUSE_ZONA_ARBOLADA,
    ("leisure", "park"):           LUSE_ZONA_ARBOLADA,
    ("leisure", "garden"):         LUSE_ZONA_ARBOLADA,
    ("landuse", "grass"):          LUSE_ZONA_ARBOLADA,
    ("landuse", "village_green"):  LUSE_ZONA_ARBOLADA,

    # --- Edificación densa ---
    ("landuse", "commercial"):     LUSE_EDIFICIO_DENSO,
    ("landuse", "retail"):         LUSE_EDIFICIO_DENSO,
    ("landuse", "residential"):    LUSE_EDIFICIO_DENSO,
    ("landuse", "construction"):   LUSE_EDIFICIO_DENSO,
    ("building", "yes"):           LUSE_EDIFICIO_DENSO,
    ("building", "university"):    LUSE_EDIFICIO_DENSO,
    ("building", "school"):        LUSE_EDIFICIO_DENSO,
    ("building", "office"):        LUSE_EDIFICIO_DENSO,

    # --- Zona abierta ---
    ("landuse", "recreation_ground"): LUSE_ZONA_ABIERTA,
    ("leisure", "pitch"):          LUSE_ZONA_ABIERTA,
    ("leisure", "stadium"):        LUSE_ZONA_ABIERTA,
    ("leisure", "track"):          LUSE_ZONA_ABIERTA,
    ("amenity", "parking"):        LUSE_ZONA_ABIERTA,
    ("landuse", "farmland"):       LUSE_ZONA_ABIERTA,
    ("natural", "heath"):          LUSE_ZONA_ABIERTA,

    # --- Agua ---
    ("natural", "water"):          LUSE_ZONA_AGUA,
    ("waterway", "river"):         LUSE_ZONA_AGUA,
    ("waterway", "canal"):         LUSE_ZONA_AGUA,
    ("landuse", "reservoir"):      LUSE_ZONA_AGUA,
    ("leisure", "swimming_pool"):  LUSE_ZONA_AGUA,

    # --- Industrial ---
    ("landuse", "industrial"):     LUSE_ZONA_INDUSTRIAL,
    ("landuse", "railway"):        LUSE_ZONA_INDUSTRIAL,
}

# ---------------------------------------------------------------------------
# 4. RESPALDO HARDCODEADO: zonas conocidas de CU-UANL y alrededores
# ---------------------------------------------------------------------------
# Si Overpass falla, estas zonas se rasterizarán sobre la rejilla.
# Cada entrada es un polígono [(lat, lon), …] con su tipo LUSE_*.
# Los vértices trazan el contorno aproximado de cada zona.

DEFAULT_LANDUSE: list[dict] = [
    # Bosque / arbolado central del campus
    {
        "name": "Área arbolada central CU",
        "tipo": LUSE_ZONA_ARBOLADA,
        "coords": [
            (25.7295, -100.3155), (25.7295, -100.3110),
            (25.7265, -100.3110), (25.7265, -100.3155),
            (25.7295, -100.3155),
        ],
    },
    # Jardines noreste del campus
    {
        "name": "Jardines norte CU",
        "tipo": LUSE_ZONA_ARBOLADA,
        "coords": [
            (25.7325, -100.3145), (25.7325, -100.3095),
            (25.7300, -100.3095), (25.7300, -100.3145),
            (25.7325, -100.3145),
        ],
    },
    # Estadio universitario (zona abierta)
    {
        "name": "Estadio Universitario",
        "tipo": LUSE_ZONA_ABIERTA,
        "coords": [
            (25.7280, -100.3185), (25.7280, -100.3155),
            (25.7260, -100.3155), (25.7260, -100.3185),
            (25.7280, -100.3185),
        ],
    },
    # Campos deportivos sur
    {
        "name": "Campos deportivos sur",
        "tipo": LUSE_ZONA_ABIERTA,
        "coords": [
            (25.7225, -100.3150), (25.7225, -100.3095),
            (25.7205, -100.3095), (25.7205, -100.3150),
            (25.7225, -100.3150),
        ],
    },
    # Zona de edificios académicos principales (FIME, FCFM, etc.)
    {
        "name": "Edificios académicos norte",
        "tipo": LUSE_EDIFICIO_DENSO,
        "coords": [
            (25.7340, -100.3130), (25.7340, -100.3070),
            (25.7305, -100.3070), (25.7305, -100.3130),
            (25.7340, -100.3130),
        ],
    },
    # Zona de Ternium / industrial al sureste
    {
        "name": "Zona industrial Ternium",
        "tipo": LUSE_ZONA_INDUSTRIAL,
        "coords": [
            (25.7240, -100.3060), (25.7240, -100.2980),
            (25.7180, -100.2980), (25.7180, -100.3060),
            (25.7240, -100.3060),
        ],
    },
    # Estacionamientos grandes (zona abierta)
    {
        "name": "Estacionamiento general",
        "tipo": LUSE_ZONA_ABIERTA,
        "coords": [
            (25.7310, -100.3190), (25.7310, -100.3160),
            (25.7285, -100.3160), (25.7285, -100.3190),
            (25.7310, -100.3190),
        ],
    },
]

# ---------------------------------------------------------------------------
# 5. CONSULTA OSM (Overpass) – mismo patrón que traffic.py
# ---------------------------------------------------------------------------

_OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    "https://overpass.openstreetmap.ru/api/interpreter",
]
_OVERPASS_HEADERS = {
    "User-Agent": "SimuladorCalidadAire-UANL/1.0 (proyecto academico)"
}

# Tags que pedimos en la consulta (como unión de valores).
_QUERY_FILTERS = """
  (
    way["landuse"]({bbox});
    way["natural"~"wood|water|scrub|heath"]({bbox});
    way["leisure"~"park|garden|pitch|stadium|track|swimming_pool"]({bbox});
    way["amenity"="parking"]({bbox});
    way["building"~"yes|university|school|office"]({bbox});
  );
  out geom;
"""


@st.cache_data(ttl=86400, show_spinner="Consultando land-use en OpenStreetMap…")
def fetch_osm_landuse(bounds: dict, buffer_deg: float = 0.001) -> list[dict]:
    """
    Descarga polígonos de uso de suelo del área de estudio desde Overpass.

    Args:
        bounds: dict con min_lat, max_lat, min_lon, max_lon
        buffer_deg: ampliación del bbox (≈ 110 m por 0.001°)

    Returns:
        Lista de dicts {name, tipo (LUSE_*), coords [(lat, lon), …]}.
        Si todos los mirrors fallan, devuelve DEFAULT_LANDUSE.
    """
    s = bounds["min_lat"] - buffer_deg
    w = bounds["min_lon"] - buffer_deg
    n = bounds["max_lat"] + buffer_deg
    e = bounds["max_lon"] + buffer_deg
    bbox = f"{s},{w},{n},{e}"
    query = "[out:json][timeout:30];" + _QUERY_FILTERS.format(bbox=bbox)

    ultimo_error = None
    for endpoint in _OVERPASS_ENDPOINTS:
        try:
            r = requests.post(
                endpoint,
                data={"data": query},
                headers=_OVERPASS_HEADERS,
                timeout=35,
            )
            r.raise_for_status()
            zonas = _parse_landuse(r.json())
            if zonas:
                return zonas
            ultimo_error = "respuesta vacía"
        except Exception as exc:  # noqa: BLE001
            ultimo_error = type(exc).__name__
            continue

    st.warning(
        f"⚠️ Land-use OSM no disponible ({ultimo_error}). "
        "Usando zonas hardcodeadas de CU."
    )
    return DEFAULT_LANDUSE


def _parse_landuse(data: dict) -> list[dict]:
    """Convierte respuesta Overpass → lista de zonas de land-use."""
    zonas = []
    for el in data.get("elements", []):
        if el.get("type") != "way" or "geometry" not in el:
            continue
        tags = el.get("tags", {})
        tipo = _classify_tags(tags)
        if tipo is None:
            continue
        coords = [(g["lat"], g["lon"]) for g in el["geometry"]]
        if len(coords) < 3:
            continue
        zonas.append({
            "name":   tags.get("name", f"osm_{el['id']}"),
            "tipo":   tipo,
            "coords": coords,
        })
    return zonas


def _classify_tags(tags: dict) -> int | None:
    """Asigna LUSE_* a un conjunto de tags OSM. Retorna None si no aplica."""
    # Prioridad: industrial > agua > arbolado > abierto > edificio
    priority_order = [
        LUSE_ZONA_INDUSTRIAL,
        LUSE_ZONA_AGUA,
        LUSE_ZONA_ARBOLADA,
        LUSE_ZONA_ABIERTA,
        LUSE_EDIFICIO_DENSO,
    ]
    found: dict[int, bool] = {}
    for (tag_key, tag_val), luse_type in _OSM_TAG_MAP.items():
        val = tags.get(tag_key, "")
        # tag_val puede ser "yes" (match exacto) o un valor específico
        if tag_val == "yes":
            if val:  # cualquier valor no vacío
                found[luse_type] = True
        elif val == tag_val:
            found[luse_type] = True

    for luse_type in priority_order:
        if found.get(luse_type):
            return luse_type
    return None


# ---------------------------------------------------------------------------
# 6. RASTERIZACIÓN DE POLÍGONOS → MATRIZ LUSE_*
# ---------------------------------------------------------------------------

def build_landuse_map(
    grid: dict,
    zonas: list[dict] | None = None,
    bounds: dict | None = None,
    usar_osm: bool = True,
) -> np.ndarray:
    """
    Rasteriza las zonas de uso de suelo sobre la rejilla.

    Args:
        grid: salida de simulator.build_grid
        zonas: lista ya cargada de zonas (si None, se llama fetch o DEFAULT)
        bounds: bounds de la rejilla (si None se usa grid["bounds"])
        usar_osm: si True intenta Overpass antes de usar el respaldo

    Returns:
        landuse_map: ndarray int8 (filas × columnas) con valores LUSE_*.
    """
    if zonas is None:
        if usar_osm:
            bounds_eff = bounds or grid["bounds"]
            zonas = fetch_osm_landuse(bounds_eff)
        else:
            zonas = DEFAULT_LANDUSE

    lm = np.zeros((grid["filas"], grid["columnas"]), dtype=np.int8)

    lon_mesh, lat_mesh = np.meshgrid(grid["lon_grid"], grid["lat_grid"])
    pts_all = np.column_stack((lat_mesh.ravel(), lon_mesh.ravel()))

    # Rasterizamos en orden inverso de prioridad (el último que escribe gana,
    # y queremos que industrial > agua > arbolado > abierto > edificio).
    # Invertimos: el de mayor prioridad se procesa ÚLTIMO.
    priority_order = [
        LUSE_EDIFICIO_DENSO,
        LUSE_ZONA_ABIERTA,
        LUSE_ZONA_ARBOLADA,
        LUSE_ZONA_AGUA,
        LUSE_ZONA_INDUSTRIAL,
    ]
    # Agrupa zonas por tipo
    por_tipo: dict[int, list[list]] = {t: [] for t in priority_order}
    for zona in zonas:
        t = zona["tipo"]
        if t in por_tipo:
            por_tipo[t].append(zona["coords"])

    for tipo in priority_order:
        for coords in por_tipo[tipo]:
            # Necesitamos (lat, lon) pairs para el Path de matplotlib
            poly = MplPath([(lat, lon) for lat, lon in coords])
            inside = poly.contains_points(pts_all).reshape(
                grid["filas"], grid["columnas"]
            )
            lm[inside] = tipo

    return lm


# ---------------------------------------------------------------------------
# 7. CONSTRUCCIÓN DE MAPAS DE VIENTO Y DIFUSIÓN
# ---------------------------------------------------------------------------

def build_wind_maps(
    grid: dict,
    landuse_map: np.ndarray,
    D_base: float,
    u: float,
    v: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Genera las matrices espacialmente variables de difusión y viento.

    Args:
        grid:        rejilla (no se usa directamente; se incluye por simetría)
        landuse_map: matriz LUSE_* (filas × columnas)
        D_base:      coeficiente de difusión escalar base (m²/s)
        u:           componente E-W del viento (m/s)
        v:           componente N-S del viento (m/s)

    Returns:
        D_map   : difusión turbulenta efectiva por celda (m²/s)
        u_map   : componente u ajustada por fricción (m/s)
        v_map   : componente v ajustada por fricción (m/s)
    """
    shape = landuse_map.shape
    D_map = np.full(shape, D_base, dtype=np.float32)
    u_map = np.full(shape, u,      dtype=np.float32)
    v_map = np.full(shape, v,      dtype=np.float32)

    for luse_type, props in PROPIEDADES_LUSE.items():
        mask = (landuse_map == luse_type)
        if not mask.any():
            continue
        D_map[mask] = D_base * props["D_factor"]
        u_map[mask] = u      * props["wind_factor"]
        v_map[mask] = v      * props["wind_factor"]

    return D_map, u_map, v_map


# ---------------------------------------------------------------------------
# 8. UTILIDAD DE INSPECCIÓN
# ---------------------------------------------------------------------------

def resumen_landuse(landuse_map: np.ndarray) -> dict:
    """Cuenta celdas por tipo de uso de suelo para mostrar en la UI."""
    nombres = {
        LUSE_VACIO:           "Vacío / asfalto",
        LUSE_ZONA_ARBOLADA:   "Arbolado / parque",
        LUSE_EDIFICIO_DENSO:  "Edificación densa",
        LUSE_ZONA_ABIERTA:    "Zona abierta",
        LUSE_ZONA_AGUA:       "Agua",
        LUSE_ZONA_INDUSTRIAL: "Industrial",
    }
    total = landuse_map.size
    resumen = {}
    for luse_type, nombre in nombres.items():
        n = int((landuse_map == luse_type).sum())
        resumen[nombre] = {"celdas": n, "pct": round(100.0 * n / total, 1)}
    return resumen
