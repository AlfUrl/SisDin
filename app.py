"""
Simulador de Calidad del Aire - Ciudad Universitaria UANL
==========================================================

Interfaz Streamlit del motor de simulación de dispersión atmosférica.

Funcionalidades:
  - Mapa interactivo con heatmap de ICA sobre el polígono de CU
  - Selección de contaminante: PM2.5 / PM10 / NOx / SO2
  - Clima en tiempo real (Open-Meteo) u overrides manuales
  - Escenarios: día normal, hora pico, inversión térmica, reducción
    de tráfico, viento desde Ternium
  - Pronóstico horario (alertas anticipadas)
  - Recomendación de ruta de menor exposición entre dos puntos
  - Recomendaciones de cubrebocas y alertas por nivel

Equipo 11 - Brigada 003 - Modelado y Simulación
"""
import streamlit as st
import time
from config import PROJECT_NAME


# =====================================================================
# CONFIGURACION
# =====================================================================

st.set_page_config(
    page_title=f"{PROJECT_NAME} - Simulador Calidad del Aire - UANL",
    page_icon="🌫️",
    layout="wide",
)


# =====================================================================
# PANTALLA DE CARGA INICIAL
# =====================================================================
from loading_screen import show_loading_screen, remove_loading_screen

loading_placeholder = show_loading_screen()


# =====================================================================
# IMPORTACIONES PESADAS Y MODELOS (ejecutados tras pintar la pantalla de carga)
# =====================================================================
import base64
import io
from datetime import datetime

import folium
import matplotlib
matplotlib.use("Agg")  # backend sin display, para renderizar a PNG
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon, Rectangle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from PIL import Image
from streamlit_folium import st_folium

from simulator import (
    CENTER_LAT, CENTER_LON, POLYGON_LIMITS, TERNIUM_AREA,
    build_grid, build_infrastructure, build_mask, calculate_emissions,
    calculate_ica, categoria_ica, coords_to_index, index_to_coords,
    run_dispersion, simular_escenario, simular_con_trafico,
    simular_dia_completo, simular_animacion,
)
from traffic import (
    DEFAULT_ROADS, fetch_osm_roads, build_traffic_map,
    emissions_from_traffic, resumen_red, HOURLY_PROFILE,
)
from tomtom import indice_congestion_zona
from sima import (
    ESTACIONES_SIMA, cargar_datos_sima, validar_serie,
    generar_sima_ejemplo, interpretar_metricas,
)
from weather import get_current_weather, get_hourly_forecast, grados_a_cardinal
from recommendations import (
    detectar_picos, factores_ambientales, find_clean_route, generate_alert,
    hourly_pollution_forecast, mask_recommendation,
    recomendaciones_de_accion, recomendaciones_pronostico, route_stats,
)
from landuse import (
    build_landuse_map, resumen_landuse, DEFAULT_LANDUSE,
    LUSE_VACIO, LUSE_ZONA_ARBOLADA, LUSE_EDIFICIO_DENSO,
    LUSE_ZONA_ABIERTA, LUSE_ZONA_AGUA, LUSE_ZONA_INDUSTRIAL,
    fetch_osm_landuse,
)


# =====================================================================
# RECURSOS EN CACHE (se construyen una sola vez por sesión)
# =====================================================================

@st.cache_resource
def cargar_grid_y_infra():
    grid = build_grid()
    mask = build_mask(grid)
    inf = build_infrastructure(grid)
    return grid, mask, inf


@st.cache_resource(show_spinner="Cargando red vial…")
def cargar_red_vial(usar_osm: bool):
    """Devuelve la red vial: OSM en vivo o respaldo local."""
    grid, _, _ = cargar_grid_y_infra()
    if usar_osm:
        return fetch_osm_roads(grid["bounds"])
    return DEFAULT_ROADS


@st.cache_resource(show_spinner="Cargando uso de suelo…")
def cargar_landuse(usar_osm: bool):
    """Rasteriza las zonas de uso de suelo sobre la rejilla (con caché)."""
    grid, _, _ = cargar_grid_y_infra()
    if usar_osm:
        zonas = fetch_osm_landuse(grid["bounds"])
    else:
        zonas = DEFAULT_LANDUSE
    return build_landuse_map(grid, zonas=zonas)


@st.cache_data(show_spinner="Calculando dispersión con tráfico…")
def simular_snapshot(hora, contaminante, viento_ms, viento_dir,
                     temperatura, presion, factor_trafico, factor_industrial,
                     inversion, es_dia_laboral, usar_osm):
    grid, _, _ = cargar_grid_y_infra()
    roads = cargar_red_vial(usar_osm)
    res = simular_con_trafico(
        grid, roads,
        hora=hora, contaminante=contaminante,
        viento_ms=viento_ms, viento_dir=viento_dir,
        temperatura=temperatura, presion=presion,
        factor_trafico=factor_trafico,
        factor_industrial=factor_industrial,
        inversion_termica=inversion,
        es_dia_laboral=es_dia_laboral,
    )
    return res


@st.cache_data(show_spinner="Simulando evolución 24h…")
def simular_24h(contaminante, viento_ms, viento_dir, temperatura, presion,
                factor_trafico, factor_industrial, es_dia_laboral,
                inversion_horas_tuple, usar_osm):
    grid, _, _ = cargar_grid_y_infra()
    roads = cargar_red_vial(usar_osm)
    clima = {"velocidad_viento": viento_ms, "direccion_viento": viento_dir,
             "temperatura": temperatura, "presion": presion}
    frames = simular_dia_completo(
        grid, roads,
        contaminante=contaminante,
        perfil_meteo=clima,
        factor_trafico=factor_trafico,
        factor_industrial=factor_industrial,
        es_dia_laboral=es_dia_laboral,
        inversion_horas=set(inversion_horas_tuple),
        minutos_por_hora=10,
    )
    return frames


@st.cache_data(show_spinner="Generando animación de la pluma…")
def simular_animacion_cached(hora, contaminante, viento_ms, viento_dir,
                             temperatura, presion, factor_trafico,
                             factor_industrial, inversion, es_dia_laboral,
                             usar_osm, tiempo_simulado_s, n_frames):
    grid, _, _ = cargar_grid_y_infra()
    roads = cargar_red_vial(usar_osm)
    return simular_animacion(
        grid, roads,
        hora=hora, contaminante=contaminante,
        viento_ms=viento_ms, viento_dir=viento_dir,
        temperatura=temperatura, presion=presion,
        factor_trafico=factor_trafico,
        factor_industrial=factor_industrial,
        inversion_termica=inversion,
        es_dia_laboral=es_dia_laboral,
        tiempo_simulado_s=tiempo_simulado_s,
        n_frames=n_frames,
    )


# =====================================================================
# UTILIDADES DE VISUALIZACION
# =====================================================================

# Paleta vívida y continua para el ICA. Mantiene la semántica
# verde=bueno -> rojo=malo -> violeta=peligroso, pero con colores
# saturados estilo Material Design para que las "manchas" resalten.
ICA_STRIKING_STOPS = [
    (0,   "#00b894"),   # verde esmeralda
    (25,  "#55c57a"),   # verde
    (50,  "#a8d65c"),   # verde-lima
    (75,  "#ffd23f"),   # amarillo dorado
    (100, "#ff9f1c"),   # ámbar
    (130, "#ff6b35"),   # naranja intenso
    (160, "#e63946"),   # rojo
    (200, "#b00020"),   # carmesí
    (250, "#8e24aa"),   # magenta-violeta
]

ICA_MAX_PLOT = 250  # tope de la escala de color


def _crear_colormap_striking():
    """Colormap de matplotlib vívido y continuo para el ICA."""
    pts = [(v / ICA_MAX_PLOT, c) for v, c in ICA_STRIKING_STOPS]
    return mcolors.LinearSegmentedColormap.from_list("ica_striking", pts, N=256)


# Paleta de ALTO CONTRASTE: colores más saturados/luminosos, pensados para
# fondo oscuro. Los azules-verdes de bajo ICA quedan muy sutiles para que
# las zonas contaminadas (naranja → rojo → magenta) resalten dramáticamente.
ICA_HIGH_CONTRAST_STOPS = [
    (0,   "#0a3d33"),   # casi negro-verdoso (aire limpio = casi invisible)
    (25,  "#1abc9c"),   # turquesa
    (50,  "#52d273"),   # verde fluo
    (75,  "#ffea00"),   # amarillo eléctrico
    (100, "#ff8c00"),   # naranja vivo
    (130, "#ff5500"),   # rojo-naranja brillante
    (160, "#ff1744"),   # rojo intenso
    (200, "#ff00aa"),   # magenta brillante
    (250, "#d500ff"),   # violeta eléctrico
]


def _crear_colormap_alto_contraste():
    """Colormap saturado/eléctrico para modo de alto contraste."""
    pts = [(v / ICA_MAX_PLOT, c) for v, c in ICA_HIGH_CONTRAST_STOPS]
    return mcolors.LinearSegmentedColormap.from_list("ica_hc", pts, N=256)


_CMAP_ICA = _crear_colormap_striking()
_CMAP_ICA_HC = _crear_colormap_alto_contraste()


def _cmap_para(alto_contraste: bool):
    """Devuelve el colormap apropiado según el modo de visualización."""
    return _CMAP_ICA_HC if alto_contraste else _CMAP_ICA


def _suavizar_campo(campo, sigma=1.8):
    """
    Suavizado gaussiano separable (implementado solo con numpy, sin scipy).

    Convierte la cuadrícula "pixelada" del ICA en un campo continuo, de modo
    que las zonas de mayor concentración se vean como manchas orgánicas.
    """
    campo = np.asarray(campo, dtype=np.float32)
    if sigma <= 0:
        return campo
    radio = max(1, int(np.ceil(sigma * 3)))
    x = np.arange(-radio, radio + 1)
    k = np.exp(-(x ** 2) / (2.0 * sigma ** 2)).astype(np.float32)
    k /= k.sum()

    # Convolución separable: primero a lo largo de las columnas, luego filas.
    padded = np.pad(campo, ((0, 0), (radio, radio)), mode="edge")
    out = np.stack([np.convolve(padded[i], k, mode="valid")
                    for i in range(padded.shape[0])])
    padded = np.pad(out, ((radio, radio), (0, 0)), mode="edge")
    out = np.stack([np.convolve(padded[:, j], k, mode="valid")
                    for j in range(padded.shape[1])], axis=1)
    return out.astype(np.float32)


# =====================================================================
# BRUJULA VISUAL DEL VIENTO
# =====================================================================

def _brujula_svg(direccion_meteo: float, velocidad_ms: float | None = None,
                 size: int = 170, alto_contraste: bool = False,
                 idx: int = 0) -> str:
    """
    Genera un SVG de brújula con una flecha que muestra HACIA DÓNDE sopla el
    viento (la pluma viaja en esa dirección).

    Args:
        direccion_meteo: dirección meteorológica en grados (de dónde VIENE).
        velocidad_ms: opcional, etiqueta de velocidad.
        size: tamaño en pixeles.
        alto_contraste: si True, usa colores oscuros.
        idx: índice único para que múltiples SVGs en la misma página no
             choquen al definir el `<marker>`.

    Returns:
        Cadena con el SVG inline (lista para usar con st.markdown).
    """
    import math as _m
    if alto_contraste:
        bg     = "#0d0d12"
        ring   = "#5a5a64"
        label  = "#eaeaef"
        arrow  = "#ff7e3a"
        cardc  = "#ffd23f"
    else:
        bg     = "#ffffff"
        ring   = "#cccccc"
        label  = "#1a1a1a"
        arrow  = "#0a7a3e"
        cardc  = "#1a1a1a"

    cx = cy = size / 2
    r = size / 2 - 22

    # Hacia dónde sopla = opuesto a "desde dónde viene"
    sopla = (direccion_meteo + 180.0) % 360.0
    rad = _m.radians(sopla)
    # SVG coords: x = cx + r·sin(rad),  y = cy − r·cos(rad)   (Y crece hacia abajo)
    ex = cx + r * 0.78 * _m.sin(rad)
    ey = cy - r * 0.78 * _m.cos(rad)
    sx = cx - r * 0.48 * _m.sin(rad)
    sy = cy + r * 0.48 * _m.cos(rad)

    # Etiquetas cardinales
    cardinal_svg = ""
    for ang_t, lbl in [(0, "N"), (90, "E"), (180, "S"), (270, "O")]:
        ar = _m.radians(ang_t)
        lx = cx + (r + 11) * _m.sin(ar)
        ly = cy - (r + 11) * _m.cos(ar) + 5
        cardinal_svg += (
            f'<text x="{lx:.1f}" y="{ly:.1f}" text-anchor="middle" '
            f'font-family="sans-serif" font-size="13" font-weight="700" '
            f'fill="{cardc}">{lbl}</text>'
        )

    # Marcas pequeñas para diagonales
    tick_svg = ""
    for ang_t in (45, 135, 225, 315):
        ar = _m.radians(ang_t)
        x1 = cx + (r - 3) * _m.sin(ar)
        y1 = cy - (r - 3) * _m.cos(ar)
        x2 = cx + (r + 2) * _m.sin(ar)
        y2 = cy - (r + 2) * _m.cos(ar)
        tick_svg += (
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{ring}" stroke-width="1.5"/>'
        )

    marker_id = f"ar_{idx}_{int(direccion_meteo) % 360}"

    return (
        f'<svg width="{size}" height="{size}" '
        f'xmlns="http://www.w3.org/2000/svg" '
        f'style="display:block;margin:auto;">'
        f'<defs>'
        f'  <marker id="{marker_id}" markerWidth="13" markerHeight="13" '
        f'          refX="11" refY="6.5" orient="auto-start-reverse">'
        f'    <path d="M0,0 L13,6.5 L0,13 L3,6.5 Z" fill="{arrow}"/>'
        f'  </marker>'
        f'</defs>'
        f'<rect width="100%" height="100%" fill="{bg}" rx="12"/>'
        f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" '
        f'        stroke="{ring}" stroke-width="2"/>'
        f'<circle cx="{cx}" cy="{cy}" r="3" fill="{ring}"/>'
        f'{tick_svg}{cardinal_svg}'
        f'<line x1="{sx:.1f}" y1="{sy:.1f}" x2="{ex:.1f}" y2="{ey:.1f}" '
        f'      stroke="{arrow}" stroke-width="5" stroke-linecap="round" '
        f'      marker-end="url(#{marker_id})"/>'
        f'</svg>'
    )


def mostrar_brujula(direccion_meteo: float, velocidad_ms: float | None = None,
                    size: int = 170, alto_contraste: bool = False,
                    idx: int = 0, leyenda: bool = True) -> None:
    """
    Muestra una brújula de solo lectura con etiqueta debajo.

    Útil para los modos en vivo y pronóstico: visualiza claramente hacia
    dónde se mueve el aire (y por tanto la pluma), sin pedirle al usuario
    que interprete números.
    """
    from weather import grados_a_cardinal as _gac
    st.markdown(
        _brujula_svg(direccion_meteo, velocidad_ms=velocidad_ms,
                     size=size, alto_contraste=alto_contraste, idx=idx),
        unsafe_allow_html=True,
    )
    if leyenda:
        cardinal_de = _gac(direccion_meteo)
        cardinal_a = _gac((direccion_meteo + 180) % 360)
        vel_txt = f"a {velocidad_ms:.1f} m/s" if velocidad_ms is not None else ""
        st.markdown(
            f"<div style='text-align:center; font-size:13px; margin-top:-6px;'>"
            f"viene del <b>{cardinal_de}</b> ({direccion_meteo:.0f}°) "
            f"→ sopla al <b>{cardinal_a}</b> {vel_txt}"
            f"</div>",
            unsafe_allow_html=True,
        )


def selector_direccion_viento(default_deg: int = 90, key: str = "viento_dir",
                              alto_contraste: bool = False) -> int:
    """
    Selector interactivo de dirección del viento.

    Muestra una brújula SVG arriba y debajo una rejilla 3×3 de botones que
    permiten elegir desde dónde viene el viento sin tocar números.

    Args:
        default_deg: dirección por defecto (sólo se aplica en la primera
                     ejecución).
        key: identificador único para el estado en session_state.

    Returns:
        Dirección actual en grados (0-359).
    """
    state_key = f"{key}_dir_state"
    if state_key not in st.session_state:
        st.session_state[state_key] = default_deg

    direccion = int(st.session_state[state_key])

    # Brújula visual arriba
    mostrar_brujula(direccion, velocidad_ms=None, size=170,
                    alto_contraste=alto_contraste, idx=hash(key) & 0xffff,
                    leyenda=True)

    # Rejilla 3×3 de botones cardinales.
    # IMPORTANTE: el LABEL (NO/N/NE/...) indica DE DÓNDE viene el viento
    # (convención meteorológica estándar). La FLECHA, en cambio, apunta hacia
    # donde se mueve la pluma — es decir, en dirección opuesta al label.
    # Así el usuario ve visualmente a dónde se desplazará el contaminante.
    direcciones_filas = [
        [("↘ NO", 315), ("↓ N", 0),   ("↙ NE", 45)],
        [("→ O", 270),  None,          ("← E", 90)],
        [("↗ SO", 225), ("↑ S", 180),  ("↖ SE", 135)],
    ]
    for fila in direcciones_filas:
        cols = st.columns(3)
        for i, item in enumerate(fila):
            with cols[i]:
                if item is None:
                    st.markdown(
                        f"<div style='text-align:center; color:#888; "
                        f"font-size:11px; padding:8px 0;'>"
                        f"{direccion}°</div>",
                        unsafe_allow_html=True,
                    )
                    continue
                lbl, ang = item
                # Detectar dirección activa (tolerancia ±22.5°)
                delta = abs(((direccion - ang + 180) % 360) - 180)
                es_activo = delta < 22.5
                btn_lbl = f"**{lbl}**" if es_activo else lbl
                if st.button(btn_lbl, key=f"{key}_btn_{ang}", width="stretch"):
                    st.session_state[state_key] = ang
                    st.rerun()

    with st.expander("Ajuste por grados"):
        st.slider("Dirección", 0, 359, key=state_key)

    return int(st.session_state[state_key])


# =====================================================================
# HEATMAP RGBA Y MAPAS
# =====================================================================

def ica_a_rgba(ica_matrix, mask=None, sigma=1.8, upscale=6,
               alto_contraste: bool = False):
    """
    Convierte una matriz de ICA en una imagen RGBA suave tipo "mancha de calor".

    - Suaviza el campo (blur gaussiano) → manchas orgánicas, no cuadrícula.
    - Aplica la paleta vívida continua.
    - El canal alfa crece con el ICA: aire limpio = transparente,
      contaminación alta = opaca e intensa.
    - Re-escala con interpolación bicúbica para un acabado liso.

    Returns:
        Array RGBA uint8 (alto*upscale, ancho*upscale, 4).
    """
    campo = np.asarray(ica_matrix, dtype=np.float32)
    if mask is not None:
        # Anular fuera del dominio antes de suavizar para no "sangrar" color.
        campo = np.where(mask, campo, 0.0)

    campo_suave = _suavizar_campo(campo, sigma=sigma)

    cmap = _cmap_para(alto_contraste)
    norm = mcolors.Normalize(vmin=0, vmax=ICA_MAX_PLOT)
    rgba = cmap(norm(campo_suave))  # (filas, cols, 4) en [0,1]

    # Canal alfa: curva suave. ICA~0 → transparente; ICA≳65 → casi opaco.
    # En alto contraste subimos la opacidad máxima para que las "manchas"
    # destaquen sobre el mapa oscuro.
    a = np.clip(campo_suave / 60.0, 0.0, 1.0) ** 0.80
    alpha_max = 0.95 if alto_contraste else 0.85
    rgba[..., 3] = a * alpha_max

    if mask is not None:
        # Borde suave del polígono (vignette) en lugar de un corte duro.
        mask_suave = _suavizar_campo(mask.astype(np.float32), sigma=1.2)
        rgba[..., 3] *= np.clip(mask_suave, 0.0, 1.0)

    rgba8 = (np.clip(rgba, 0.0, 1.0) * 255).astype(np.uint8)

    # Upscale para acabado liso (la malla original es de solo ~120×136).
    if upscale and upscale > 1:
        img = Image.fromarray(rgba8, mode="RGBA")
        img = img.resize(
            (rgba8.shape[1] * upscale, rgba8.shape[0] * upscale),
            resample=Image.BICUBIC,
        )
        rgba8 = np.array(img)

    return rgba8


# ---------------------------------------------------------------------------
# COLORES Y ETIQUETAS PARA LAS CAPAS DE USO DE SUELO
# ---------------------------------------------------------------------------

_LUSE_STYLE = {
    LUSE_VACIO:           {"color": "#888888", "label": "Vacío / asfalto",    },
    LUSE_ZONA_ARBOLADA:   {"color": "#2ecc71", "label": "Arbolado / parque",   },
    LUSE_EDIFICIO_DENSO:  {"color": "#e74c3c", "label": "Edificación densa",   },
    LUSE_ZONA_ABIERTA:    {"color": "#f39c12", "label": "Zona abierta",        },
    LUSE_ZONA_AGUA:       {"color": "#3498db", "label": "Agua",                },
    LUSE_ZONA_INDUSTRIAL: {"color": "#8e44ad", "label": "Industrial",          },
}


def _landuse_rgba(landuse_map: np.ndarray, mask: np.ndarray,
                  upscale: int = 6) -> np.ndarray:
    """
    Convierte la matriz landuse_map en una imagen RGBA para overlay en Folium.
    Cada tipo de celda recibe su color semisólido; fuera de la máscara es transparente.
    """
    h, w = landuse_map.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    for luse_type, style in _LUSE_STYLE.items():
        if luse_type == LUSE_VACIO:
            continue  # vacío queda transparente (no distraye)
        m = (landuse_map == luse_type) & mask
        color = mcolors.to_rgba(style["color"])
        rgba[m] = [int(color[0]*255), int(color[1]*255),
                   int(color[2]*255), 160]  # 160/255 ≈ 63% opacidad
    # Aplicar máscara del polígono
    rgba[~mask, 3] = 0
    img = Image.fromarray(rgba, mode="RGBA")
    if upscale > 1:
        img = img.resize((w * upscale, h * upscale), resample=Image.NEAREST)
    return np.array(img)


def _agregar_landuse_overlay(mapa: folium.Map, grid: dict, mask: np.ndarray,
                              landuse_map: np.ndarray) -> None:
    """
    Añade una capa de uso de suelo semi-transparente sobre el mapa Folium.
    Los colores representan el tipo de celda que afecta la fricción del viento.
    """
    rgba8 = _landuse_rgba(landuse_map, mask)
    rgba_flip = np.flipud(rgba8)
    img = Image.fromarray(rgba_flip, mode="RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("ascii")
    data_url = f"data:image/png;base64,{img_b64}"

    b = grid["bounds"]
    half_lat = (b["max_lat"] - b["min_lat"]) / max(1, grid["filas"] - 1) / 2
    half_lon = (b["max_lon"] - b["min_lon"]) / max(1, grid["columnas"] - 1) / 2
    bounds_overlay = [
        [b["min_lat"] - half_lat, b["min_lon"] - half_lon],
        [b["max_lat"] + half_lat, b["max_lon"] + half_lon],
    ]
    folium.raster_layers.ImageOverlay(
        image=data_url,
        bounds=bounds_overlay,
        opacity=1.0,          # el alfa ya va en la imagen
        interactive=False,
        name="Uso de suelo",
        show=False,
    ).add_to(mapa)


def construir_mapa(grid, mask, inf, ica_matrix, roads=None, ruta_puntos=None,
                   ruta_limpia=None, contaminante="PM2.5", usar_osm=False,
                   alto_contraste: bool = False, landuse_map=None):
    """Construye el mapa Folium con todas las capas."""
    tiles = "CartoDB dark_matter" if alto_contraste else "CartoDB Positron"
    mapa = folium.Map(
        location=[CENTER_LAT, CENTER_LON],
        zoom_start=15,
        tiles=tiles,
        control_scale=True,
    )

    # --- Capa de Uso de Suelo (Opcional, se dibuja debajo del ICA para que no lo tape por completo) ---
    if landuse_map is not None:
        _agregar_landuse_overlay(mapa, grid, mask, landuse_map)

    # --- Heatmap de ICA (overlay de imagen) ---
    # La matriz está en orden (filas, cols) = (lat ascendente, lon ascendente)
    # Para folium ImageOverlay necesitamos la imagen con fila 0 = norte.
    img_rgba = ica_a_rgba(ica_matrix, mask=mask, alto_contraste=alto_contraste)
    # Voltear verticalmente porque PIL usa origen arriba-izquierda
    img_rgba_flip = np.flipud(img_rgba)
    img = Image.fromarray(img_rgba_flip, mode="RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    # folium requiere un data-URL (string), no bytes crudos
    img_b64 = base64.b64encode(buf.read()).decode("ascii")
    data_url = f"data:image/png;base64,{img_b64}"

    # Corrección de media celda: lat_grid/lon_grid son los CENTROS de las
    # celdas (linspace de min a max). El ImageOverlay estira la imagen entre
    # los bounds, así que hay que expandir medio paso de malla por lado para
    # que cada píxel quede centrado en su punto de malla.
    b = grid["bounds"]
    half_lat = (b["max_lat"] - b["min_lat"]) / max(1, grid["filas"] - 1) / 2
    half_lon = (b["max_lon"] - b["min_lon"]) / max(1, grid["columnas"] - 1) / 2
    bounds_overlay = [
        [b["min_lat"] - half_lat, b["min_lon"] - half_lon],
        [b["max_lat"] + half_lat, b["max_lon"] + half_lon],
    ]
    folium.raster_layers.ImageOverlay(
        image=data_url,
        bounds=bounds_overlay,
        opacity=0.85 if alto_contraste else 0.65,
        interactive=False,
        name=f"ICA {contaminante}",
    ).add_to(mapa)

    # Colores adaptados al modo (claros en alto contraste para que destaquen
    # sobre el mapa oscuro).
    if alto_contraste:
        col_poly  = "#80b3ff"
        col_via   = "#dddddd"
        col_ter   = "#ff5050"
        col_ter_fill = "#ff5050"
    else:
        col_poly  = "#1f4e79"
        col_via   = "#333333"
        col_ter   = "#8B0000"
        col_ter_fill = "#8B0000"

    # --- Polígono de Ciudad Universitaria ---
    folium.Polygon(
        locations=POLYGON_LIMITS,
        color=col_poly,
        weight=3,
        fill=False,
        popup="Ciudad Universitaria UANL",
        tooltip="Polígono de estudio — Ciudad Universitaria UANL",
    ).add_to(mapa)

    # --- Vialidades (líneas) ---
    # Se dibujan las MISMAS vialidades que alimentan el modelo de emisiones,
    # de modo que el mapa, la animación y la simulación sean consistentes.
    if roads is None:
        roads = []
    grupo_vias = folium.FeatureGroup(
        name="Vialidades" + (" (OSM)" if usar_osm else " (aprox.)")
    )
    # Grosor según jerarquía de la vía
    grosor_por_tipo = {
        "motorway": 5, "trunk": 5, "primary": 4,
        "secondary": 3, "tertiary": 2,
    }
    for road in roads:
        tipo = road.get("type", "unclassified")
        peso = grosor_por_tipo.get(tipo, 2)
        nombre = road.get("name", "vía sin nombre")
        folium.PolyLine(
            locations=road["coords"],
            color=col_via,
            weight=peso,
            opacity=0.85 if alto_contraste else 0.75,
            tooltip=f"{nombre} ({tipo})",
            popup=f"<b>{nombre}</b><br>Tipo OSM: {tipo}<br>"
                  f"Carriles: {road.get('lanes', '?')}",
        ).add_to(grupo_vias)
    grupo_vias.add_to(mapa)

    # --- Ternium (rectángulo) ---
    folium.Rectangle(
        bounds=[
            [TERNIUM_AREA["lat_range"][0], TERNIUM_AREA["lon_range"][0]],
            [TERNIUM_AREA["lat_range"][1], TERNIUM_AREA["lon_range"][1]],
        ],
        color=col_ter,
        fill=True,
        fill_color=col_ter_fill,
        fill_opacity=0.30 if alto_contraste else 0.25,
        weight=2,
        tooltip="Planta Ternium — fuente fija industrial",
        popup="<b>Planta Ternium</b><br>Fuente fija industrial "
              "(emisión continua 24/7)<br><i>Ubicación aproximada</i>",
    ).add_to(mapa)

    # --- Ruta corta (rojo) ---
    if ruta_puntos:
        coords = [index_to_coords(i, j, grid) for (i, j) in ruta_puntos]
        folium.PolyLine(
            locations=coords,
            color="#D62728",
            weight=4,
            opacity=0.9,
            dash_array="5,5",
            popup="Ruta más corta",
        ).add_to(mapa)
        folium.Marker(coords[0], icon=folium.Icon(color="green"),
                      popup="Inicio").add_to(mapa)
        folium.Marker(coords[-1], icon=folium.Icon(color="red"),
                      popup="Destino").add_to(mapa)

    # --- Ruta limpia (verde) ---
    if ruta_limpia:
        coords = [index_to_coords(i, j, grid) for (i, j) in ruta_limpia]
        folium.PolyLine(
            locations=coords,
            color="#2CA02C",
            weight=5,
            opacity=0.95,
            popup="Ruta de menor exposición",
        ).add_to(mapa)

    folium.LayerControl(collapsed=False).add_to(mapa)
    return mapa


def _render_frame_mpl(ica_matrix, t_segundos, grid, mask, roads,
                      contaminante, viento_dir, sigma=1.6,
                      alto_contraste: bool = False):
    """
    Renderiza un frame de la animación como imagen PNG usando matplotlib.

    Usa `imshow` con `extent` exacto y dibuja avenidas / Ternium / polígono
    en el MISMO sistema de coordenadas (lon, lat), por lo que todo queda
    matemáticamente alineado con el heatmap.

    El tamaño de figura y los ejes son fijos para que todos los frames
    tengan dimensiones idénticas → la reproducción no "salta".

    Si `alto_contraste=True`, usa fondo oscuro, colormap saturado y trazos
    en claros, para resaltar al máximo el movimiento de la pluma.
    """
    campo = np.where(mask, np.asarray(ica_matrix, dtype=np.float32), 0.0)
    campo_suave = _suavizar_campo(campo, sigma=sigma)
    z = np.where(mask, campo_suave, np.nan)

    b = grid["bounds"]
    half_lon = (b["max_lon"] - b["min_lon"]) / max(1, grid["columnas"] - 1) / 2
    half_lat = (b["max_lat"] - b["min_lat"]) / max(1, grid["filas"] - 1) / 2
    extent = [b["min_lon"] - half_lon, b["max_lon"] + half_lon,
              b["min_lat"] - half_lat, b["max_lat"] + half_lat]

    # Paleta y colores según modo
    if alto_contraste:
        fig_bg     = "#0d0d12"
        ax_bg      = "#16161e"
        text_c     = "#eaeaef"
        road_c     = "#dddddd"
        poly_c     = "#80b3ff"
        ter_c      = "#ff5050"
        ter_text   = "#ffa0a0"
    else:
        fig_bg     = "#ffffff"
        ax_bg      = "#ffffff"
        text_c     = "#1a1a1a"
        road_c     = "#1a1a1a"
        poly_c     = "#0d3b66"
        ter_c      = "#1a1a1a"
        ter_text   = "#1a1a1a"

    fig = plt.figure(figsize=(8.6, 6.6), dpi=92, facecolor=fig_bg)
    ax = fig.add_axes([0.10, 0.10, 0.76, 0.80], facecolor=ax_bg)
    cmap = _cmap_para(alto_contraste)

    im = ax.imshow(z, extent=extent, origin="lower", cmap=cmap,
                   vmin=0, vmax=ICA_MAX_PLOT, interpolation="bilinear")
    ax.set_aspect(111139.0 / 100300.0)

    # --- Avenidas principales ---
    primera = True
    for road in roads:
        if road["type"] not in ("trunk", "primary", "motorway"):
            continue
        ax.plot(
            [c[1] for c in road["coords"]],
            [c[0] for c in road["coords"]],
            color=road_c, lw=1.8,
            alpha=0.85 if alto_contraste else 0.75,
            solid_capstyle="round",
            label="Avenidas principales" if primera else None,
        )
        primera = False

    # --- Polígono de Ciudad Universitaria ---
    poly = [(c[1], c[0]) for c in POLYGON_LIMITS]
    ax.add_patch(MplPolygon(poly, closed=True, fill=False,
                            edgecolor=poly_c, lw=2.0, ls=(0, (4, 2)),
                            label="Polígono CU"))

    # --- Ternium (fuente fija) ---
    tlon0, tlon1 = TERNIUM_AREA["lon_range"]
    tlat0, tlat1 = TERNIUM_AREA["lat_range"]
    ax.add_patch(Rectangle((tlon0, tlat0), tlon1 - tlon0, tlat1 - tlat0,
                           fill=False, edgecolor=ter_c, lw=2.0,
                           label="Ternium (fuente fija)"))
    ax.annotate("Ternium", ((tlon0 + tlon1) / 2, tlat1),
                ha="center", va="bottom", fontsize=9, weight="bold",
                color=ter_text)

    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_xlabel("Longitud", fontsize=9, color=text_c)
    ax.set_ylabel("Latitud", fontsize=9, color=text_c)
    ax.tick_params(labelsize=7, colors=text_c)
    for spine in ax.spines.values():
        spine.set_color(text_c)
    ax.set_title(
        f"Pluma de {contaminante} — viento desde {viento_dir:.0f}°\n"
        f"t = {t_segundos:.0f} s  ({t_segundos/60:.1f} min)",
        fontsize=10, color=text_c,
    )
    leg = ax.legend(loc="upper left", fontsize=7,
                    framealpha=0.85 if not alto_contraste else 0.70,
                    facecolor=fig_bg, edgecolor=text_c)
    for txt in leg.get_texts():
        txt.set_color(text_c)

    # Colorbar en posición fija
    cax = fig.add_axes([0.885, 0.10, 0.025, 0.80])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("ICA", fontsize=9, color=text_c)
    cbar.ax.tick_params(labelsize=7, colors=text_c)
    cbar.outline.set_edgecolor(text_c)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=fig_bg)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()



# =====================================================================
# UI: SIDEBAR
# =====================================================================

st.title(f"{PROJECT_NAME} — Simulador de Calidad del Aire — Ciudad Universitaria UANL")
st.caption(
    "Motor de dispersión advección-difusión 2D · Datos Open-Meteo · "
    "ICA según NOM-172-SEMARNAT-2019 · Equipo 11 · Brigada 003"
)

# ---- CSS global: tamaño de letra de títulos de expander ----
st.markdown(
    """
    <style>
    /* Título de los st.expander */
    [data-testid="stExpander"] details summary p {
        font-size: 1.4rem !important;   
        font-weight: 700 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Modos disponibles: identifican claramente qué se está mostrando.
MODO_AHORA      = "Tiempo real (ahora)"
MODO_PRONOSTICO = "Pronóstico (próximas horas)"
MODO_ESCENARIO  = "Escenario hipotético"

# Estado global de visualización (persiste entre reruns).
if "alto_contraste" not in st.session_state:
    st.session_state["alto_contraste"] = False

with st.sidebar:
    st.header("Panel de control")

    # ---- Toggle de alto contraste (siempre arriba y visible) ----
    # Usa key="alto_contraste" para que SU VALOR SE PERSISTA en
    # st.session_state entre reruns. Mover otros parámetros (sliders,
    # botones de la brújula, etc.) NO afecta este toggle: solo se apaga
    # cuando el usuario lo apaga explícitamente.
    st.toggle(
        "Modo alto contraste",
        key="alto_contraste",
        help="Oscurece el mapa y satura los colores: las zonas con "
             "contaminación destacan dramáticamente sobre el fondo, "
             "facilitando ver el movimiento de la pluma con el viento.",
    )
    alto_contraste = st.session_state["alto_contraste"]

    st.markdown("---")

    # ---- Selector de modo ----
    modo = st.radio(
        "Modo de simulación",
        [MODO_AHORA, MODO_PRONOSTICO, MODO_ESCENARIO],
        index=0,
        help=(
            "**Tiempo real**: usa el clima actual y la hora actual "
            "(Open-Meteo).\n\n"
            "**Pronóstico**: clima previsto para una hora futura "
            "(hasta 12 h adelante).\n\n"
            "**Escenario**: condiciones manuales para experimentación."
        ),
    )
    es_modo_real = modo == MODO_AHORA
    es_modo_pronostico = modo == MODO_PRONOSTICO
    es_modo_escenario = modo == MODO_ESCENARIO

    # ---- Contaminante (común a todos los modos) ----
    contaminante = st.selectbox(
        "Contaminante",
        ["PM2.5 (Material Particulado fino)", "PM10 (Material Particulado respirable)", "NOx (Óxidos de Nitrógeno)", "SO2 (Dióxido de Azufre)"],
        help="Cada contaminante usa sus factores de emisión y cortes ICA "
             "según norma mexicana.",
    )

    # ----------------------------------------------------------------
    # MODO 1: TIEMPO REAL  (clima + hora actuales, factores reales)
    # ----------------------------------------------------------------
    if es_modo_real:
        clima = get_current_weather()
        hora_pron_etiqueta = "Ahora"
        st.markdown("### Condiciones meteorológicas actuales")
        st.markdown(
            f"Fuente: _{clima['fuente']}_  \n"
            f"Temperatura: **{clima['temperatura']:.1f} °C**  ·  "
            f"Humedad: {clima['humedad']:.0f}% HR  \n"
            f"Presión: **{clima['presion']:.0f} hPa**"
        )
        # Brújula visual del viento (en lugar de números crudos)
        mostrar_brujula(
            clima["direccion_viento"], clima["velocidad_viento"],
            size=150, alto_contraste=st.session_state["alto_contraste"],
            idx=1, leyenda=True,
        )
        viento_ms = clima["velocidad_viento"]
        viento_dir = clima["direccion_viento"]
        temperatura = clima["temperatura"]
        presion = clima["presion"]
        hora = clima["hora"]
        # Factores reales: traffic factor base 1.0, congestion vendrá de TomTom
        factor_trafico = 1.0
        factor_industrial = 1.0
        # Día laboral se infiere de la fecha real (lun=0 ... dom=6)
        es_dia_laboral = datetime.now().weekday() < 5
        # Inversión térmica: heurística automática (T<10°C, viento<2, P>1018)
        inversion = (temperatura < 10 and viento_ms < 2.0 and presion > 1018)
        if inversion:
            st.warning("Condiciones de **inversión térmica** detectadas "
                       "automáticamente (T baja, viento débil, presión alta).")

    # ----------------------------------------------------------------
    # MODO 2: PRONOSTICO  (hora futura con clima previsto)
    # ----------------------------------------------------------------
    elif es_modo_pronostico:
        pron = get_hourly_forecast(horas=13)
        # Slider sobre 0-12 horas adelante
        idx_pron = st.slider(
            "Horas hacia el futuro",
            0, min(12, len(pron) - 1),
            value=1,
            help="Cuántas horas adelante quieres ver. 0 = próxima hora.",
        )
        p = pron[idx_pron]
        hora_pron_etiqueta = p["datetime"].strftime("%a %d %b %H:00")
        st.markdown("### Clima previsto")
        st.markdown(
            f"Hora: **{hora_pron_etiqueta}**  \n"
            f"Temperatura: **{p['temperatura']:.1f} °C**  \n"
            f"Presión: **{p['presion']:.0f} hPa**"
        )
        # Brújula visual del viento previsto
        mostrar_brujula(
            p["direccion_viento"], p["velocidad_viento"],
            size=150, alto_contraste=st.session_state["alto_contraste"],
            idx=2, leyenda=True,
        )
        viento_ms = p["velocidad_viento"]
        viento_dir = p["direccion_viento"]
        temperatura = p["temperatura"]
        presion = p["presion"]
        hora = p["hora"]
        factor_trafico = 1.0
        factor_industrial = 1.0
        es_dia_laboral = p["datetime"].weekday() < 5
        inversion = (temperatura < 10 and viento_ms < 2.0 and presion > 1018)
        if inversion:
            st.warning("Inversión térmica esperada a esa hora.")

    # ----------------------------------------------------------------
    # MODO 3: ESCENARIO  (todo manual)
    # ----------------------------------------------------------------
    else:
        hora_pron_etiqueta = None
        with st.expander("Meteorología manual", expanded=True):
            viento_ms = st.slider("Velocidad del viento (m/s)",
                                  0.5, 12.0, 3.0, 0.1, key="esc_viento_ms")
            st.markdown("**Dirección del viento** (desde dónde sopla):")
            # Selector visual con brújula + botones cardinales
            viento_dir = selector_direccion_viento(
                default_deg=90, key="esc_viento",
                alto_contraste=st.session_state["alto_contraste"],
            )
            temperatura = st.slider("Temperatura (°C)", -5, 45, 22, key="esc_temp")
            presion = st.slider("Presión (hPa)", 990, 1030, 1013, key="esc_presion")

        with st.expander("Hora y tráfico", expanded=True):
            hora = st.slider("Hora del día", 0, 23, datetime.now().hour,
                             help="Determina el patrón de tráfico (horas pico).", key="esc_hora")
            es_dia_laboral = st.checkbox(
                "Día laboral (lun-vie)", value=True,
                help="Fines de semana: flujo vehicular ~40% menor.", key="esc_dia_laboral")
            factor_trafico = st.slider(
                "Factor de tráfico", 0.0, 2.0, 1.0, 0.1,
                help="1.0 = normal. 0.7 = -30% (hoy no circula). 1.5 = +50%.", key="esc_f_trafico")
            factor_industrial = st.slider(
                "Factor industrial (Ternium)", 0.0, 2.0, 1.0, 0.1, key="esc_f_ind")
            inversion = st.checkbox(
                "Inversión térmica", value=False,
                help="Reduce mezcla vertical y concentra contaminantes.", key="esc_inv")

        def set_preset(preset_name):
            if preset_name == "normal":
                st.session_state.update({"esc_hora": 14, "esc_viento_ms": 4.0, "esc_viento_dir_state": 90, "esc_temp": 25, "esc_presion": 1013, "esc_f_trafico": 1.0, "esc_f_ind": 1.0, "esc_inv": False})
            elif preset_name == "pico":
                st.session_state.update({"esc_hora": 8, "esc_viento_ms": 2.0, "esc_viento_dir_state": 90, "esc_temp": 18, "esc_presion": 1015, "esc_f_trafico": 1.3, "esc_f_ind": 1.0, "esc_inv": False})
            elif preset_name == "inversion":
                st.session_state.update({"esc_hora": 7, "esc_viento_ms": 1.2, "esc_viento_dir_state": 0, "esc_temp": 4, "esc_presion": 1022, "esc_f_trafico": 1.0, "esc_f_ind": 1.0, "esc_inv": True})
            elif preset_name == "ternium":
                st.session_state.update({"esc_hora": 12, "esc_viento_ms": 4.0, "esc_viento_dir_state": 45, "esc_temp": 22, "esc_presion": 1013, "esc_f_trafico": 1.0, "esc_f_ind": 1.7, "esc_inv": False})
            st.session_state["last_preset"] = preset_name

        # Escenarios rápidos solo en modo escenario
        with st.expander("Escenarios predefinidos"):
            c1, c2 = st.columns(2)
            c1.button("Día normal", width="stretch", on_click=set_preset, args=("normal",))
            c2.button("Hora pico", width="stretch", on_click=set_preset, args=("pico",))
            c1.button("Inversión térmica", width="stretch", on_click=set_preset, args=("inversion",))
            c2.button("Viento hacia Ternium", width="stretch", on_click=set_preset, args=("ternium",))

    # ----------------------------------------------------------------
    # AJUSTES COMUNES (red vial, TomTom, visualización)
    # ----------------------------------------------------------------
    st.markdown("---")
    with st.expander("Red vial y tráfico", expanded=False):
        usar_osm = st.checkbox(
            "Usar red vial real (OpenStreetMap)",
            value=True,    # ← ahora ENCENDIDO por defecto
            help="Consulta Overpass API para la red real del área. "
                 "Si falla, usa la red de respaldo.",
        )
        tomtom_key = st.text_input(
            "Clave TomTom (opcional)",
            value="", type="password",
            help="Con clave gratuita de TomTom (developer.tomtom.com), el "
                 "simulador ajusta emisiones por congestión real.",
        )

    # Consultar congestión real (siempre)
    congestion = indice_congestion_zona(tomtom_key)
    if congestion["disponible"]:
        st.sidebar.success(f"{congestion['mensaje']}")
        factor_congestion = congestion["multiplicador_emision"]
    else:
        if tomtom_key:
            st.sidebar.warning(f"{congestion['mensaje']}")
        factor_congestion = 1.0


# ----------------------------------------------------------------
# Mostrar mensaje si se acaba de aplicar un preset
# ----------------------------------------------------------------
last_preset = st.session_state.pop("last_preset", None)
if last_preset:
    st.sidebar.success(f"Preset aplicado: **{last_preset}**.")


# Factor de tráfico efectivo = control manual × congestión real (TomTom)
factor_trafico_efectivo = factor_trafico * factor_congestion
if factor_congestion > 1.01:
    st.sidebar.caption(
        f"Factor de tráfico efectivo: {factor_trafico:.2f} × "
        f"{factor_congestion:.2f} = **{factor_trafico_efectivo:.2f}**"
    )


# =====================================================================
# EJECUTAR SIMULACION
# =====================================================================

grid, mask, inf = cargar_grid_y_infra()
roads = cargar_red_vial(usar_osm)

res = simular_snapshot(
    hora=hora,
    contaminante=contaminante,
    viento_ms=viento_ms,
    viento_dir=viento_dir,
    temperatura=temperatura,
    presion=presion,
    factor_trafico=factor_trafico_efectivo,
    factor_industrial=factor_industrial,
    inversion=inversion,
    es_dia_laboral=es_dia_laboral,
    usar_osm=usar_osm,
)

C = res["concentracion"]
A = res["ica"]
traffic_map = res["traffic"]

# Métricas dentro del polígono únicamente
A_dentro = A[mask]
C_dentro = C[mask]


# =====================================================================
# BANNER DE MODO + PANEL DE RECOMENDACIONES
# =====================================================================

# Color del banner según el modo
_modo_colores = {
    MODO_AHORA:      ("#0a7a3e", "rgba(10,122,62,0.10)"),   # verde
    MODO_PRONOSTICO: ("#1c5b9f", "rgba(28,91,159,0.10)"),   # azul
    MODO_ESCENARIO:  ("#a8551a", "rgba(168,85,26,0.10)"),   # naranja
}
_borde, _fondo = _modo_colores.get(modo, ("#666", "rgba(0,0,0,0.05)"))

_ts = datetime.now().strftime("%a %d %b · %H:%M")
if es_modo_real:
    _sub = f"hora actual: {hora:02d}:00 · datos meteorológicos en vivo"
elif es_modo_pronostico:
    _sub = (f"hora proyectada: {hora_pron_etiqueta} · "
            f"clima previsto por Open-Meteo")
else:
    _sub = (f"condiciones manuales · simulando hora {hora:02d}:00 · "
            f"factores ajustables")

st.markdown(
    f"""
    <div style="border-left: 5px solid {_borde}; background: {_fondo};
                padding: 12px 16px; border-radius: 6px;
                margin-bottom: 18px;">
      <div style="font-size: 16px; font-weight: 700; color: {_borde};">
          {modo}
      </div>
      <div style="font-size: 13px; color: #444; margin-top: 2px;">
          {_ts} · {_sub}
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# --- Alerta principal (color-coded por nivel ICA) ---
alert = generate_alert(A_dentro.max(), A_dentro.mean())
ica_max_val = float(A_dentro.max())
ica_med_val = float(A_dentro.mean())
ica_p95_val = float(np.percentile(A_dentro, 95))
cat_med, _ = categoria_ica(ica_med_val)

st.markdown(
    f"""
    <div style="background-color:{alert['color']}; padding:18px 24px;
                border-radius:10px; color:#1a1a1a; margin-bottom:10px;
                border: 2px solid rgba(0,0,0,0.35);">
        <div style="display:flex; justify-content:space-between; align-items:baseline;">
          <div style="font-size:22px; font-weight:800; display:flex; align-items:center; gap:8px;">
              {alert['icono']} {alert['nivel']} &nbsp;·&nbsp; ICA medio {ica_med_val:.0f} ({cat_med})
          </div>
          <div style="font-size:14px; opacity:0.85;">
              máx {ica_max_val:.0f} · p95 {ica_p95_val:.0f}
          </div>
        </div>
        <div style="font-size:14px; margin-top:6px; line-height:1.5;">
            {alert['mensaje']}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------
# Calcular pronóstico próximas horas (modos real y pronóstico)
# ----------------------------------------------------------------
pico_proximas = None
forecast_ica = []  # lista de (datetime, ica_max, categoria_color)
if es_modo_real or es_modo_pronostico:
    try:
        pron_meteo = get_hourly_forecast(horas=7)
        for p in pron_meteo:
            r_p = simular_snapshot(
                hora=p["hora"], contaminante=contaminante,
                viento_ms=p["velocidad_viento"],
                viento_dir=p["direccion_viento"],
                temperatura=p["temperatura"],
                presion=p["presion"],
                factor_trafico=factor_trafico_efectivo,
                factor_industrial=factor_industrial,
                inversion=(p["temperatura"] < 10 and p["velocidad_viento"] < 2.0
                           and p["presion"] > 1018),
                es_dia_laboral=p["datetime"].weekday() < 5,
                usar_osm=usar_osm,
            )
            ica_p = float(r_p["ica"][mask].max())
            forecast_ica.append({
                "datetime": p["datetime"],
                "hora": p["hora"],
                "ica_max": ica_p,
            })
        # Pico futuro
        if forecast_ica:
            pico = max(forecast_ica, key=lambda x: x["ica_max"])
            pico_proximas = {"hora_pico": pico["hora"],
                             "ica_pico": pico["ica_max"]}
    except Exception:
        forecast_ica = []


# ----------------------------------------------------------------
# Recomendaciones de acción (basadas en estado actual + futuro)
# ----------------------------------------------------------------
perfil_pct = HOURLY_PROFILE.get(hora, 0.5) * 100.0 * (0.6 if not es_dia_laboral else 1.0)
factores = factores_ambientales(
    viento_ms, viento_dir, temperatura, presion,
    hora, perfil_pct, factor_trafico_efectivo, inversion,
)
acciones = recomendaciones_de_accion(
    ica_max_val, ica_med_val, factores, pico_proximas
)

# Mostrar acciones recomendadas
with st.expander("**¿Qué hacer ahora?**", expanded=True):
    for a in acciones:
        st.markdown(f"{a}", unsafe_allow_html=True)
    st.markdown(f"**Cubrebocas**: {mask_recommendation(ica_max_val)}", unsafe_allow_html=True)


# ----------------------------------------------------------------
# Panel: factores ambientales + pronóstico próximas horas
# ----------------------------------------------------------------
with st.expander("**Explicación de las condiciones**", expanded=True):
    col_fac, col_pron = st.columns([3, 2])

    with col_fac:
        st.markdown("##### Factores ambientales")
        _color_imp = {
            "bueno":   "#1a9850",
            "neutro":  "#777",
            "malo":    "#f46d43",
            "crítico": "#a50026",
        }
        for f in factores:
            c = _color_imp[f["impacto"]]
            st.markdown(
                f"""
                <div style="border-left: 4px solid {c}; padding: 6px 10px;
                            margin-bottom: 6px; background: rgba(0,0,0,0.025);">
                  <span style="font-size: 14px; display:flex; align-items:center;">
                    {f['icono']} <b>{f['etiqueta']}</b>:&nbsp;
                    <span style="color:{c};"><b>{f['valor']}</b></span>
                  </span>
                  <div style="font-size: 12px; color: #555; margin-top: 2px;">
                    {f['mensaje']}
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with col_pron:
        if forecast_ica:
            st.markdown("##### Próximas horas")
            for f in forecast_ica:
                cat_f, color_f = categoria_ica(f["ica_max"])
                etiqueta = f["datetime"].strftime("%H:00")
                # Barra horizontal proporcional al ICA (escala 0-200)
                ancho = min(100, int(f["ica_max"] / 200 * 100))
                st.markdown(
                    f"""
                    <div style="margin-bottom: 5px;">
                      <div style="display:flex; justify-content:space-between;
                                  font-size:12px; color:#333;">
                        <span><b>{etiqueta}</b></span>
                        <span><b>{f['ica_max']:.0f}</b> {cat_f}</span>
                      </div>
                      <div style="background:#e9ecef; border-radius:3px;
                                  height:6px; overflow:hidden;">
                        <div style="width:{ancho}%; height:100%; background:{color_f};"></div>
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        elif es_modo_escenario:
            st.markdown("##### Próximas horas")
            st.caption("_Disponible en modos Tiempo real y Pronóstico._")
        else:
            st.markdown("##### Próximas horas")
            st.caption("_No hay datos de pronóstico disponibles._")


# =====================================================================
# METRICAS DETALLADAS
# =====================================================================

with st.expander("**Métricas detalladas de la simulación actual**", expanded=True):
    cat_max, _ = categoria_ica(ica_max_val)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("ICA máximo", f"{ica_max_val:.0f}", cat_max)
    m2.metric("ICA medio", f"{ica_med_val:.0f}", cat_med)
    m3.metric("ICA p95", f"{ica_p95_val:.0f}",
              help="95% del área tiene ICA por debajo de este valor")
    m4.metric(f"{contaminante} máx", f"{C_dentro.max():.1f} μg/m³")


# Tarjeta "Medido en vivo": solo en modos tiempo real / pronóstico,
# trae automáticamente la lectura modelada de Open-Meteo Air Quality
# (CAMS global) y la compara con lo que el simulador local predice.
if not es_modo_escenario:
    from weather import get_current_air_quality
    with st.spinner("Consultando Open-Meteo Air Quality..."):
        med = get_current_air_quality(lat=CENTER_LAT, lon=CENTER_LON)

    col_map_om = {"PM2.5": "pm2_5", "PM10": "pm10",
                  "NOx": "no2", "SO2": "so2"}
    valor_medido = med.get(col_map_om.get(contaminante, "pm2_5"))

    if med["ok"] and valor_medido is not None:
        # Calcular ICA simulado del centro del polígono para comparar
        from simulator import calculate_ica
        centro_i = grid["filas"] // 2
        centro_j = grid["columnas"] // 2
        # Tomar promedio de 5x5 alrededor del centro de CU
        sub_C = C_dentro[max(0, centro_i-2):centro_i+3,
                         max(0, centro_j-2):centro_j+3] \
                if C_dentro.ndim == 2 else None
        c_sim_centro = float(sub_C.mean()) if sub_C is not None else \
                       float(np.mean(C_dentro))
        ica_medido_array = calculate_ica(np.array([valor_medido]), contaminante)
        ica_medido = float(ica_medido_array[0])
        cat_med_om, _ = categoria_ica(ica_medido)

        diff_ugm3 = c_sim_centro - valor_medido
        diff_str = f"{diff_ugm3:+.1f} μg/m³ vs simulado"

        st.markdown("##### Comparación con lectura en vivo")
        live1, live2, live3, live4 = st.columns(4)
        live1.metric(
            f"{contaminante} medido",
            f"{valor_medido:.1f} μg/m³",
            help=f"Fuente: {med['fuente']} · "
                 f"hora: {med['fecha'] or 'reciente'}",
        )
        live2.metric(
            f"{contaminante} simulado (centro CU)",
            f"{c_sim_centro:.1f} μg/m³",
            diff_str,
            delta_color="normal",
        )
        live3.metric("ICA medido (NOM-172)",
                     f"{ica_medido:.0f}", cat_med_om)
        live4.metric("US AQI (Open-Meteo)",
                     f"{med['us_aqi']:.0f}" if med['us_aqi'] is not None else "—",
                     "EPA estándar")

        st.caption(
            f"Datos de **CAMS** vía Open-Meteo (modelo global ~45 km, "
            f"actualizado cada 12 h). Para comparar con otas fuentes"
            f"de datos, ve al tab **Validación SIMA** y usa "
            f"un CSV descargado de SINAICA."
        )
    elif not med["ok"]:
        st.caption(
            f"*Lectura en vivo de Open-Meteo Air Quality no disponible "
            f"({med['fuente']}). La simulación local sigue funcionando.*"
        )


# =====================================================================
# TABS: MAPA / PRONOSTICO / RUTAS
# =====================================================================

(tab_mapa, tab_animacion, tab_evolucion, tab_pronostico,
 tab_rutas, tab_validacion, tab_info) = st.tabs(
    ["Mapa actual", "Tendencia",
     "Día completo (24h)", "Próximas 12 horas",
     "Rutas seguras", "Validación SIMA",
     "Sobre el modelo"]
)

# ----- TAB 1: MAPA -----
with tab_mapa:
    st.markdown(f"#### Distribución espacial del {contaminante} sobre el polígono de CU")

    lm = cargar_landuse(usar_osm)
    mapa = construir_mapa(grid, mask, inf, A, roads=roads,
                          contaminante=contaminante, usar_osm=usar_osm,
                          alto_contraste=alto_contraste, landuse_map=lm)

    st_folium(mapa, width=None, height=580, returned_objects=[])

    with st.expander("Mostrar leyenda de uso de suelo", expanded=False):
        # Leyenda de uso de suelo (horizontal sin emojis)
        leyenda_items = "".join(
            f"""
            <div style="display:flex; align-items:center; gap:6px;
                        font-size:13px; margin-right: 15px; margin-bottom: 5px;">
              <span style="display:inline-block; width:14px; height:14px;
                           border-radius:3px; background:{s['color']};
                           flex-shrink:0;"></span>
              <b>{s['label']}</b>
            </div>
            """
            for t, s in _LUSE_STYLE.items() if t != LUSE_VACIO
        )
        st.markdown(
            f"""
            <div style="border:1px solid rgba(128,128,128,0.2); border-radius:8px;
                        padding:12px 16px; margin-top:6px; margin-bottom:12px;">
              <b style="font-size:14px;">Colores en la capa de uso de suelo</b>
              <div style="display:flex; flex-wrap:wrap; margin-top:8px;">
                {leyenda_items}
              </div>
              <div style="margin-top:4px; font-size:12px; color:#555;">
                Las celdas de <b>vías y vacío</b> no se colorean (transparentes). Activa la capa en el control superior derecho del mapa.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Resumen de celdas
        lm_resumen = resumen_landuse(lm)
        st.markdown("##### Distribución de tipos de celda")
        cols_r = st.columns(len(_LUSE_STYLE) - 1)  # sin VACIO
        idx_col = 0
        for t, s in _LUSE_STYLE.items():
            if t == LUSE_VACIO:
                continue
            data = lm_resumen.get(s["label"], {"celdas": 0, "pct": 0.0})
            cols_r[idx_col].metric(
                f"{s['label']}",
                f"{data['celdas']:,}"
            )
            idx_col += 1

    st.markdown(
        """
        **Leyenda ICA**:  
        0-50 Buena (verde) · 51-100 Aceptable (amarillo) · 101-150 Mala (naranja) ·
        151-200 Muy Mala (rojo) · 201-300 Extr. Mala (violeta) · 301+ Peligrosa (negro)
        """
    )

    # Información de la red vial usada
    info_red = resumen_red(roads)
    veh_total_actual = traffic_map.sum()
    cv1, cv2, cv3 = st.columns(3)
    cv1.metric("Vialidades cargadas", info_red["total_segmentos"],
               f"{info_red['km_totales']} km totales")
    cv2.metric("Flujo total (veh-celda/h)",
               f"{veh_total_actual/1000:.0f}k",
               f"Hora {hora}: perfil {HOURLY_PROFILE.get(hora, 0.5)*100:.0f}%")
    cv3.metric("Flujo máximo por celda",
               f"{traffic_map.max():.0f} veh/h",
               "OSM real" if usar_osm else "Red de respaldo")


# ----- TAB ANIMACION -----
with tab_animacion:
    st.markdown(
        """
        <div style="border-left: 5px solid #6c5ce7;
                    background: rgba(108,92,231,0.08);
                    padding: 10px 16px; border-radius: 6px;
                    margin-bottom: 16px;">
          <div style="font-size: 15px; font-weight: 700; color: #6c5ce7;">
              ILUSTRATIVO · transitorio físico
          </div>
          <div style="font-size: 13px; color: #444;">
              Parte de aire limpio y muestra cómo se construye y mueve la
              pluma con el viento. No es una predicción de momento concreto;
              es una visualización del comportamiento físico.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("#### Movimiento de la pluma de contaminación")
    st.caption(
        "La simulación parte de aire limpio y muestra cómo se emite el "
        "contaminante desde las vialidades y la industria, y cómo el viento "
        "lo transporta y dispersa."
    )

    col_an1, col_an2, col_an3 = st.columns(3)
    with col_an1:
        duracion_anim = st.select_slider(
            "Duración simulada",
            options=[300, 600, 900, 1200, 1800],
            value=600,
            format_func=lambda s: f"{s//60} min",
            help="Tiempo físico que cubre la animación. Con viento de 3 m/s, "
                 "10 min ≈ 1.8 km de transporte.",
        )
    with col_an2:
        n_frames_anim = st.select_slider(
            "Suavidad (frames)",
            options=[18, 24, 36, 48, 60],
            value=36,
            help="Más frames = animación más fluida (y más tarda en "
                 "pre-renderizar).",
        )
    with col_an3:
        hora_anim_val = st.slider("Hora a simular", 0, 23, hora,
                                  disabled=not es_modo_escenario,
                                  help="Determina el nivel de tráfico de partida." if es_modo_escenario else "Solo disponible en el modo Escenario hipotético")
        hora_anim = hora_anim_val if es_modo_escenario else hora

    anim = simular_animacion_cached(
        hora=hora_anim, contaminante=contaminante,
        viento_ms=viento_ms, viento_dir=viento_dir,
        temperatura=temperatura, presion=presion,
        factor_trafico=factor_trafico_efectivo,
        factor_industrial=factor_industrial,
        inversion=inversion, es_dia_laboral=es_dia_laboral,
        usar_osm=usar_osm,
        tiempo_simulado_s=duracion_anim, n_frames=n_frames_anim,
    )
    frames_an = anim["frames"]
    n_fr = len(frames_an)

    # Clave de identidad de esta animación: si cambia algún parámetro,
    # los frames pre-renderizados guardados dejan de ser válidos.
    anim_key = (
        hora_anim, contaminante, round(viento_ms, 2), round(viento_dir, 1),
        round(temperatura, 1), round(presion, 1),
        round(factor_trafico_efectivo, 3), round(factor_industrial, 3),
        inversion, es_dia_laboral, usar_osm, duracion_anim, n_frames_anim,
    )

    generar = st.button("Generar / actualizar animación", width="stretch",
                        type="primary")

    if generar:
        # Pre-renderizar TODOS los frames como PNG y guardarlos en sesión.
        barra = st.progress(0.0, "Pre-renderizando frames…")
        pngs = []
        for k, fr in enumerate(frames_an):
            pngs.append(_render_frame_mpl(
                fr["ica"], fr["t_segundos"], grid, mask, roads,
                contaminante, viento_dir,
                alto_contraste=alto_contraste,
            ))
            barra.progress((k + 1) / n_fr,
                           f"Pre-renderizando frame {k+1}/{n_fr}…")
        barra.empty()
        st.session_state["anim_frames_png"] = pngs
        st.session_state["anim_key"] = anim_key
        st.success(f"{n_fr} frames listos. Reproducción fluida activada.")

    frames_png = st.session_state.get("anim_frames_png")
    key_guardada = st.session_state.get("anim_key")

    if frames_png is None:
        st.info("Presiona **Generar animación** para pre-renderizar los "
                "frames. Una vez listos, podrás reproducirlos sin parpadeo.")
    else:
        desactualizada = (key_guardada != anim_key)
        if desactualizada:
            st.warning(
                "⚠️ Cambiaste parámetros desde la última generación. "
                "Se muestran los frames anteriores; presiona "
                "**Generar animación** para actualizar."
            )

        # --- Controles de reproducción ---
        cpa, cpb = st.columns([1, 3])
        with cpa:
            velocidad = st.select_slider(
                "Velocidad", options=["Lenta", "Normal", "Rápida"],
                value="Normal",
            )
        with cpb:
            idx_frame = st.slider(
                "Instante de la simulación (frame)",
                0, len(frames_png) - 1, 0,
                help="Arrastra para inspeccionar cualquier momento. "
                     "El frame 0 es aire limpio; el último es el estado final.",
            )

        reproducir = st.button("▶ Reproducir animación completa",
                               width="stretch")

        # Contenedor único: la reproducción solo intercambia la imagen
        # dentro de él (no remonta componentes → sin parpadeo).
        placeholder_anim = st.empty()
        pausa_s = {"Lenta": 0.30, "Normal": 0.13, "Rápida": 0.05}[velocidad]

        if reproducir:
            for png in frames_png:
                placeholder_anim.image(png, width="stretch")
                time.sleep(pausa_s)
            # Tras reproducir, dejar fijo el último frame
            placeholder_anim.image(frames_png[-1], width="stretch")
            idx_frame = len(frames_png) - 1
        else:
            placeholder_anim.image(frames_png[idx_frame], width="stretch")

        st.markdown(
            "Las **líneas negras** son las avenidas principales (`trunk` y "
            "`primary`); el **recuadro negro** es la planta Ternium. Las "
            "**manchas de color** marcan dónde se concentra el contaminante: "
            "verde = aire limpio, amarillo/naranja = moderado, rojo/violeta = "
            "alto."
        )

        # Métricas de evolución de la pluma
        ica_series = [f["ica"][mask].mean() for f in frames_an]
        ica_max_series = [f["ica"][mask].max() for f in frames_an]
        t_series = [f["t_segundos"] for f in frames_an]

        ica_final = ica_series[-1]
        t_90 = next((t for t, v in zip(t_series, ica_series)
                     if v >= 0.9 * ica_final and ica_final > 0), t_series[-1])

        idx_safe = min(idx_frame, len(frames_an) - 1)
        am1, am2, am3 = st.columns(3)
        am1.metric("ICA medio (frame actual)",
                   f"{frames_an[idx_safe]['ica'][mask].mean():.0f}",
                   f"final: {ica_final:.0f}")
        am2.metric("Tiempo a 90% saturación", f"{t_90:.0f} s",
                   f"{t_90/60:.1f} min")
        am3.metric("Frames generados", len(frames_png),
                   f"{duracion_anim//60} min simulados")

        # Curva de acumulación con marca del frame actual
        fig_acum = go.Figure()
        fig_acum.add_trace(go.Scatter(
            x=t_series, y=ica_max_series, name="ICA máximo",
            mode="lines", line=dict(color="#e63946", width=2),
        ))
        fig_acum.add_trace(go.Scatter(
            x=t_series, y=ica_series, name="ICA medio",
            mode="lines", line=dict(color="#00b894", width=2),
            fill="tozeroy", fillcolor="rgba(0,184,148,0.15)",
        ))
        fig_acum.add_vline(x=t_series[idx_safe], line_dash="solid",
                           line_color="#888", line_width=2,
                           annotation_text=f"{t_series[idx_safe]:.0f} s")
        fig_acum.update_layout(
            title="Acumulación de la contaminación durante la simulación",
            xaxis_title="Tiempo simulado (s)", yaxis_title="ICA",
            height=300, hovermode="x unified",
            margin=dict(l=40, r=20, t=50, b=40),
        )
        st.plotly_chart(fig_acum, width="stretch")


# ----- TAB 2: EVOLUCION 24h -----
with tab_evolucion:
    st.markdown(
        """
        <div style="border-left: 5px solid #a8551a;
                    background: rgba(168,85,26,0.08);
                    padding: 10px 16px; border-radius: 6px;
                    margin-bottom: 16px;">
          <div style="font-size: 15px; font-weight: 700; color: #a8551a;">
              EXPLORATORIO · día tipo (24h)
          </div>
          <div style="font-size: 13px; color: #444;">
              Simula un día completo bajo las condiciones meteorológicas
              configuradas. Útil para ver cómo el flujo vehicular construye
              el ciclo diario de contaminación.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("#### Cómo se mueve la contaminación con el flujo vehicular")
    st.caption(
        "Simulación 24h con arrastre temporal: el contaminante emitido a cada "
        "hora se transporta por el viento y persiste mientras la atmósfera "
        "no lo dispersa. Muestra cómo el pico vehicular hace subir el ICA y "
        "cómo cambia el patrón espacial con la dirección del viento."
    )

    # Configuración del escenario 24h
    col_e1, col_e2 = st.columns(2)
    with col_e1:
        usar_inv_madrugada = st.checkbox(
            "Inversión térmica de madrugada (5–9 AM)",
            value=False,
            help="Si está activa, las primeras horas del día concentran "
                 "el material por inhibición de mezcla vertical."
        )
    with col_e2:
        comparar_dia = st.radio(
            "Día tipo",
            ["Día laboral", "Fin de semana"],
            index=0 if es_dia_laboral else 1,
            horizontal=True,
        )

    inversion_horas = (5, 6, 7, 8, 9) if usar_inv_madrugada else ()

    with st.spinner("Simulando 24 horas (esto toma 1-3 segundos)…"):
        frames = simular_24h(
            contaminante=contaminante,
            viento_ms=viento_ms, viento_dir=viento_dir,
            temperatura=temperatura, presion=presion,
            factor_trafico=factor_trafico_efectivo,
            factor_industrial=factor_industrial,
            es_dia_laboral=(comparar_dia == "Día laboral"),
            inversion_horas_tuple=inversion_horas,
            usar_osm=usar_osm,
        )

    # Slider de hora para ver el frame
    hora_view = st.slider(
        "Hora del día para visualizar",
        0, 23, 8,
        help="Mueve el slider para ver cómo cambia el mapa de contaminación "
             "a lo largo del día.",
    )

    frame = frames[hora_view]
    A_frame = frame["ica"]
    A_in_frame = A_frame[mask]
    veh_frame = frame["traffic"].sum()

    # Métricas del frame
    cat_max_f, _ = categoria_ica(A_in_frame.max())
    fm1, fm2, fm3, fm4 = st.columns(4)
    fm1.metric("Hora", f"{hora_view:02d}:00")
    fm2.metric("Vehículos-celda/h", f"{veh_frame/1000:.0f}k",
               f"perfil {HOURLY_PROFILE[hora_view]*100:.0f}%")
    fm3.metric("ICA medio", f"{A_in_frame.mean():.0f}")
    fm4.metric("ICA máximo", f"{A_in_frame.max():.0f}", cat_max_f)

    # Mapa del frame
    mapa_evol = construir_mapa(grid, mask, inf, A_frame, roads=roads,
                               contaminante=contaminante, usar_osm=usar_osm,
                               alto_contraste=alto_contraste)
    st_folium(mapa_evol, width=None, height=460,
              returned_objects=[], key=f"map_h{hora_view}")

    # ---- Gráfico de evolución 24h ----
    with st.expander("**Evolución horaria del ICA y del flujo vehicular**", expanded=True):
        df_evol = pd.DataFrame([{
            "hora": f["hora"],
            "ica_medio": float(f["ica"][mask].mean()),
            "ica_max":   float(f["ica"][mask].max()),
            "ica_p95":   float(np.percentile(f["ica"][mask], 95)),
            "veh_total": float(f["traffic"].sum()) / 1000.0,  # miles
        } for f in frames])

        # Doble eje: ICA + flujo vehicular
        fig_evol = go.Figure()
        fig_evol.add_trace(go.Scatter(
            x=df_evol["hora"], y=df_evol["ica_max"], name="ICA máximo",
            mode="lines+markers", line=dict(color="#D62728", width=3),
        ))
        fig_evol.add_trace(go.Scatter(
            x=df_evol["hora"], y=df_evol["ica_p95"], name="ICA p95",
            mode="lines", line=dict(color="#FF7F0E", dash="dash"),
        ))
        fig_evol.add_trace(go.Scatter(
            x=df_evol["hora"], y=df_evol["ica_medio"], name="ICA medio",
            mode="lines+markers", line=dict(color="#2CA02C", width=2),
            fill="tozeroy", fillcolor="rgba(44,160,44,0.15)",
        ))
        fig_evol.add_trace(go.Scatter(
            x=df_evol["hora"], y=df_evol["veh_total"], name="Flujo (miles veh-celda/h)",
            mode="lines", line=dict(color="#1F77B4", width=2, dash="dot"),
            yaxis="y2", opacity=0.7,
        ))
        # marcar hora seleccionada
        fig_evol.add_vline(x=hora_view, line_dash="solid",
                           line_color="#888", line_width=2,
                           annotation_text=f"{hora_view}:00",
                           annotation_position="top")
        fig_evol.update_layout(
            height=380,
            xaxis_title="Hora del día",
            yaxis=dict(title="ICA", side="left"),
            yaxis2=dict(title="Flujo (miles veh-celda/h)", side="right",
                        overlaying="y", showgrid=False),
            hovermode="x unified",
            legend=dict(orientation="h", y=1.10),
        )
        st.plotly_chart(fig_evol, width="stretch")

        # ---- Insight automático ----
        hora_pico_ica = int(df_evol["ica_max"].idxmax())
        ica_min_h = int(df_evol["ica_max"].idxmin())
        delta = df_evol.loc[hora_pico_ica, "ica_max"] - df_evol.loc[ica_min_h, "ica_max"]
        st.info(
            f"**Análisis del día**: el ICA máximo se alcanza a las "
            f"**{hora_pico_ica:02d}:00** (ICA={df_evol.loc[hora_pico_ica, 'ica_max']:.0f}), "
            f"un aumento de **+{delta:.0f}** puntos respecto al mínimo de las "
            f"{ica_min_h:02d}:00. El flujo vehicular es el principal motor de "
            f"esta variación diaria."
        )

# ----- TAB 2: PRONOSTICO -----
with tab_pronostico:
    # Banner: este tab es SIEMPRE predictivo, independientemente del modo
    st.markdown(
        """
        <div style="border-left: 5px solid #1c5b9f;
                    background: rgba(28,91,159,0.08);
                    padding: 10px 16px; border-radius: 6px;
                    margin-bottom: 16px;">
          <div style="font-size: 15px; font-weight: 700; color: #1c5b9f;">
              PREDICCIÓN · próximas 12 horas
          </div>
          <div style="font-size: 13px; color: #444;">
              Combina el pronóstico meteorológico de Open-Meteo con el modelo
              de tráfico y emisiones para anticipar la calidad del aire.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not es_modo_escenario:
        pronostico_meteo = get_hourly_forecast(horas=12)

        def _sim(hora, viento, direccion, temperatura, presion):
            r = simular_snapshot(
                hora=hora,
                contaminante=contaminante,
                viento_ms=viento,
                viento_dir=direccion,
                temperatura=temperatura,
                presion=presion,
                factor_trafico=factor_trafico_efectivo,
                factor_industrial=factor_industrial,
                inversion=inversion,
                es_dia_laboral=es_dia_laboral,
                usar_osm=usar_osm,
            )
            return r["ica"]

        with st.spinner("Calculando 12 horas de pronóstico…"):
            forecast = hourly_pollution_forecast(_sim, pronostico_meteo,
                                                 contaminante=contaminante)
        picos = detectar_picos(forecast, umbral=100.0)

        # Resumen rápido en métricas
        ica_max_proximas = max(f["ica_max"] for f in forecast)
        hora_pico_pron = next((f for f in forecast
                               if f["ica_max"] == ica_max_proximas), None)
        ica_min_proximas = min(f["ica_max"] for f in forecast)

        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("ICA máximo previsto", f"{ica_max_proximas:.0f}",
                   hora_pico_pron["datetime"].strftime("a las %H:%M")
                   if hora_pico_pron else None)
        mc2.metric("ICA mínimo previsto", f"{ica_min_proximas:.0f}",
                   "mejor momento")
        mc3.metric("Picos detectados", len(picos),
                   "ICA ≥ 100" if picos else "✅ sin picos")

        # Gráfico
        df = pd.DataFrame(forecast)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["datetime"], y=df["ica_max"],
                                 name="ICA máximo", mode="lines+markers",
                                 line=dict(color="#e63946", width=3)))
        fig.add_trace(go.Scatter(x=df["datetime"], y=df["ica_p95"],
                                 name="ICA p95", mode="lines",
                                 line=dict(color="#ff9f1c", dash="dash")))
        fig.add_trace(go.Scatter(x=df["datetime"], y=df["ica_medio"],
                                 name="ICA medio", mode="lines+markers",
                                 line=dict(color="#00b894", width=2),
                                 fill="tozeroy",
                                 fillcolor="rgba(0,184,148,0.12)"))
        for y, lbl, c in [(50, "Buena/Aceptable", "#999"),
                          (100, "Aceptable/Mala", "#ff9f1c"),
                          (150, "Mala/Muy Mala", "#e63946")]:
            fig.add_hline(y=y, line_dash="dot", line_color=c,
                          annotation_text=lbl, annotation_position="right")
        fig.update_layout(
            title=f"Pronóstico de ICA ({contaminante}) — próximas 12 horas",
            xaxis_title="Hora", yaxis_title="ICA",
            hovermode="x unified", height=380,
            margin=dict(l=40, r=20, t=50, b=40),
        )
        st.plotly_chart(fig, width="stretch")

        # --- Timeline de recomendaciones por hora ---
        st.markdown("##### Qué hacer hora por hora")
        recs_pron = recomendaciones_pronostico(forecast, umbral=100.0)

        # Mostrar como tabla con estilo + emojis para acciones
        st.markdown(
            "Recomendaciones para cada ventana horaria del pronóstico:"
        )
        for r in recs_pron:
            hora_fmt = r["datetime"].strftime("%a %H:%M")
            st.markdown(
                f"""
                <div style="display:flex; align-items:center; gap:14px;
                            padding: 6px 12px; margin: 4px 0;
                            border-left: 5px solid {r['color']};
                            background: rgba(0,0,0,0.025);
                            border-radius: 4px;">
                    <div style="min-width: 110px; font-weight: 600;
                                color: #333;">{hora_fmt}</div>
                    <div style="min-width: 130px;">
                        {r['icono']} <b>ICA {r['ica_max']:.0f}</b>
                        <span style="color:#666; font-size:12px;">
                          ({r['categoria']})
                        </span>
                    </div>
                    <div style="flex:1; color:#222;">{r['accion']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Picos resaltados
        if picos:
            st.markdown("##### Ventanas críticas detectadas")
            df_picos = pd.DataFrame([{
                "Hora": p["datetime"].strftime("%a %d %b · %H:%M"),
                "ICA máximo previsto": f"{p['ica_max']:.0f}",
                "Severidad": p["severidad"].upper(),
                "Recomendación": ("Quédate en interiores"
                                  if p["ica_max"] >= 150
                                  else "Limita actividad exterior"),
            } for p in picos])
            st.dataframe(df_picos, hide_index=True, width="stretch")
        else:
            st.success("No se proyectan episodios de contaminación elevada "
                       "en las próximas 12 horas.")
    else:
        st.info("El pronóstico horario se construye sobre las condiciones "
                "previstas por Open-Meteo, así que solo está disponible en los "
                "modos **Tiempo real** y **Pronóstico**. En modo "
                "**Escenario hipotético** el clima es fijo y no hay serie "
                "temporal que proyectar. Cambia el modo en el panel lateral.")

# ----- TAB 3: RUTAS -----
with tab_rutas:
    st.markdown("#### Encuentra la ruta de menor exposición a contaminantes")
    st.caption("Algoritmo: Dijkstra con costo ponderado por ICA local "
               "(cuadrático sobre nivel 100).")

    cr1, cr2 = st.columns(2)
    with cr1:
        st.markdown("**Punto de partida**")
        lat_o = st.number_input("Latitud (origen)", value=25.7212,
                                step=0.0005, format="%.4f")
        lon_o = st.number_input("Longitud (origen)", value=-100.3138,
                                step=0.0005, format="%.4f")
    with cr2:
        st.markdown("**Destino**")
        lat_d = st.number_input("Latitud (destino)", value=25.7298,
                                step=0.0005, format="%.4f")
        lon_d = st.number_input("Longitud (destino)", value=-100.3045,
                                step=0.0005, format="%.4f")

    peso_contam = st.slider(
        "Peso de la limpieza (0 = más corta, 1 = más limpia)",
        0.0, 1.0, 0.7, 0.05,
    )

    if st.button("Calcular ruta"):
        idx_o = coords_to_index(lat_o, lon_o, grid)
        idx_d = coords_to_index(lat_d, lon_d, grid)

        if not mask[idx_o]:
            st.error("El punto de partida está fuera del polígono de CU.")
        elif not mask[idx_d]:
            st.error("El destino está fuera del polígono de CU.")
        else:
            with st.spinner("Buscando rutas óptimas…"):
                ruta_corta = find_clean_route(A, mask, idx_o, idx_d,
                                              pollution_weight=0.0)
                ruta_limpia = find_clean_route(A, mask, idx_o, idx_d,
                                               pollution_weight=peso_contam)

            if ruta_corta and ruta_limpia:
                stats_corta = route_stats(ruta_corta, A,
                                          cell_size_m=grid["cell_size_m"])
                stats_limpia = route_stats(ruta_limpia, A,
                                           cell_size_m=grid["cell_size_m"])

                ccol1, ccol2 = st.columns(2)
                with ccol1:
                    st.markdown("##### Ruta más corta")
                    st.metric("Distancia", f"{stats_corta['longitud_m']:.0f} m")
                    st.metric("ICA medio en ruta",
                              f"{stats_corta['ica_medio']:.0f}")
                    st.metric("ICA máximo expuesto",
                              f"{stats_corta['ica_max']:.0f}")
                    st.markdown(mask_recommendation(stats_corta["ica_max"]), unsafe_allow_html=True)
                with ccol2:
                    st.markdown("##### Ruta de menor exposición")
                    st.metric("Distancia", f"{stats_limpia['longitud_m']:.0f} m",
                              delta=f"{stats_limpia['longitud_m'] - stats_corta['longitud_m']:+.0f} m")
                    st.metric("ICA medio en ruta",
                              f"{stats_limpia['ica_medio']:.0f}",
                              delta=f"{stats_limpia['ica_medio'] - stats_corta['ica_medio']:+.0f}",
                              delta_color="inverse")
                    st.metric("ICA máximo expuesto",
                              f"{stats_limpia['ica_max']:.0f}",
                              delta=f"{stats_limpia['ica_max'] - stats_corta['ica_max']:+.0f}",
                              delta_color="inverse")
                    st.markdown(mask_recommendation(stats_limpia["ica_max"]), unsafe_allow_html=True)

                reduccion = 100 * (1 - stats_limpia["ica_medio"] /
                                   max(stats_corta["ica_medio"], 0.01))
                if reduccion > 5:
                    st.success(
                        f"La ruta alternativa reduce tu exposición media en "
                        f"**{reduccion:.0f}%** a costa de **"
                        f"{stats_limpia['longitud_m'] - stats_corta['longitud_m']:+.0f} m** "
                        f"adicionales."
                    )

                mapa_rutas = construir_mapa(grid, mask, inf, A,
                                            roads=roads,
                                            ruta_puntos=ruta_corta,
                                            ruta_limpia=ruta_limpia,
                                            contaminante=contaminante,
                                            usar_osm=usar_osm,
                                            alto_contraste=alto_contraste)
                st_folium(mapa_rutas, width=None, height=540,
                          returned_objects=[])
            else:
                st.error("No se pudo encontrar una ruta entre esos puntos. "
                         "Verifica que ambos estén dentro del polígono.")


# ----- TAB VALIDACION SIMA -----
with tab_validacion:
    st.markdown("#### Validación contra el SIMA Nuevo León")
    st.caption(
        "Compara las predicciones del simulador con observaciones de "
        "las estaciones del Sistema Integral de Monitoreo Ambiental. "
        "Sube un CSV exportado del portal SIMA, o usa el conjunto de "
        "OpenMeteo para ver el flujo de validación."
    )

    fuente_sima = st.radio(
        "Fuente de datos de observación",
        ["Datos de OpenMeteo", "Subir CSV del SIMA (Anaisis Previo o Actual)"],
        horizontal=True,
    )

    df_sima = None
    if fuente_sima == "Subir CSV del SIMA":
        archivo = st.file_uploader(
            "CSV del SIMA",
            type=["csv"],
            help="Formato esperado: columnas fecha, hora, estacion y una "
                 "columna por contaminante (PM2.5, PM10, NO2, SO2).",
        )
        if archivo is not None:
            try:
                df_sima = cargar_datos_sima(archivo, contaminante)
                st.success(f"{len(df_sima)} observaciones cargadas.")
            except Exception as e:
                st.error(f"No se pudo leer el CSV: {e}")
    else:
        estacion_demo = st.selectbox(
            "Estación de Datos", list(ESTACIONES_SIMA.keys())
        )
        df_raw = generar_sima_ejemplo(contaminante, estacion=estacion_demo,
                                      dias=3)
        import io as _io
        _buf = _io.StringIO()
        df_raw.to_csv(_buf, index=False)
        _buf.seek(0)
        df_sima = cargar_datos_sima(_buf, contaminante)
        st.info(f"Usando {len(df_sima)} observaciones de "
                f"3 días para la estación **{estacion_demo}**.")

    if df_sima is not None and not df_sima.empty:
        estaciones_disp = sorted(df_sima["estacion"].unique())
        estacion_val = st.selectbox("Estación a validar", estaciones_disp)

        # Generar las predicciones del simulador para las 24 h del día,
        # evaluadas en la celda más cercana a la estación.
        with st.spinner("Simulando 24 h para comparar con la estación…"):
            frames_val = simular_24h(
                contaminante=contaminante,
                viento_ms=viento_ms, viento_dir=viento_dir,
                temperatura=temperatura, presion=presion,
                factor_trafico=factor_trafico_efectivo,
                factor_industrial=factor_industrial,
                es_dia_laboral=es_dia_laboral,
                inversion_horas_tuple=(5, 6, 7, 8, 9),
                usar_osm=usar_osm,
            )

        # La estación puede estar fuera del polígono: usamos el valor del
        # borde más cercano como proxy de la contribución local modelada.
        est_coords = ESTACIONES_SIMA.get(estacion_val)
        if est_coords:
            i_est, j_est = coords_to_index(est_coords[0], est_coords[1], grid)
        else:
            i_est, j_est = grid["filas"] // 2, grid["columnas"] // 2

        pred_por_hora = {}
        for f in frames_val:
            # Promedio de la concentración en una vecindad 5×5 de la estación
            i0, i1 = max(0, i_est - 2), min(grid["filas"], i_est + 3)
            j0, j1 = max(0, j_est - 2), min(grid["columnas"], j_est + 3)
            sub = f["C"][i0:i1, j0:j1]
            pred_por_hora[f["hora"]] = float(sub.mean())

        resultado = validar_serie(df_sima, estacion_val, pred_por_hora)
        m = resultado["metricas"]

        if "error" in m:
            st.warning(m["error"])
        else:
            # Métricas
            v1, v2, v3, v4 = st.columns(4)
            v1.metric("RMSE", f"{m['rmse']:.1f} μg/m³")
            v2.metric("Sesgo", f"{m['sesgo']:+.1f} μg/m³")
            v3.metric("Correlación", f"{m['correlacion']:.3f}")
            v4.metric("Índice de concordancia", f"{m['ioa']:.3f}")

            st.info(f"**Interpretación**: {resultado['interpretacion']}")

            # Gráfico observado vs simulado
            fig_val = go.Figure()
            fig_val.add_trace(go.Scatter(
                x=resultado["horas"], y=resultado["observado"],
                name="Observado (SIMA)", mode="lines+markers",
                line=dict(color="#1F77B4", width=3),
            ))
            fig_val.add_trace(go.Scatter(
                x=resultado["horas"], y=resultado["simulado"],
                name="Simulado", mode="lines+markers",
                line=dict(color="#D62728", width=3, dash="dash"),
            ))
            fig_val.update_layout(
                title=f"Observado vs. Simulado — {estacion_val} ({contaminante})",
                xaxis_title="Hora del día",
                yaxis_title=f"Concentración {contaminante} (μg/m³)",
                height=400, hovermode="x unified",
            )
            st.plotly_chart(fig_val, width="stretch")

            st.caption(
                "Nota metodológica: las estaciones del SIMA están fuera "
                "del polígono de CU, por lo que esta comparación valida la "
                "**forma del ciclo diario** y el orden de magnitud, no el "
                "valor absoluto puntual. Para validación rigurosa se "
                "requeriría una estación dentro del dominio o anidar el "
                "modelo en un dominio regional más amplio."
            )

            # Tabla de scatter obs-sim
            with st.expander("Ver tabla de datos comparados"):
                df_comp = pd.DataFrame({
                    "Hora": resultado["horas"],
                    "Observado (μg/m³)": [round(x, 1) for x in resultado["observado"]],
                    "Simulado (μg/m³)":  [round(x, 1) for x in resultado["simulado"]],
                    "Error":             [round(s - o, 1) for o, s in
                                          zip(resultado["observado"],
                                              resultado["simulado"])],
                })
                st.dataframe(df_comp, hide_index=True, width="stretch")

# ----- TAB 4: INFO -----
with tab_info:
    st.markdown(
        r"""
        ### Acerca del modelo

        Esta simulación implementa la **propuesta del Equipo 11 — Brigada 003**
        para modelar la dispersión de contaminantes atmosféricos en Ciudad
        Universitaria UANL.

        #### 🔬 Modelo físico
        Se resuelve la **ecuación de advección–difusión 2D** sobre una rejilla
        regular de **15 m × 15 m**:

        $$\frac{\partial C}{\partial t} = -u\frac{\partial C}{\partial x}
        - v\frac{\partial C}{\partial y} + D\,\nabla^2 C + S(x,y) - k\,C$$

        donde:
        - $C$ = concentración del contaminante (μg/m³)
        - $(u, v)$ = componentes del viento (m/s)
        - $D$ = coeficiente de difusión turbulenta (m²/s)
        - $S$ = emisiones (fuentes móviles + fuentes fijas)
        - $k$ = tasa de pérdida vertical / deposición / decaimiento

        El esquema temporal es **explícito**: advección upwind de primer orden
        y difusión por diferencias centradas, con dt ajustado dinámicamente
        para cumplir las condiciones CFL.

        #### 📡 Fuentes de datos
        - **Meteorología**: API pública [Open-Meteo](https://open-meteo.com/)
          (temperatura, viento, presión).
        - **Red vial**: [OpenStreetMap](https://openstreetmap.org/) vía Overpass API
          con clasificación por jerarquía (`trunk`, `primary`, `secondary`…).
          Cada vía aporta una capacidad vehicular según el Highway Capacity
          Manual.
        - **Flujo vehicular**: capacidad por categoría × perfil horario
          calibrado para el Área Metropolitana de Monterrey × factor laboral.
        - **Factores de emisión vehicular** (g/km/veh): estilo EMFAC/COPERT
          para una flota mexicana típica (75% gasolina ligero, 20% diésel
          pesado, 5% otros).
        - **Industria**: planta Ternium como fuente fija de área con tasa
          de emisión 24/7.
        - **Estándares ICA**: NOM-172-SEMARNAT-2019.

        #### 🚗 Modelo de flujo vehicular
        Para cada celda de 15 × 15 m por la que pasa una vialidad:

        $$E_{\text{celda}} \;\left[\frac{\mu g}{m^3 \cdot s}\right] =
        \frac{Q\;\text{[veh/h]} \cdot FE\;[g/km] \cdot \Delta x\;[km]
        \cdot 10^6}{3600 \cdot \Delta x^2 \cdot H_{mix}}$$

        donde $Q$ varía hora a hora con el perfil:
        - 3 AM:   2 % de la capacidad
        - 8 AM:  100 % (pico matutino)
        - 14 PM:  65 %
        - 18 PM: 100 % (pico vespertino)
        - 22 PM:  25 %

        El módulo de **evolución 24h** simula la jornada completa con
        **arrastre temporal**: la concentración a la hora $t$ es el resultado
        de la atmósfera de la hora $t-1$ más las nuevas emisiones, transportada
        por el viento y disipada por difusión y deposición.

        El módulo de **animación en tiempo casi real** parte de aire limpio y
        captura decenas de instantáneas mientras la pluma se forma y se mueve,
        permitiendo observar el transporte advectivo cuadro a cuadro.

        #### 🚦 Tráfico real (TomTom, opcional)
        Si se proporciona una clave gratuita de TomTom, el simulador consulta
        la **congestión real** en varios puntos de las avenidas. La razón
        entre velocidad actual y velocidad en flujo libre se traduce a un
        **multiplicador de emisión**: el tráfico detenido (stop-and-go) emite
        hasta ~2× más PM y NOx por kilómetro que el flujo libre.

        #### 📊 Validación contra SIMA
        El módulo de validación compara el ciclo diario simulado con
        observaciones del **Sistema Integral de Monitoreo Ambiental** de
        Nuevo León, calculando RMSE, sesgo, correlación de Pearson e índice
        de concordancia de Willmott. Acepta un CSV exportado del portal SIMA
        o un conjunto de datos de OpenMeteo.

        #### ⚠️ Limitaciones reconocidas
        - Modelo 2D: no resuelve gradiente vertical (capa de mezcla simulada
          por parámetro `inversión térmica`).
        - Factores de emisión calibrados de forma aproximada — para uso
          operativo requieren validación contra datos del SIMA.
        - Las coordenadas exactas de las avenidas son aproximadas; en versión
          de producción se geocodificarán desde OpenStreetMap.
        - El esquema upwind de primer orden introduce **difusión numérica**:
          aceptable para análisis de impacto, no para gradientes finos.

        #### 🛠️ Roadmap de mejoras
        1. Validación contra datos históricos del **SIMA Nuevo León**
           (estaciones Obispado, San Nicolás).
        2. Geocodificación de avenidas vía Overpass API / OSM.
        3. Integración con datos de tráfico (Google Maps Distance Matrix API).
        4. Modelo 3D con capa de mezcla resuelta.
        5. Notificaciones push a usuarios suscritos a una zona.
        """
    )


# =====================================================================
# REMOVER PANTALLA DE CARGA INICIAL
# =====================================================================
remove_loading_screen(loading_placeholder)

