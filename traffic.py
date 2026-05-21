"""
Modelo de flujo vehicular y emisiones vehiculares.

Proporciona:
  - fetch_osm_roads     : red vial real desde OpenStreetMap (Overpass API)
  - DEFAULT_ROADS       : red vial de respaldo (más rica que las 3 avenidas iniciales)
  - build_traffic_map   : matriz de flujo vehicular (veh/h) por celda
  - emissions_from_traffic : emisión (μg/m³/s) a partir del flujo vehicular
  - HOURLY_PROFILE      : perfil horario de demanda vehicular

Factores de emisión basados en literatura de EMFAC / COPERT para una flota
urbana mexicana típica (mezcla de gasolina/diesel, distintas edades).
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import requests
import streamlit as st

# ---------------------------------------------------------------------------
# 1. CAPACIDAD VIAL POR CATEGORIA OSM (vehículos/hora · sentido · carril)
# ---------------------------------------------------------------------------
# Valores de referencia del Highway Capacity Manual y revisiones IMT-México.
ROAD_CAPACITY = {
    "motorway":       4000,
    "motorway_link":  1500,
    "trunk":          3000,
    "trunk_link":     1500,
    "primary":        2000,
    "primary_link":    900,
    "secondary":      1200,
    "secondary_link":  700,
    "tertiary":        800,
    "tertiary_link":   500,
    "unclassified":    300,
    "residential":     250,
    "service":         100,
    "living_street":   100,
}

# Perfil horario de demanda (fracción del flujo máximo).
# Calibrado a observaciones del Área Metropolitana de Monterrey.
HOURLY_PROFILE = {
    0: 0.05, 1: 0.03, 2: 0.02, 3: 0.02, 4: 0.04, 5: 0.12,
    6: 0.35, 7: 0.80, 8: 1.00, 9: 0.80, 10: 0.55, 11: 0.55,
    12: 0.65, 13: 0.75, 14: 0.65, 15: 0.60, 16: 0.70, 17: 0.95,
    18: 1.00, 19: 0.85, 20: 0.55, 21: 0.40, 22: 0.25, 23: 0.15,
}


# ---------------------------------------------------------------------------
# 2. FACTORES DE EMISION (g por km por vehículo, flota promedio MTY)
# ---------------------------------------------------------------------------
# Mezcla típica: 75% gasolina ligero, 20% diesel pesado, 5% otros.
# Calibrados para que el flujo máximo de un trunk (3000 veh/h) produzca
# concentraciones cercanas a 50-100 μg/m³ de PM2.5 a nivel de suelo,
# consistente con observaciones del SIMA en sitios cercanos a vialidades.
EMISSION_FACTORS_G_PER_VEH_KM = {
    "PM2.5": 0.040,
    "PM10":  0.078,
    "NOx":   0.580,
    "SO2":   0.045,
}

# Altura de mezcla efectiva (m). En la práctica varía con la hora y la
# estabilidad atmosférica; aquí se usa un valor representativo de zona
# urbana diurna sin inversión.
DEFAULT_MIXING_HEIGHT_M = 50.0


# ---------------------------------------------------------------------------
# 3. RED VIAL DE RESPALDO (si no hay conexión a Overpass)
# ---------------------------------------------------------------------------
# Trazos aproximados de las vialidades principales y secundarias dentro y
# alrededor del polígono de CU. Cada vía es una secuencia de (lat, lon).
DEFAULT_ROADS = [
    # --- Vialidades primarias ---
    {"name": "Av. Universidad", "type": "trunk", "lanes": 4,
     "coords": [(25.73503, -100.31487), (25.73000, -100.31370),
                (25.72500, -100.31210), (25.72000, -100.31130),
                (25.71861, -100.31091)]},
    {"name": "Av. Manuel L. Barragán / Nogalar", "type": "primary", "lanes": 3,
     "coords": [(25.73230, -100.31750), (25.72850, -100.30800),
                (25.72500, -100.30200), (25.72350, -100.29950)]},
    {"name": "Av. Fidel Velázquez", "type": "primary", "lanes": 4,
     "coords": [(25.73450, -100.30420), (25.73000, -100.30200),
                (25.72500, -100.30050), (25.71900, -100.29950)]},
    {"name": "Av. Alfonso Reyes (sur de CU)", "type": "primary", "lanes": 3,
     "coords": [(25.72100, -100.31900), (25.72050, -100.31200),
                (25.72000, -100.30500), (25.71950, -100.29850)]},

    # --- Vialidades secundarias ---
    {"name": "Pedro de Alba (interna)", "type": "secondary", "lanes": 2,
     "coords": [(25.72950, -100.31800), (25.72950, -100.30500)]},
    {"name": "Praga (interna)", "type": "secondary", "lanes": 2,
     "coords": [(25.72400, -100.31700), (25.72400, -100.30000)]},
    {"name": "Conexión norte-sur 1", "type": "secondary", "lanes": 2,
     "coords": [(25.73400, -100.31000), (25.72100, -100.30900)]},
    {"name": "Conexión norte-sur 2", "type": "secondary", "lanes": 2,
     "coords": [(25.73300, -100.30500), (25.72050, -100.30400)]},

    # --- Calles residenciales (carga baja) ---
    {"name": "Residencial 1", "type": "residential", "lanes": 1,
     "coords": [(25.73100, -100.31500), (25.73100, -100.30700)]},
    {"name": "Residencial 2", "type": "residential", "lanes": 1,
     "coords": [(25.72700, -100.31600), (25.72700, -100.30200)]},
    {"name": "Residencial 3", "type": "residential", "lanes": 1,
     "coords": [(25.72300, -100.31500), (25.72300, -100.30200)]},
]


# ---------------------------------------------------------------------------
# 4. INTEGRACION CON OVERPASS / OSM
# ---------------------------------------------------------------------------

# Varios mirrors públicos de Overpass. Se prueban en orden hasta que uno
# responda; así el sistema es robusto ante saturación o caídas de un mirror.
OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    "https://overpass.openstreetmap.ru/api/interpreter",
]

# Algunos mirrors rechazan peticiones sin User-Agent identificable.
_OVERPASS_HEADERS = {
    "User-Agent": "SimuladorCalidadAire-UANL/1.0 (proyecto academico)"
}


@st.cache_data(ttl=86400, show_spinner="Consultando OpenStreetMap…")
def fetch_osm_roads(bounds: dict, buffer_deg: float = 0.003) -> list[dict]:
    """
    Obtiene la red vial dentro del bounding box (con buffer) desde Overpass.

    Prueba varios mirrors de Overpass en orden; si todos fallan, devuelve
    DEFAULT_ROADS como respaldo para que la app siga funcionando.

    Args:
        bounds: dict con keys min_lat, max_lat, min_lon, max_lon
        buffer_deg: ampliación del bbox en grados (≈ 330 m por 0.003°)

    Returns:
        Lista de dicts {name, type, lanes, coords[]}.
    """
    south = bounds["min_lat"] - buffer_deg
    west  = bounds["min_lon"] - buffer_deg
    north = bounds["max_lat"] + buffer_deg
    east  = bounds["max_lon"] + buffer_deg

    types = "|".join([
        "motorway", "motorway_link", "trunk", "trunk_link",
        "primary", "primary_link", "secondary", "secondary_link",
        "tertiary", "tertiary_link", "residential", "unclassified",
    ])
    # Nota: la salida correcta es "out geom;" (incluye geometría + tags).
    # "out geom tags;" es sintaxis inválida y provoca HTTP 400.
    query = (
        "[out:json][timeout:25];"
        f'(way["highway"~"^({types})$"]({south},{west},{north},{east}););'
        "out geom;"
    )

    ultimo_error = None
    for endpoint in OVERPASS_ENDPOINTS:
        try:
            r = requests.post(
                endpoint,
                data={"data": query},
                headers=_OVERPASS_HEADERS,
                timeout=30,
            )
            r.raise_for_status()
            data = r.json()
            roads = _parse_overpass(data)
            if roads:
                return roads
            # Respuesta válida pero vacía: probar siguiente mirror
            ultimo_error = "respuesta vacía"
        except Exception as e:  # noqa: BLE001
            ultimo_error = f"{type(e).__name__}"
            continue

    st.warning(
        f"⚠️ No se pudo consultar Overpass ({ultimo_error}). "
        f"Usando red vial de respaldo (11 vialidades principales). "
        f"La simulación funciona igual; solo cambia el detalle de la red."
    )
    return DEFAULT_ROADS


def _parse_overpass(data: dict) -> list[dict]:
    """Convierte la respuesta JSON de Overpass a la lista de vialidades."""
    roads = []
    for el in data.get("elements", []):
        if el.get("type") != "way" or "geometry" not in el:
            continue
        tags = el.get("tags", {})
        coords = [(g["lat"], g["lon"]) for g in el["geometry"]]
        if len(coords) < 2:
            continue
        # Carriles: tomar el dato si existe y es numérico, si no usar default
        lanes_raw = tags.get("lanes", "")
        try:
            lanes = max(1, int(str(lanes_raw).split(";")[0]))
        except (ValueError, AttributeError):
            lanes = 2 if tags.get("highway", "").startswith(("primary", "trunk")) else 1
        roads.append({
            "name":   tags.get("name", f"way_{el['id']}"),
            "type":   tags.get("highway", "unclassified"),
            "lanes":  lanes,
            "coords": coords,
            "oneway": tags.get("oneway") == "yes",
        })
    return roads


# ---------------------------------------------------------------------------
# 5. MAPA DE FLUJO VEHICULAR EN LA REJILLA
# ---------------------------------------------------------------------------

def _bresenham(i0, j0, i1, j1):
    """Línea discreta entre dos celdas (8-conectada)."""
    points = []
    di, dj = abs(i1 - i0), abs(j1 - j0)
    si = 1 if i0 < i1 else -1
    sj = 1 if j0 < j1 else -1
    err = di - dj
    while True:
        points.append((i0, j0))
        if i0 == i1 and j0 == j1:
            break
        e2 = 2 * err
        if e2 > -dj:
            err -= dj
            i0 += si
        if e2 < di:
            err += di
            j0 += sj
    return points


def _coords_to_index(lat, lon, grid):
    i = int((lat - grid["bounds"]["min_lat"]) /
            (grid["bounds"]["max_lat"] - grid["bounds"]["min_lat"]) * (grid["filas"] - 1))
    j = int((lon - grid["bounds"]["min_lon"]) /
            (grid["bounds"]["max_lon"] - grid["bounds"]["min_lon"]) * (grid["columnas"] - 1))
    return int(np.clip(i, 0, grid["filas"] - 1)), int(np.clip(j, 0, grid["columnas"] - 1))


def build_traffic_map(grid, roads: list[dict], hora: int,
                      es_dia_laboral: bool = True,
                      factor_global: float = 1.0,
                      ancho_celdas: int = 1) -> np.ndarray:
    """
    Construye la matriz de flujo vehicular en veh/h por celda.

    Args:
        grid: rejilla de simulator.build_grid
        roads: lista de vialidades (de fetch_osm_roads o DEFAULT_ROADS)
        hora: 0-23
        es_dia_laboral: fines de semana reducen flujo ~40%
        factor_global: multiplicador (escenarios de reducción/aumento)
        ancho_celdas: ensancha cada vía (1 = una sola celda, 2 = ±2 celdas)

    Returns:
        Matriz (filas, columnas) con flujo vehicular en veh/h.
    """
    traffic = np.zeros((grid["filas"], grid["columnas"]), dtype=np.float32)
    perfil = HOURLY_PROFILE.get(hora, 0.5)
    if not es_dia_laboral:
        perfil *= 0.60

    for road in roads:
        cap_por_carril = ROAD_CAPACITY.get(road["type"], 200)
        lanes = max(1, int(road.get("lanes", 1)))
        # Si la vía es de doble sentido y "lanes" cuenta el total, dividimos.
        # Convención: capacidad efectiva = cap_por_carril × num_carriles.
        flujo_pico = cap_por_carril * lanes
        flujo = flujo_pico * perfil * factor_global

        coords = road["coords"]
        for k in range(len(coords) - 1):
            la0, lo0 = coords[k]
            la1, lo1 = coords[k + 1]
            i0, j0 = _coords_to_index(la0, lo0, grid)
            i1, j1 = _coords_to_index(la1, lo1, grid)
            for (i, j) in _bresenham(i0, j0, i1, j1):
                for di in range(-ancho_celdas, ancho_celdas + 1):
                    for dj in range(-ancho_celdas, ancho_celdas + 1):
                        ni, nj = i + di, j + dj
                        if 0 <= ni < grid["filas"] and 0 <= nj < grid["columnas"]:
                            # Si varias vías cruzan la misma celda,
                            # tomamos el máximo (no se suman para evitar
                            # doble conteo en cruces).
                            if flujo > traffic[ni, nj]:
                                traffic[ni, nj] = flujo
    return traffic


# ---------------------------------------------------------------------------
# 6. CONVERSION FLUJO -> EMISIONES
# ---------------------------------------------------------------------------

def emissions_from_traffic(
    traffic_veh_h: np.ndarray,
    contaminante: str = "PM2.5",
    cell_size_m: float = 15.0,
    mixing_height_m: float = DEFAULT_MIXING_HEIGHT_M,
) -> np.ndarray:
    """
    Convierte flujo vehicular en emisiones por celda (μg/m³/s).

    Derivación:
        E[g/s/celda] = veh_h × FE[g/km] × (Δx[km]) / 3600
        V[m³/celda]  = Δx² × H_mix
        E[μg/m³/s]   = E[g/s] × 10⁶ / V

    Args:
        traffic_veh_h: matriz de veh/h por celda
        contaminante:  'PM2.5'|'PM10'|'NOx'|'SO2'
        cell_size_m:   tamaño de celda
        mixing_height_m: altura efectiva de mezcla

    Returns:
        Matriz de emisiones (μg/m³/s) lista para pasarse al motor.
    """
    if contaminante not in EMISSION_FACTORS_G_PER_VEH_KM:
        raise ValueError(f"Contaminante desconocido: {contaminante}")

    FE = EMISSION_FACTORS_G_PER_VEH_KM[contaminante]      # g/km/veh
    cell_volume_m3 = (cell_size_m ** 2) * mixing_height_m  # m³

    # g/s por celda  (Δx en km = cell_size_m / 1000)
    g_per_s = traffic_veh_h * FE * (cell_size_m / 1000.0) / 3600.0
    # μg/m³/s
    return (g_per_s * 1e6 / cell_volume_m3).astype(np.float32)


# ---------------------------------------------------------------------------
# 7. FUENTE FIJA INDUSTRIAL (Ternium)
# ---------------------------------------------------------------------------

INDUSTRIAL_EMISSION_FACTORS = {  # μg/m³/s por celda industrial
    "PM2.5": 0.180,
    "PM10":  0.320,
    "NOx":   0.095,
    "SO2":   0.260,
}


def industrial_emissions(grid, lat_range, lon_range, contaminante,
                         factor: float = 1.0) -> np.ndarray:
    """Matriz de emisiones de una fuente fija de área (planta industrial)."""
    E = np.zeros((grid["filas"], grid["columnas"]), dtype=np.float32)
    base = INDUSTRIAL_EMISSION_FACTORS.get(contaminante, 0.1) * factor
    for i in range(grid["filas"]):
        lat = grid["lat_grid"][i]
        if not (lat_range[0] <= lat <= lat_range[1]):
            continue
        for j in range(grid["columnas"]):
            lon = grid["lon_grid"][j]
            if lon_range[0] <= lon <= lon_range[1]:
                E[i, j] = base
    return E


# ---------------------------------------------------------------------------
# 8. UTILIDADES DE INSPECCION
# ---------------------------------------------------------------------------

def resumen_red(roads: list[dict]) -> dict:
    """Resumen de la red vial cargada (para mostrar en la UI)."""
    por_tipo = {}
    total_km = 0.0
    for r in roads:
        por_tipo[r["type"]] = por_tipo.get(r["type"], 0) + 1
        coords = r["coords"]
        for k in range(len(coords) - 1):
            la0, lo0 = coords[k]
            la1, lo1 = coords[k + 1]
            dla = (la1 - la0) * 111139
            dlo = (lo1 - lo0) * 100300
            total_km += np.hypot(dla, dlo) / 1000.0
    return {"total_segmentos": len(roads),
            "km_totales": round(total_km, 2),
            "por_tipo": por_tipo}
