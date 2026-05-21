"""
Motor de simulación de dispersión atmosférica para Ciudad Universitaria UANL.

Implementa el modelo descrito en la propuesta:
- Rejilla espacial de 15 m sobre el polígono de CU
- Fuentes móviles (avenidas: Universidad, Nogalar, Fidel Velázquez)
- Fuente fija (planta Ternium)
- Modelo de advección-difusión 2D con esquema upwind explícito
- Modulación por velocidad/dirección del viento, temperatura, presión
- Fricción diferencial por uso de suelo (landuse.py): zonas arboladas,
  edificios, zonas abiertas, agua e industrial modifican D y velocidad
  efectiva celda a celda, basándose en datos OSM con respaldo hardcodeado.
- Cálculo de ICA según NOM-172-SEMARNAT-2019 (puntos de corte simplificados)

Equipo 11 - Brigada 003 - Modelado y Simulación de Sistemas Dinámicos
"""
from __future__ import annotations
import numpy as np
from matplotlib.path import Path

# ---------------------------------------------------------------------------
# 1. POLIGONO DE ESTUDIO Y CONFIGURACION DE LA REJILLA
# ---------------------------------------------------------------------------

POLYGON_LIMITS = [
    [25.730516, -100.319192],  # NW
    [25.735033, -100.302866],  # N
    [25.720806, -100.298841],  # E
    [25.718611, -100.310050],  # S
    [25.722354, -100.318129],  # SW
    [25.730516, -100.319192],  # cierre
]

CELL_SIZE_M = 15  # metros por celda (resolución espacial)
CENTER_LAT = 25.7255
CENTER_LON = -100.3118


def build_grid(polygon=POLYGON_LIMITS, cell_size_m=CELL_SIZE_M):
    """Construye la rejilla regular sobre el bounding box del polígono."""
    lats = [c[0] for c in polygon]
    lons = [c[1] for c in polygon]
    bounds = {
        "min_lat": min(lats), "max_lat": max(lats),
        "min_lon": min(lons), "max_lon": max(lons),
    }
    # Conversión grado -> metros (aproximada a esta latitud)
    alto_m = (bounds["max_lat"] - bounds["min_lat"]) * 111139
    ancho_m = (bounds["max_lon"] - bounds["min_lon"]) * 100300
    filas = int(alto_m / cell_size_m)
    columnas = int(ancho_m / cell_size_m)
    lat_grid = np.linspace(bounds["min_lat"], bounds["max_lat"], filas)
    lon_grid = np.linspace(bounds["min_lon"], bounds["max_lon"], columnas)
    return {
        "bounds": bounds,
        "filas": filas,
        "columnas": columnas,
        "lat_grid": lat_grid,
        "lon_grid": lon_grid,
        "cell_size_m": cell_size_m,
    }


def build_mask(grid, polygon=POLYGON_LIMITS):
    """Máscara booleana de celdas dentro del polígono de CU."""
    path = Path(polygon)
    lon_mesh, lat_mesh = np.meshgrid(grid["lon_grid"], grid["lat_grid"])
    pts = np.vstack((lat_mesh.flatten(), lon_mesh.flatten())).T
    return path.contains_points(pts).reshape(grid["filas"], grid["columnas"])


def coords_to_index(lat, lon, grid):
    """Convierte (lat, lon) -> (i, j) en la matriz."""
    i = int((lat - grid["bounds"]["min_lat"]) /
            (grid["bounds"]["max_lat"] - grid["bounds"]["min_lat"]) * (grid["filas"] - 1))
    j = int((lon - grid["bounds"]["min_lon"]) /
            (grid["bounds"]["max_lon"] - grid["bounds"]["min_lon"]) * (grid["columnas"] - 1))
    i = int(np.clip(i, 0, grid["filas"] - 1))
    j = int(np.clip(j, 0, grid["columnas"] - 1))
    return i, j


def index_to_coords(i, j, grid):
    """Convierte (i, j) -> (lat, lon)."""
    return float(grid["lat_grid"][i]), float(grid["lon_grid"][j])


def bresenham_line(i0, j0, i1, j1):
    """Algoritmo de Bresenham para discretizar una línea sobre la matriz."""
    points = []
    di = abs(i1 - i0)
    dj = abs(j1 - j0)
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


# ---------------------------------------------------------------------------
# 2. INFRAESTRUCTURA: AVENIDAS (líneas) Y FABRICA (área)
# ---------------------------------------------------------------------------

INF_VACIO = 0
INF_VIAL_PRINCIPAL = 1
INF_FABRICA = 2
INF_VIAL_SECUNDARIA = 3

# Coordenadas aproximadas de las vialidades dentro/cerca del polígono.
# (En producción se geocodificarían con OSM; aquí se mantienen como aproximación
# para que el modelo opere sobre las tres avenidas mencionadas en la propuesta.)
AVENIDAS = {
    "Av. Universidad": {
        "coords": [(25.73503, -100.31487), (25.71861, -100.31091)],
        "tipo": INF_VIAL_PRINCIPAL,
        "flujo_base": 1.00,    # peso relativo de tráfico
        "ancho_celdas": 2,
    },
    "Av. Nogalar": {
        "coords": [(25.73230, -100.31750), (25.72350, -100.29950)],
        "tipo": INF_VIAL_PRINCIPAL,
        "flujo_base": 0.85,
        "ancho_celdas": 2,
    },
    "Av. Fidel Velázquez": {
        "coords": [(25.73450, -100.30420), (25.71900, -100.29950)],
        "tipo": INF_VIAL_PRINCIPAL,
        "flujo_base": 0.95,
        "ancho_celdas": 2,
    },
}

# Planta Ternium - ubicación real al este/sureste de Ciudad Universitaria.
TERNIUM_AREA = {
    "lat_range": (25.7180, 25.7240),
    "lon_range": (-100.3060, -100.2980),
    "tipo": INF_FABRICA,
    "tasa_emision": 1.0,
}


def build_infrastructure(grid):
    """Construye la matriz de infraestructura con avenidas y fábrica."""
    inf = np.zeros((grid["filas"], grid["columnas"]), dtype=np.int8)

    # Avenidas (líneas con grosor)
    for nombre, info in AVENIDAS.items():
        (la0, lo0), (la1, lo1) = info["coords"]
        i0, j0 = coords_to_index(la0, lo0, grid)
        i1, j1 = coords_to_index(la1, lo1, grid)
        line = bresenham_line(i0, j0, i1, j1)
        w = info["ancho_celdas"]
        for (i, j) in line:
            for di in range(-w, w + 1):
                for dj in range(-w, w + 1):
                    ni, nj = i + di, j + dj
                    if 0 <= ni < grid["filas"] and 0 <= nj < grid["columnas"]:
                        # No sobrescribir fábrica
                        if inf[ni, nj] != INF_FABRICA:
                            inf[ni, nj] = info["tipo"]

    # Ternium (rectángulo)
    for i in range(grid["filas"]):
        lat = grid["lat_grid"][i]
        if not (TERNIUM_AREA["lat_range"][0] <= lat <= TERNIUM_AREA["lat_range"][1]):
            continue
        for j in range(grid["columnas"]):
            lon = grid["lon_grid"][j]
            if TERNIUM_AREA["lon_range"][0] <= lon <= TERNIUM_AREA["lon_range"][1]:
                inf[i, j] = INF_FABRICA

    return inf


# ---------------------------------------------------------------------------
# 3. EMISIONES (μg/m³/s aproximado, escala relativa)
# ---------------------------------------------------------------------------

# Factores por contaminante (μg/m³/s por celda emisora).
# Calibrados para producir ICA realista en Monterrey:
#   - tráfico normal:   ICA 30–80
#   - hora pico:        ICA 60–130
#   - inversión térmica: ICA 150–300
FACTORES_EMISION = {
    "PM2.5": {"trafico": 0.050, "industrial": 0.140},
    "PM10":  {"trafico": 0.095, "industrial": 0.250},
    "NOx":   {"trafico": 0.190, "industrial": 0.075},
    "SO2":   {"trafico": 0.030, "industrial": 0.200},
}


def perfil_horario_trafico(hora):
    """Intensidad relativa de tráfico según hora (0-23)."""
    horas_pico_am = {7, 8, 9}
    horas_pico_pm = {17, 18, 19}
    horas_media  = {10, 11, 12, 13, 14, 15, 16, 20, 21}
    if hora in horas_pico_am or hora in horas_pico_pm:
        return 1.0
    if hora in horas_media:
        return 0.55
    return 0.20  # madrugada


def calculate_emissions(mapa_inf, hora, contaminante="PM2.5",
                        factor_trafico=1.0, factor_industrial=1.0):
    """
    Matriz de tasa de emisión por celda.

    Args:
        mapa_inf: matriz de infraestructura (output de build_infrastructure)
        hora: 0-23
        contaminante: 'PM2.5'|'PM10'|'NOx'|'SO2'
        factor_trafico: multiplicador (1.0 = base; 0.7 = -30%)
        factor_industrial: multiplicador para Ternium

    Returns:
        E: matriz de emisiones (mismas dimensiones que mapa_inf)
    """
    f = FACTORES_EMISION[contaminante]
    intensidad = perfil_horario_trafico(hora)

    E = np.zeros_like(mapa_inf, dtype=np.float32)
    mask_via = (mapa_inf == INF_VIAL_PRINCIPAL) | (mapa_inf == INF_VIAL_SECUNDARIA)
    mask_fab = (mapa_inf == INF_FABRICA)

    E[mask_via] = f["trafico"] * intensidad * factor_trafico
    E[mask_fab] = f["industrial"] * factor_industrial  # fábrica = casi constante
    return E


# ---------------------------------------------------------------------------
# 4. MODELO DE DISPERSION (advección-difusión 2D)
# ---------------------------------------------------------------------------

def wind_to_uv(speed_ms, direction_deg):
    """
    Convierte velocidad/dirección meteorológica a componentes (u, v).

    direction_deg = dirección DESDE la que viene el viento (norma meteorológica).
    Retorna componentes HACIA donde sopla:
      u = componente en +longitud (este positivo)
      v = componente en +latitud  (norte positivo)
    """
    rad = np.radians((direction_deg + 180.0) % 360.0)
    u = speed_ms * np.sin(rad)   # E-W (j creciente)
    v = speed_ms * np.cos(rad)   # N-S (i creciente)
    return float(u), float(v)


def _setup_dispersion(grid, emisiones, wind_speed_ms, wind_direction_deg,
                      temperatura_c, presion_hpa, tiempo_simulado_s,
                      inversion_termica, landuse_map=None):
    """
    Calcula todos los parámetros del esquema numérico.

    Si se pasa `landuse_map`, construye matrices 2-D de D, u y v
    usando landuse.build_wind_maps (fricción diferencial por celda).

    Returns:
        dict con dx, u, v, D (escalares base), D_map/u_map/v_map
        (matrices o None), S_scaled, k_decay, dt_safe, n_steps.
    """
    dx = float(grid["cell_size_m"])
    u, v = wind_to_uv(wind_speed_ms, wind_direction_deg)

    # Coeficiente de difusión turbulenta base (m²/s): D ≈ c·|U|·L.
    D_base = 0.30 * max(wind_speed_ms, 0.3) * dx

    # Modulación por presión y temperatura.
    factor_presion = np.clip((presion_hpa - 1000.0) / 25.0, -0.5, 1.5)
    factor_temp = np.clip((20.0 - temperatura_c) / 25.0, -0.3, 0.8)
    factor_concentracion = 1.0 + 0.6 * factor_presion + 0.5 * factor_temp
    factor_concentracion = float(np.clip(factor_concentracion, 0.5, 3.0))

    D = D_base
    if inversion_termica:
        factor_concentracion *= 2.2
        D *= 0.30  # mezcla turbulenta severamente reducida
        D_base = D  # las matrices se escalan a partir del D ya corregido

    # -- Mapas espaciales de D, u, v por uso de suelo --
    if landuse_map is not None:
        from landuse import build_wind_maps  # importación local para evitar circular
        D_map, u_map, v_map = build_wind_maps(grid, landuse_map, D_base, u, v)
        # CFL usando el máximo de velocidad y difusión en toda la malla
        u_max = float(np.abs(u_map).max()) + 1e-6
        v_max = float(np.abs(v_map).max()) + 1e-6
        D_max = float(D_map.max()) + 1e-6
    else:
        D_map = u_map = v_map = None
        u_max = abs(u) + 1e-6
        v_max = abs(v) + 1e-6
        D_max = D + 1e-6

    # Estabilidad CFL (factor de seguridad 0.4).
    dt_adv = dx / (u_max + v_max)
    dt_dif = dx * dx / (4.0 * D_max)
    dt_safe = max(min(0.4 * dt_adv, 0.4 * dt_dif), 0.05)
    n_steps = max(1, int(np.ceil(tiempo_simulado_s / dt_safe)))

    # Tasa de pérdida efectiva (deposición + dispersión vertical + química).
    k_decay = 3.0e-3
    if inversion_termica:
        k_decay *= 0.35

    return {
        "dx": dx, "u": u, "v": v, "D": D,
        "D_map": D_map, "u_map": u_map, "v_map": v_map,
        "S_scaled": emisiones * factor_concentracion,
        "k_decay": k_decay, "dt_safe": dt_safe, "n_steps": n_steps,
    }


def _dispersion_step(C, p):
    """
    Avanza un paso temporal del esquema advección-difusión.

    Si p contiene D_map/u_map/v_map (matrices 2-D) las usa celda a celda;
    de lo contrario usa los escalares D, u, v (compatibilidad hacia atrás).
    """
    dx = p["dx"]
    # Frontera abierta: aire entrante trae C=0 (aire limpio regional).
    Cp = np.pad(C, 1, mode="constant", constant_values=0.0)

    # --- Advección upwind (u y v pueden ser escalares o matrices) ---
    if p["u_map"] is not None:
        u_map = p["u_map"]
        v_map = p["v_map"]
        # Upwind vectorizado: elige diferencia backward/forward según signo
        dCdx = np.where(
            u_map >= 0,
            (Cp[1:-1, 1:-1] - Cp[1:-1, :-2]) / dx,
            (Cp[1:-1, 2:]   - Cp[1:-1, 1:-1]) / dx,
        )
        dCdy = np.where(
            v_map >= 0,
            (Cp[1:-1, 1:-1] - Cp[:-2, 1:-1]) / dx,
            (Cp[2:,   1:-1] - Cp[1:-1, 1:-1]) / dx,
        )
        adv = -u_map * dCdx - v_map * dCdy
    else:
        u, v = p["u"], p["v"]
        dCdx = (Cp[1:-1, 1:-1] - Cp[1:-1, :-2]) / dx if u >= 0 else (Cp[1:-1, 2:] - Cp[1:-1, 1:-1]) / dx
        dCdy = (Cp[1:-1, 1:-1] - Cp[:-2, 1:-1]) / dx if v >= 0 else (Cp[2:, 1:-1] - Cp[1:-1, 1:-1]) / dx
        adv = -u * dCdx - v * dCdy

    # --- Difusión: laplaciano de 5 puntos (D puede ser escalar o matriz) ---
    lap = (Cp[1:-1, :-2] + Cp[1:-1, 2:] +
           Cp[:-2, 1:-1] + Cp[2:, 1:-1] -
           4.0 * Cp[1:-1, 1:-1]) / (dx * dx)
    D_eff = p["D_map"] if p["D_map"] is not None else p["D"]
    diff = D_eff * lap

    C = C + p["dt_safe"] * (adv + diff + p["S_scaled"] - p["k_decay"] * C)
    np.maximum(C, 0.0, out=C)
    return C


def run_dispersion(
    grid,
    emisiones,
    wind_speed_ms,
    wind_direction_deg,
    temperatura_c,
    presion_hpa,
    tiempo_simulado_s=900.0,
    inversion_termica=False,
    C_inicial=None,
    landuse_map=None,
):
    """
    Resuelve la ecuación de advección-difusión:
        ∂C/∂t = -u(x,y)·∂C/∂x - v(x,y)·∂C/∂y + D(x,y)·∇²C + S - k·C

    Esquema explícito en diferencias finitas:
      - Advección: upwind de 1er orden (estable), con u/v por celda si
                   se proporciona landuse_map.
      - Difusión:  laplaciano centrado con D variable por celda.
      - Frontera:  aire entrante = 0 (aire regional limpio)

    Args:
        grid: dict de build_grid
        emisiones: matriz S (fuentes)
        wind_speed_ms: |U| (m/s)
        wind_direction_deg: dirección meteorológica (desde)
        temperatura_c, presion_hpa: variables ambientales
        tiempo_simulado_s: duración física a integrar (default 15 min)
        inversion_termica: True modela inversión térmica
        C_inicial: matriz de concentración previa (arrastre hora a hora).
                   Si es None, empieza desde 0.
        landuse_map: matriz LUSE_* de landuse.build_landuse_map.
                     Si se pasa, D, u y v varían celda a celda según el
                     tipo de uso de suelo (arbolado, edificios, etc.).

    Returns:
        C: matriz de concentración (μg/m³ aproximado)
    """
    p = _setup_dispersion(grid, emisiones, wind_speed_ms, wind_direction_deg,
                          temperatura_c, presion_hpa, tiempo_simulado_s,
                          inversion_termica, landuse_map=landuse_map)
    if C_inicial is not None and C_inicial.shape == emisiones.shape:
        C = C_inicial.astype(np.float32, copy=True)
    else:
        C = np.zeros_like(emisiones, dtype=np.float32)

    for _ in range(p["n_steps"]):
        C = _dispersion_step(C, p)
    return C


def run_dispersion_animated(
    grid,
    emisiones,
    wind_speed_ms,
    wind_direction_deg,
    temperatura_c,
    presion_hpa,
    tiempo_simulado_s=600.0,
    n_frames=40,
    inversion_termica=False,
    C_inicial=None,
    landuse_map=None,
):
    """
    Igual que run_dispersion pero captura instantáneas a lo largo de la
    integración, para animar el movimiento de la pluma "en tiempo casi real".

    Args:
        (idénticos a run_dispersion, incluyendo landuse_map opcional)
        tiempo_simulado_s: duración física a integrar (default 10 min)
        n_frames: número de instantáneas a capturar

    Returns:
        frames: lista de dicts {t_segundos, C}, ordenada cronológicamente.
                El primer frame es t=0 (estado inicial) y el último es el
                estado final.
    """
    p = _setup_dispersion(grid, emisiones, wind_speed_ms, wind_direction_deg,
                          temperatura_c, presion_hpa, tiempo_simulado_s,
                          inversion_termica, landuse_map=landuse_map)
    if C_inicial is not None and C_inicial.shape == emisiones.shape:
        C = C_inicial.astype(np.float32, copy=True)
    else:
        C = np.zeros_like(emisiones, dtype=np.float32)

    n_steps = p["n_steps"]
    dt = p["dt_safe"]
    frame_every = max(1, n_steps // max(1, n_frames - 1))

    frames = [{"t_segundos": 0.0, "C": C.copy()}]
    for step in range(1, n_steps + 1):
        C = _dispersion_step(C, p)
        if step % frame_every == 0:
            frames.append({"t_segundos": step * dt, "C": C.copy()})
    # Garantizar el frame final
    if frames[-1]["t_segundos"] < n_steps * dt - 1e-6:
        frames.append({"t_segundos": n_steps * dt, "C": C.copy()})
    return frames




# ---------------------------------------------------------------------------
# 5. INDICE DE CALIDAD DEL AIRE (ICA) - NOM-172-SEMARNAT-2019 (simplificado)
# ---------------------------------------------------------------------------

# Breakpoints: (C_lo, C_hi, ICA_lo, ICA_hi)
ICA_BREAKPOINTS = {
    "PM2.5": [
        (0,    12.0, 0,   50),
        (12.1, 45.0, 51,  100),
        (45.1, 97.4, 101, 150),
        (97.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 500.0, 301, 500),
    ],
    "PM10": [
        (0,   50,  0,   50),
        (51,  75,  51,  100),
        (76,  214, 101, 150),
        (215, 354, 151, 200),
        (355, 424, 201, 300),
        (425, 604, 301, 500),
    ],
    "NOx": [
        (0,   53,   0,   50),
        (54,  105,  51,  100),
        (106, 360,  101, 150),
        (361, 649,  151, 200),
        (650, 1249, 201, 300),
        (1250, 2049, 301, 500),
    ],
    "SO2": [
        (0,   35,  0,   50),
        (36,  75,  51,  100),
        (76,  185, 101, 150),
        (186, 304, 151, 200),
        (305, 604, 201, 300),
        (605, 1004, 301, 500),
    ],
}


def calculate_ica(concentracion, contaminante="PM2.5"):
    """Calcula matriz de ICA a partir de matriz de concentración."""
    bps = ICA_BREAKPOINTS[contaminante]
    ica = np.zeros_like(concentracion, dtype=np.float32)
    for (clo, chi, ilo, ihi) in bps:
        m = (concentracion >= clo) & (concentracion <= chi)
        ica[m] = ilo + (concentracion[m] - clo) / (chi - clo) * (ihi - ilo)
    # Saturar arriba del último breakpoint
    ica[concentracion > bps[-1][1]] = 500.0
    return ica


def categoria_ica(valor):
    """Devuelve (nombre, color_hex) para un valor de ICA."""
    if valor <= 50:
        return "Buena", "#00E400"
    if valor <= 100:
        return "Aceptable", "#FFFF00"
    if valor <= 150:
        return "Mala", "#FF7E00"
    if valor <= 200:
        return "Muy Mala", "#FF0000"
    if valor <= 300:
        return "Extr. Mala", "#8F3F97"
    return "Peligrosa", "#7E0023"


# ---------------------------------------------------------------------------
# 6. UTILIDAD: pipeline de un escenario completo
# ---------------------------------------------------------------------------

def simular_dia_completo(
    grid, roads,
    contaminante,
    perfil_meteo,
    factor_trafico=1.0, factor_industrial=1.0,
    es_dia_laboral=True,
    inversion_horas=None,
    minutos_por_hora=10,
):
    """
    Evolución 24 h con tráfico variable hora a hora.

    Para cada hora del día:
      1. Construye el mapa de tráfico (veh/h) con el perfil horario.
      2. Calcula la matriz de emisiones (tráfico + industrial).
      3. Avanza la simulación los segundos solicitados, partiendo del
         estado anterior. Así la contaminación SE ARRASTRA con el flujo:
         lo emitido a las 7 AM persiste y se transporta mientras crece
         la emisión a las 8 AM.

    Args:
        grid: rejilla
        roads: lista de vialidades (de traffic.fetch_osm_roads o DEFAULT_ROADS)
        contaminante: 'PM2.5'|'PM10'|'NOx'|'SO2'
        perfil_meteo: dict {hora -> {viento, direccion, temperatura, presion}}
                      o un único dict (clima constante)
        factor_trafico, factor_industrial: multiplicadores de escenario
        es_dia_laboral: bool
        inversion_horas: set/list de horas con inversión térmica activa
                        (default: 6, 7, 8 si invierno meteorológico)
        minutos_por_hora: minutos físicos a simular por cada paso horario
                         (10 min es suficiente para quasi-estacionario
                         dentro de la hora, manteniendo arrastre temporal)

    Returns:
        Lista de 24 dicts con keys: hora, traffic, emisiones, C, ica, alerta.
    """
    # Importes locales para evitar dependencia circular
    from traffic import build_traffic_map, emissions_from_traffic, industrial_emissions

    if inversion_horas is None:
        inversion_horas = set()
    else:
        inversion_horas = set(inversion_horas)

    # Permitir clima constante o por-hora
    if "hora" in (perfil_meteo if isinstance(perfil_meteo, dict) else {}):
        # un solo dict -> aplicar igual a todas las horas
        clima_const = perfil_meteo
        perfil_meteo = {h: clima_const for h in range(24)}
    elif not isinstance(perfil_meteo, dict) or not all(h in perfil_meteo for h in range(24)):
        # Asumir que es un único dict de clima
        clima_const = perfil_meteo
        perfil_meteo = {h: clima_const for h in range(24)}

    # Emisiones industriales son aproximadamente constantes (operación 24/7)
    E_ind = industrial_emissions(
        grid,
        lat_range=TERNIUM_AREA["lat_range"],
        lon_range=TERNIUM_AREA["lon_range"],
        contaminante=contaminante,
        factor=factor_industrial,
    )

    frames = []
    C_prev = None

    for hora in range(24):
        clima = perfil_meteo[hora]
        traffic = build_traffic_map(
            grid, roads, hora,
            es_dia_laboral=es_dia_laboral,
            factor_global=factor_trafico,
        )
        E_traf = emissions_from_traffic(
            traffic, contaminante, cell_size_m=grid["cell_size_m"],
        )
        E_total = E_traf + E_ind

        C = run_dispersion(
            grid, E_total,
            wind_speed_ms=clima["velocidad_viento"],
            wind_direction_deg=clima["direccion_viento"],
            temperatura_c=clima["temperatura"],
            presion_hpa=clima["presion"],
            tiempo_simulado_s=minutos_por_hora * 60,
            inversion_termica=(hora in inversion_horas),
            C_inicial=C_prev,
        )
        A = calculate_ica(C, contaminante)
        frames.append({
            "hora":       hora,
            "traffic":    traffic,
            "emisiones":  E_total,
            "C":          C,
            "ica":        A,
            "veh_total":  float(traffic.sum()),
        })
        C_prev = C  # arrastre temporal

    return frames


def simular_con_trafico(
    grid, roads,
    hora, contaminante,
    viento_ms, viento_dir, temperatura, presion,
    factor_trafico=1.0, factor_industrial=1.0,
    inversion_termica=False,
    es_dia_laboral=True,
    tiempo_simulado_s=900.0,
):
    """
    Snapshot a una hora específica usando el modelo de tráfico real.

    Equivalente a simular_escenario pero usa la red vial (OSM o respaldo)
    y los factores de emisión por vehículo en lugar de las constantes
    de AVENIDAS.
    """
    from traffic import build_traffic_map, emissions_from_traffic, industrial_emissions

    traffic = build_traffic_map(
        grid, roads, hora,
        es_dia_laboral=es_dia_laboral,
        factor_global=factor_trafico,
    )
    E_traf = emissions_from_traffic(
        traffic, contaminante, cell_size_m=grid["cell_size_m"]
    )
    E_ind = industrial_emissions(
        grid,
        lat_range=TERNIUM_AREA["lat_range"],
        lon_range=TERNIUM_AREA["lon_range"],
        contaminante=contaminante,
        factor=factor_industrial,
    )
    E = E_traf + E_ind

    C = run_dispersion(
        grid, E,
        wind_speed_ms=viento_ms,
        wind_direction_deg=viento_dir,
        temperatura_c=temperatura,
        presion_hpa=presion,
        tiempo_simulado_s=tiempo_simulado_s,
        inversion_termica=inversion_termica,
    )
    A = calculate_ica(C, contaminante)
    return {
        "traffic":     traffic,
        "emisiones":   E,
        "emisiones_traf": E_traf,
        "emisiones_ind": E_ind,
        "concentracion": C,
        "ica": A,
    }


def simular_animacion(
    grid, roads,
    hora, contaminante,
    viento_ms, viento_dir, temperatura, presion,
    factor_trafico=1.0, factor_industrial=1.0,
    inversion_termica=False,
    es_dia_laboral=True,
    tiempo_simulado_s=600.0,
    n_frames=36,
    C_inicial=None,
):
    """
    Genera la secuencia de frames para animar el movimiento de la pluma
    "en tiempo casi real" usando el modelo de tráfico.

    Parte de aire limpio (o de C_inicial) y captura n_frames instantáneas
    mientras la contaminación se emite desde las vialidades + la industria
    y es transportada por el viento.

    Returns:
        dict con:
          - frames: lista de {t_segundos, C, ica}
          - traffic: matriz de flujo vehicular usada
          - emisiones: matriz de emisiones total
          - meta: parámetros de la corrida
    """
    from traffic import build_traffic_map, emissions_from_traffic, industrial_emissions

    traffic = build_traffic_map(
        grid, roads, hora,
        es_dia_laboral=es_dia_laboral,
        factor_global=factor_trafico,
    )
    E_traf = emissions_from_traffic(
        traffic, contaminante, cell_size_m=grid["cell_size_m"]
    )
    E_ind = industrial_emissions(
        grid,
        lat_range=TERNIUM_AREA["lat_range"],
        lon_range=TERNIUM_AREA["lon_range"],
        contaminante=contaminante,
        factor=factor_industrial,
    )
    E = E_traf + E_ind

    raw_frames = run_dispersion_animated(
        grid, E,
        wind_speed_ms=viento_ms,
        wind_direction_deg=viento_dir,
        temperatura_c=temperatura,
        presion_hpa=presion,
        tiempo_simulado_s=tiempo_simulado_s,
        n_frames=n_frames,
        inversion_termica=inversion_termica,
        C_inicial=C_inicial,
    )

    frames = []
    for f in raw_frames:
        frames.append({
            "t_segundos": f["t_segundos"],
            "C": f["C"],
            "ica": calculate_ica(f["C"], contaminante),
        })

    return {
        "frames": frames,
        "traffic": traffic,
        "emisiones": E,
        "meta": {
            "hora": hora,
            "contaminante": contaminante,
            "viento_ms": viento_ms,
            "viento_dir": viento_dir,
            "tiempo_simulado_s": tiempo_simulado_s,
            "n_frames": len(frames),
        },
    }


def simular_escenario(
    grid, mapa_inf,
    hora, contaminante,
    viento_ms, viento_dir, temperatura, presion,
    factor_trafico=1.0, factor_industrial=1.0,
    inversion_termica=False,
    tiempo_simulado_s=900.0,
):
    """
    Pipeline conveniente: emisiones -> dispersión -> ICA.

    Devuelve dict con C (concentración) y A (ICA).
    """
    E = calculate_emissions(
        mapa_inf, hora, contaminante,
        factor_trafico=factor_trafico,
        factor_industrial=factor_industrial,
    )
    C = run_dispersion(
        grid, E,
        wind_speed_ms=viento_ms,
        wind_direction_deg=viento_dir,
        temperatura_c=temperatura,
        presion_hpa=presion,
        tiempo_simulado_s=tiempo_simulado_s,
        inversion_termica=inversion_termica,
    )
    A = calculate_ica(C, contaminante)
    return {"emisiones": E, "concentracion": C, "ica": A}
