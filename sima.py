"""
Validación del modelo contra datos del SIMA Nuevo León.

El Sistema Integral de Monitoreo Ambiental (SIMA) publica concentraciones
horarias de PM2.5, PM10, NO2, SO2, O3, CO en sus estaciones. Las estaciones
más cercanas al polígono de Ciudad Universitaria son:

    - Noroeste 2 / San Bernabé
    - Centro / Obispado
    - Sureste / La Pastora
    - Norte / Escobedo
    - Noreste / San Nicolás

Este módulo permite:
  - cargar un CSV de observaciones del SIMA (descargado de su portal)
  - alinear las observaciones con las predicciones del simulador
  - calcular métricas de validación (RMSE, MAE, sesgo, correlación,
    índice de concordancia de Willmott)

Formato de CSV esperado (flexible, se autodetecta):
    fecha,hora,estacion,PM2.5,PM10,NO2,SO2,O3
    2025-01-15,07:00,Centro,38.2,72.1,...
o bien el export estándar del portal SIMA con columnas por contaminante.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


# Coordenadas aproximadas de las estaciones SIMA relevantes.
ESTACIONES_SIMA = {
    "Centro / Obispado":      (25.6759, -100.3380),
    "Noroeste / San Bernabé": (25.7569, -100.3656),
    "Noreste / San Nicolás":  (25.7497, -100.2553),
    "Norte / Escobedo":       (25.7878, -100.3437),
    "Sureste / La Pastora":   (25.6675, -100.2477),
}


# ---------------------------------------------------------------------------
# Carga de datos
# ---------------------------------------------------------------------------

def cargar_datos_sima(csv_path_or_buffer, contaminante: str = "PM2.5") -> pd.DataFrame:
    """
    Carga un CSV del SIMA y lo normaliza a columnas estándar.

    Returns:
        DataFrame con columnas: datetime, estacion, valor
        (valor en μg/m³ para PM/SO2, ppb para NO2/O3 según el contaminante)
    """
    df = pd.read_csv(csv_path_or_buffer)
    df.columns = [c.strip() for c in df.columns]

    # Mapear nombres de contaminante comunes a la columna real
    alias = {
        "PM2.5": ["PM2.5", "PM25", "PM2_5", "pm25"],
        "PM10":  ["PM10", "pm10"],
        "NOx":   ["NO2", "NOX", "NOx", "no2"],
        "SO2":   ["SO2", "so2"],
    }
    col_contaminante = None
    for cand in alias.get(contaminante, [contaminante]):
        if cand in df.columns:
            col_contaminante = cand
            break
    if col_contaminante is None:
        raise ValueError(
            f"No se encontró columna para {contaminante}. "
            f"Columnas disponibles: {list(df.columns)}"
        )

    # Construir datetime
    if "datetime" in df.columns:
        dt = pd.to_datetime(df["datetime"], errors="coerce")
    elif "fecha" in df.columns and "hora" in df.columns:
        dt = pd.to_datetime(df["fecha"].astype(str) + " " + df["hora"].astype(str),
                            errors="coerce")
    elif "fecha" in df.columns:
        dt = pd.to_datetime(df["fecha"], errors="coerce")
    else:
        raise ValueError("El CSV debe incluir 'datetime' o 'fecha'[+'hora'].")

    estacion = df["estacion"] if "estacion" in df.columns else "desconocida"

    out = pd.DataFrame({
        "datetime": dt,
        "estacion": estacion,
        "valor": pd.to_numeric(df[col_contaminante], errors="coerce"),
    }).dropna(subset=["datetime", "valor"])

    return out.sort_values("datetime").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Métricas de validación
# ---------------------------------------------------------------------------

def metricas_validacion(obs: np.ndarray, sim: np.ndarray) -> dict:
    """
    Calcula métricas estándar de validación de modelos de calidad del aire.

    Args:
        obs: observaciones (SIMA)
        sim: predicciones del simulador (mismo orden temporal)

    Returns:
        dict con n, rmse, mae, sesgo, correlacion, ioa (índice de
        concordancia de Willmott), nmse (error cuadrático medio normalizado).
    """
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)
    mask = np.isfinite(obs) & np.isfinite(sim)
    obs, sim = obs[mask], sim[mask]
    n = len(obs)
    if n < 2:
        return {"n": n, "error": "Datos insuficientes para validar."}

    err = sim - obs
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    sesgo = float(np.mean(err))

    obs_mean = obs.mean()
    sim_mean = sim.mean()

    # Correlación de Pearson
    denom = np.std(obs) * np.std(sim)
    correlacion = float(np.mean((obs - obs_mean) * (sim - sim_mean)) / denom) \
        if denom > 0 else float("nan")

    # Índice de concordancia de Willmott (0 = nulo, 1 = perfecto)
    num = np.sum(err ** 2)
    den = np.sum((np.abs(sim - obs_mean) + np.abs(obs - obs_mean)) ** 2)
    ioa = float(1 - num / den) if den > 0 else float("nan")

    # Error cuadrático medio normalizado
    nmse = float(np.mean(err ** 2) / (obs_mean * sim_mean)) \
        if obs_mean > 0 and sim_mean > 0 else float("nan")

    return {
        "n": n,
        "rmse": rmse,
        "mae": mae,
        "sesgo": sesgo,
        "correlacion": correlacion,
        "ioa": ioa,
        "nmse": nmse,
        "obs_media": float(obs_mean),
        "sim_media": float(sim_mean),
    }


def interpretar_metricas(m: dict) -> str:
    """Devuelve una interpretación cualitativa de las métricas."""
    if "error" in m:
        return m["error"]
    partes = []
    # Sesgo
    if abs(m["sesgo"]) < 0.1 * m["obs_media"]:
        partes.append("sesgo bajo (modelo bien centrado)")
    elif m["sesgo"] > 0:
        partes.append(f"sobrestima en promedio {m['sesgo']:.1f} μg/m³")
    else:
        partes.append(f"subestima en promedio {abs(m['sesgo']):.1f} μg/m³")
    # Correlación
    r = m["correlacion"]
    if r > 0.8:
        partes.append("correlación temporal fuerte")
    elif r > 0.5:
        partes.append("correlación temporal moderada")
    else:
        partes.append("correlación temporal débil")
    # IOA
    if m["ioa"] > 0.8:
        partes.append("excelente concordancia (IOA > 0.8)")
    elif m["ioa"] > 0.6:
        partes.append("concordancia aceptable")
    else:
        partes.append("concordancia limitada — requiere recalibración")
    return "; ".join(partes) + "."


# ---------------------------------------------------------------------------
# Pipeline de validación
# ---------------------------------------------------------------------------

def validar_serie(df_sima: pd.DataFrame, estacion: str,
                  predicciones_por_hora: dict) -> dict:
    """
    Valida una serie temporal del simulador contra una estación SIMA.

    Args:
        df_sima: DataFrame de cargar_datos_sima
        estacion: nombre de la estación a validar
        predicciones_por_hora: dict {hora_int (0-23) -> valor_simulado}
                               valores que el simulador predice en la
                               ubicación de esa estación

    Returns:
        dict con metricas + serie alineada (para graficar).
    """
    sub = df_sima[df_sima["estacion"] == estacion].copy()
    if sub.empty:
        # Si no hay columna estación o no coincide, usar todo
        sub = df_sima.copy()

    sub["hora"] = sub["datetime"].dt.hour
    # Promediar observaciones por hora del día
    obs_por_hora = sub.groupby("hora")["valor"].mean()

    horas = sorted(set(obs_por_hora.index) & set(predicciones_por_hora.keys()))
    obs = np.array([obs_por_hora[h] for h in horas])
    sim = np.array([predicciones_por_hora[h] for h in horas])

    m = metricas_validacion(obs, sim)
    return {
        "metricas": m,
        "interpretacion": interpretar_metricas(m),
        "horas": horas,
        "observado": obs.tolist(),
        "simulado": sim.tolist(),
        "estacion": estacion,
    }


# ---------------------------------------------------------------------------
# Datos sintéticos de ejemplo (para demostración sin CSV real)
# ---------------------------------------------------------------------------

def generar_sima_ejemplo(contaminante: str = "PM2.5",
                         estacion: str = "Centro / Obispado",
                         dias: int = 3, semilla: int = 42) -> pd.DataFrame:
    """
    ⚠️ DATOS SINTÉTICOS — no son mediciones reales.

    Genera un DataFrame con la forma de un export del SIMA, usando un perfil
    diurno típico + ruido aleatorio. Sirve únicamente para DEMOSTRAR el flujo
    de validación cuando no se tiene aún un CSV real descargado. NO úsese
    para conclusiones sobre la calidad del aire.

    Para datos reales, ver:
        - cargar_csv_sinaica()  → CSV oficial descargado de SINAICA
        - cargar_csv_sima_nl()  → CSV del portal aire.nl.gob.mx
        - descargar_openaq()    → API pública de OpenAQ
    """
    rng = np.random.default_rng(semilla)
    base_perfil = np.array([
        18, 16, 15, 14, 15, 20,    # 0-5
        32, 48, 55, 45, 38, 35,    # 6-11
        33, 36, 34, 33, 38, 50,    # 12-17
        58, 52, 40, 32, 26, 21,    # 18-23
    ], dtype=float)
    escala = {"PM2.5": 1.0, "PM10": 1.9, "NOx": 1.4, "SO2": 0.6}
    base_perfil = base_perfil * escala.get(contaminante, 1.0)

    filas = []
    fecha0 = pd.Timestamp("2025-01-13")
    for d in range(dias):
        for h in range(24):
            valor = base_perfil[h] * (1 + rng.normal(0, 0.12)) + rng.normal(0, 2)
            filas.append({
                "fecha": (fecha0 + pd.Timedelta(days=d)).strftime("%Y-%m-%d"),
                "hora": f"{h:02d}:00",
                "estacion": estacion,
                contaminante if contaminante != "NOx" else "NO2":
                    round(max(0, valor), 1),
            })
    return pd.DataFrame(filas)


# ===========================================================================
# CARGA DE DATOS REALES
# ===========================================================================

def cargar_csv_sinaica(csv_path, contaminante: str = "PM2.5") -> pd.DataFrame:
    """
    Carga un CSV descargado de SINAICA (https://sinaica.inecc.gob.mx/scica/).

    El CSV de SINAICA tiene columnas típicas:
        Fecha,Hora,Estación,Parametro,Valor,Unidad,...

    Cómo obtener el CSV (paso a paso):
        1. Ir a https://sinaica.inecc.gob.mx/scica/
        2. Sistema → Nuevo León → Red Monterrey (MTY)
        3. Estación: la más cercana a CU es típicamente "Universidad" (UNI)
           o "San Nicolás" (SAN)
        4. Parámetro: PM2.5 (o el que necesites)
        5. Rango de fechas
        6. Click "Descargar" → se genera el CSV
    """
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]

    # Detectar nombres de columnas (SINAICA varía mayúsculas/acentos)
    col_fecha = next((c for c in df.columns
                      if c.lower().startswith("fecha")), None)
    col_hora = next((c for c in df.columns
                     if c.lower().startswith("hora")), None)
    col_est = next((c for c in df.columns
                    if "estac" in c.lower()), None)
    col_param = next((c for c in df.columns
                      if "param" in c.lower()), None)
    col_val = next((c for c in df.columns
                    if c.lower() in ("valor", "value", "concentracion")),
                   None)

    if not (col_fecha and col_hora and col_val):
        raise ValueError(
            f"CSV no parece de SINAICA. Columnas encontradas: {list(df.columns)}. "
            "Espera al menos Fecha, Hora y Valor."
        )

    # Filtrar al contaminante de interés si existe la columna parámetro
    if col_param:
        contaminante_norm = contaminante.upper().replace(".", "")
        df = df[df[col_param].astype(str).str.upper()
                .str.replace(".", "", regex=False)
                .str.contains(contaminante_norm.replace("PM25", "PM2"))]

    df["datetime"] = pd.to_datetime(
        df[col_fecha].astype(str) + " " + df[col_hora].astype(str),
        errors="coerce",
    )
    df = df.dropna(subset=["datetime"])
    df["valor"] = pd.to_numeric(df[col_val], errors="coerce")
    df["estacion"] = df[col_est] if col_est else "Desconocida"
    return df[["datetime", "estacion", "valor"]].dropna()


def cargar_csv_sima_nl(csv_path, contaminante: str = "PM2.5") -> pd.DataFrame:
    """
    Carga un CSV descargado del portal SIMA Nuevo León
    (https://aire.nl.gob.mx).

    Como obtenerlo:
        1. https://aire.nl.gob.mx/SIMA2017phpgoogle/mapasimaprbBing.php
           (mapa actual del SIMA)
        2. https://aire.nl.gob.mx/rep_estadisticas.html
           (datos históricos 2005-2018, exportables)
        3. Alternativa: http://aire.nl.gob.mx/sima/ (plataforma nueva)

    Cuando descargues el archivo, pasalo a esta función.
    """
    # La función cargar_datos_sima (más arriba) ya autodetecta el formato
    # común del SIMA, así que la usamos como base.
    return cargar_datos_sima(csv_path, contaminante=contaminante)


def descargar_openaq(api_key: str,
                     bbox: tuple = (-100.32, 25.71, -100.29, 25.74),
                     contaminante: str = "PM2.5",
                     dias: int = 7) -> pd.DataFrame:
    """
    Descarga datos de OpenAQ (https://openaq.org) para el área de CU.

    OpenAQ agrega datos oficiales de SIMA/SINAICA + sensores PurpleAir
    en una API limpia. Requiere registro gratuito en
    https://openaq.org/developers para obtener una API key.

    Args:
        api_key: clave gratuita de OpenAQ.
        bbox: caja delimitadora (lon_min, lat_min, lon_max, lat_max).
              El default cubre el polígono de Ciudad Universitaria UANL.
        contaminante: 'PM2.5', 'PM10', 'NO2', 'SO2', 'O3', 'CO'.
        dias: cuántos días hacia atrás traer.

    Returns:
        DataFrame con columnas datetime, estacion, valor (en μg/m³).
    """
    import requests
    from datetime import datetime, timedelta, timezone

    # Mapeo de contaminante a parameter_id de OpenAQ v3
    param_id = {
        "PM2.5": 2, "PM10": 1, "O3": 10, "NO2": 7,
        "SO2": 9, "CO": 8,
    }.get(contaminante)
    if param_id is None:
        raise ValueError(f"Contaminante {contaminante!r} no soportado.")

    headers = {"X-API-Key": api_key}
    base = "https://api.openaq.org/v3"

    # 1. Buscar estaciones en la bbox
    bbox_str = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"
    r = requests.get(
        f"{base}/locations",
        params={"bbox": bbox_str, "parameters_id": param_id, "limit": 100},
        headers=headers, timeout=20,
    )
    r.raise_for_status()
    estaciones = r.json().get("results", [])
    if not estaciones:
        raise RuntimeError(
            f"OpenAQ no encontró estaciones de {contaminante} en la bbox "
            f"{bbox_str}. Prueba ampliarla."
        )

    # 2. Traer mediciones de cada sensor relevante
    desde = (datetime.now(timezone.utc) - timedelta(days=dias)).isoformat()
    filas = []
    for est in estaciones:
        for sensor in est.get("sensors", []):
            if sensor.get("parameter", {}).get("id") != param_id:
                continue
            sid = sensor["id"]
            r2 = requests.get(
                f"{base}/sensors/{sid}/measurements",
                params={"datetime_from": desde, "limit": 1000},
                headers=headers, timeout=20,
            )
            if not r2.ok:
                continue
            for m in r2.json().get("results", []):
                filas.append({
                    "datetime": pd.to_datetime(
                        m["period"]["datetimeTo"]["utc"]),
                    "estacion": est.get("name", f"OpenAQ#{est.get('id')}"),
                    "valor": m["value"],
                })

    if not filas:
        raise RuntimeError("OpenAQ devolvió 0 mediciones para el rango pedido.")

    return pd.DataFrame(filas).sort_values("datetime").reset_index(drop=True)

