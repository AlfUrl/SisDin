"""
Adquisición de datos meteorológicos vía Open-Meteo (API pública sin key).

Tal como lo establece la propuesta: la información meteorológica se obtiene
en tiempo real desde una API externa, y los datos históricos se utilizan en
la fase de validación.
"""
from __future__ import annotations
import requests
from datetime import datetime, timedelta
import streamlit as st


OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL    = "https://archive-api.open-meteo.com/v1/archive"


@st.cache_data(ttl=600, show_spinner=False)
def get_current_weather(lat: float = 25.7255, lon: float = -100.3118) -> dict:
    """
    Devuelve la observación meteorológica actual en el campus.

    En caso de error de red, devuelve valores típicos de Monterrey para
    permitir que la simulación continúe operando.
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "temperature_2m,wind_speed_10m,wind_direction_10m,"
                   "surface_pressure,relative_humidity_2m",
        "wind_speed_unit": "ms",
        "timezone": "America/Monterrey",
    }
    fallback = {
        "temperatura": 24.0,
        "velocidad_viento": 3.0,
        "direccion_viento": 90.0,
        "presion": 1013.0,
        "humedad": 50.0,
        "hora": datetime.now().hour,
        "fuente": "fallback (sin red)",
    }
    try:
        r = requests.get(OPEN_METEO_URL, params=params, timeout=8)
        r.raise_for_status()
        cur = r.json().get("current", {})
        return {
            "temperatura":      float(cur.get("temperature_2m", fallback["temperatura"])),
            "velocidad_viento": float(cur.get("wind_speed_10m", fallback["velocidad_viento"])),
            "direccion_viento": float(cur.get("wind_direction_10m", fallback["direccion_viento"])),
            "presion":          float(cur.get("surface_pressure", fallback["presion"])),
            "humedad":          float(cur.get("relative_humidity_2m", fallback["humedad"])),
            "hora":             datetime.now().hour,
            "fuente":           "Open-Meteo (tiempo real)",
        }
    except Exception as e:  # pragma: no cover
        fallback["fuente"] = f"fallback ({type(e).__name__})"
        return fallback


@st.cache_data(ttl=3600, show_spinner=False)
def get_hourly_forecast(lat: float = 25.7255, lon: float = -100.3118,
                       horas: int = 12) -> list[dict]:
    """
    Pronóstico horario para las próximas N horas (para alertas anticipadas).
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,wind_speed_10m,wind_direction_10m,surface_pressure",
        "wind_speed_unit": "ms",
        "forecast_days": 2,
        "timezone": "America/Monterrey",
    }
    out: list[dict] = []
    try:
        r = requests.get(OPEN_METEO_URL, params=params, timeout=8)
        r.raise_for_status()
        h = r.json().get("hourly", {})
        times = h.get("time", [])
        # Localizar el índice de la hora actual
        ahora_iso = datetime.now().strftime("%Y-%m-%dT%H:00")
        try:
            i0 = times.index(ahora_iso)
        except ValueError:
            i0 = 0
        for k in range(horas):
            idx = i0 + k
            if idx >= len(times):
                break
            t = datetime.fromisoformat(times[idx])
            out.append({
                "datetime":         t,
                "hora":             t.hour,
                "temperatura":      float(h["temperature_2m"][idx]),
                "velocidad_viento": float(h["wind_speed_10m"][idx]),
                "direccion_viento": float(h["wind_direction_10m"][idx]),
                "presion":          float(h["surface_pressure"][idx]),
            })
    except Exception:
        # fallback: replicar clima actual durante N horas
        base = get_current_weather(lat, lon)
        for k in range(horas):
            t = datetime.now() + timedelta(hours=k)
            out.append({
                "datetime":         t,
                "hora":             t.hour,
                "temperatura":      base["temperatura"],
                "velocidad_viento": base["velocidad_viento"],
                "direccion_viento": base["direccion_viento"],
                "presion":          base["presion"],
            })
    return out


def grados_a_cardinal(deg: float) -> str:
    """Convierte grados a punto cardinal (N, NE, E, ..., NW)."""
    cardinales = ["N", "NE", "E", "SE", "S", "SO", "O", "NO"]
    idx = int(((deg + 22.5) % 360) // 45)
    return cardinales[idx]


# ===========================================================================
# CALIDAD DEL AIRE EN VIVO — Open-Meteo Air Quality API (sin clave)
# ===========================================================================
# Endpoint: https://air-quality-api.open-meteo.com/v1/air-quality
# Fuente: CAMS (Copernicus Atmosphere Monitoring Service), Global ~45 km y
#         Europeo ~11 km. Para Monterrey aplica el dominio global.
# Datos: modelados a partir de satélites + emisiones + meteorología, NO son
#        mediciones directas en estación física. Aun así son la mejor fuente
#        AUTOMÁTICA disponible: actualizada cada 12 h, hasta 92 días al pasado
#        y 7 días de pronóstico, en μg/m³ directos. Equivalen al estándar
#        usado por reportes de calidad del aire en navegadores y apps.

OPEN_METEO_AIR_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"


@st.cache_data(ttl=900, show_spinner=False)  # 15 min: CAMS no se actualiza más rápido
def get_current_air_quality(lat: float = 25.7255,
                            lon: float = -100.3118) -> dict:
    """
    Devuelve las concentraciones actuales modeladas de contaminantes en el
    campus, vía Open-Meteo Air Quality (CAMS global). NO requiere API key.

    Returns:
        dict con: pm2_5, pm10, no2, so2, o3, co  (todos en μg/m³)
                  us_aqi (índice EPA EE.UU., 0-500)
                  fecha (timestamp del modelo)
                  fuente (string descriptivo)
                  ok (True si los datos vienen de la red, False si fallback)
    """
    params = {
        "latitude":  lat,
        "longitude": lon,
        "current": "pm2_5,pm10,carbon_monoxide,nitrogen_dioxide,"
                   "sulphur_dioxide,ozone,us_aqi",
        "timezone": "America/Monterrey",
    }
    fallback = {
        "pm2_5": None, "pm10": None, "no2": None, "so2": None,
        "o3": None, "co": None, "us_aqi": None,
        "fecha": None,
        "fuente": "fallback (sin red)",
        "ok": False,
    }
    try:
        r = requests.get(OPEN_METEO_AIR_URL, params=params, timeout=8)
        r.raise_for_status()
        cur = r.json().get("current", {})
        return {
            "pm2_5":  _flt(cur.get("pm2_5")),
            "pm10":   _flt(cur.get("pm10")),
            "no2":    _flt(cur.get("nitrogen_dioxide")),
            "so2":    _flt(cur.get("sulphur_dioxide")),
            "o3":     _flt(cur.get("ozone")),
            "co":     _flt(cur.get("carbon_monoxide")),
            "us_aqi": _flt(cur.get("us_aqi")),
            "fecha":  cur.get("time"),
            "fuente": "Open-Meteo · CAMS Global (modelo)",
            "ok": True,
        }
    except Exception as e:  # pragma: no cover
        fallback["fuente"] = f"fallback ({type(e).__name__})"
        return fallback


@st.cache_data(ttl=3600, show_spinner=False)
def get_air_quality_series(lat: float = 25.7255,
                           lon: float = -100.3118,
                           past_days: int = 7,
                           forecast_days: int = 1) -> dict:
    """
    Serie temporal horaria de PM2.5 / PM10 / NO2 / SO2 / O3, vía Open-Meteo.

    Por defecto trae 7 días pasados + 1 día futuro: ideal para comparar el
    historial medido vs. las simulaciones del modelo.

    Returns:
        dict con keys 'datetime', 'pm2_5', 'pm10', 'no2', 'so2', 'o3' (listas)
        + 'ok' (bool) y 'fuente' (string).
    """
    params = {
        "latitude":  lat,
        "longitude": lon,
        "hourly": "pm2_5,pm10,nitrogen_dioxide,sulphur_dioxide,ozone",
        "past_days": int(past_days),
        "forecast_days": int(forecast_days),
        "timezone": "America/Monterrey",
    }
    salida = {
        "datetime": [], "pm2_5": [], "pm10": [], "no2": [],
        "so2": [], "o3": [], "ok": False,
        "fuente": "Open-Meteo · CAMS Global",
    }
    try:
        r = requests.get(OPEN_METEO_AIR_URL, params=params, timeout=12)
        r.raise_for_status()
        h = r.json().get("hourly", {})
        salida["datetime"] = [datetime.fromisoformat(t)
                              for t in h.get("time", [])]
        salida["pm2_5"] = [_flt(x) for x in h.get("pm2_5", [])]
        salida["pm10"]  = [_flt(x) for x in h.get("pm10", [])]
        salida["no2"]   = [_flt(x) for x in h.get("nitrogen_dioxide", [])]
        salida["so2"]   = [_flt(x) for x in h.get("sulphur_dioxide", [])]
        salida["o3"]    = [_flt(x) for x in h.get("ozone", [])]
        salida["ok"] = bool(salida["datetime"])
    except Exception as e:  # pragma: no cover
        salida["fuente"] = f"fallback ({type(e).__name__})"
    return salida


def _flt(x):
    """float() seguro: None y NaN devuelven None."""
    if x is None:
        return None
    try:
        v = float(x)
        return v if v == v else None  # filtra NaN
    except (TypeError, ValueError):
        return None

