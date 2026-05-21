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
