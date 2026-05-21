"""
Integración opcional con la API de tráfico de TomTom.

TomTom Traffic Flow Segment Data devuelve, para un punto dado, la velocidad
actual vs. la velocidad en flujo libre. La razón entre ambas es un indicador
directo de congestión: cuando el tráfico está detenido o en "stop-and-go",
los vehículos emiten considerablemente más por kilómetro (ralentí, acelerones).

Este módulo:
  - consulta varios puntos sobre las vialidades principales
  - calcula un índice de congestión promedio
  - traduce ese índice a un multiplicador de emisión

La API requiere una clave gratuita (https://developer.tomtom.com/, free tier
de 2 500 peticiones/día). Si no se proporciona clave o la red falla, el
sistema sigue funcionando con el perfil horario sintético.
"""
from __future__ import annotations
import requests
import streamlit as st


TOMTOM_FLOW_URL = (
    "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/{zoom}/json"
)

# Puntos de muestreo sobre las vialidades principales del polígono de CU.
# Se eligieron sobre Av. Universidad, Nogalar, Fidel Velázquez y Alfonso Reyes.
PUNTOS_MUESTREO = [
    (25.7300, -100.3137, "Av. Universidad (norte)"),
    (25.7220, -100.3110, "Av. Universidad (sur)"),
    (25.7270, -100.3050, "Av. Nogalar"),
    (25.7280, -100.3010, "Av. Fidel Velázquez"),
    (25.7205, -100.3090, "Av. Alfonso Reyes"),
]


@st.cache_data(ttl=300, show_spinner="Consultando tráfico en TomTom…")
def fetch_tomtom_flow(api_key: str, lat: float, lon: float,
                      zoom: int = 12) -> dict | None:
    """
    Consulta el flujo de tráfico en un punto.

    Returns:
        dict con current_speed, free_flow_speed, congestion_ratio,
        confidence; o None si falla.
    """
    if not api_key:
        return None
    url = TOMTOM_FLOW_URL.format(zoom=zoom)
    params = {"point": f"{lat},{lon}", "key": api_key, "unit": "KMPH"}
    try:
        r = requests.get(url, params=params, timeout=8)
        r.raise_for_status()
        seg = r.json().get("flowSegmentData", {})
        cur = float(seg.get("currentSpeed", 0) or 0)
        free = float(seg.get("freeFlowSpeed", 0) or 0)
        if free <= 0:
            return None
        return {
            "current_speed":   cur,
            "free_flow_speed": free,
            "congestion_ratio": max(0.05, min(1.0, cur / free)),
            "confidence":      float(seg.get("confidence", 0) or 0),
        }
    except Exception:
        return None


def indice_congestion_zona(api_key: str,
                           puntos: list | None = None) -> dict:
    """
    Promedia el estado de tráfico en varios puntos de la zona.

    Returns:
        dict con:
          - disponible: bool
          - congestion_ratio: 1.0 = flujo libre, <1 = congestión
          - multiplicador_emision: factor a aplicar a las emisiones de tráfico
          - detalle: lista por punto
          - mensaje: texto descriptivo
    """
    if puntos is None:
        puntos = PUNTOS_MUESTREO

    if not api_key:
        return {
            "disponible": False,
            "congestion_ratio": 1.0,
            "multiplicador_emision": 1.0,
            "detalle": [],
            "mensaje": "Sin clave TomTom: se usa el perfil horario sintético.",
        }

    detalle = []
    ratios = []
    for (lat, lon, nombre) in puntos:
        flow = fetch_tomtom_flow(api_key, lat, lon)
        if flow:
            ratios.append(flow["congestion_ratio"])
            detalle.append({
                "punto": nombre,
                "velocidad_actual": flow["current_speed"],
                "velocidad_libre": flow["free_flow_speed"],
                "ratio": flow["congestion_ratio"],
            })

    if not ratios:
        return {
            "disponible": False,
            "congestion_ratio": 1.0,
            "multiplicador_emision": 1.0,
            "detalle": [],
            "mensaje": "TomTom no devolvió datos. Se usa el perfil sintético.",
        }

    ratio_medio = sum(ratios) / len(ratios)
    mult = congestion_a_multiplicador(ratio_medio)

    if ratio_medio > 0.85:
        estado = "fluido"
    elif ratio_medio > 0.6:
        estado = "moderado"
    elif ratio_medio > 0.4:
        estado = "congestionado"
    else:
        estado = "saturado"

    return {
        "disponible": True,
        "congestion_ratio": ratio_medio,
        "multiplicador_emision": mult,
        "detalle": detalle,
        "mensaje": (
            f"Tráfico {estado} ({ratio_medio*100:.0f}% de velocidad libre). "
            f"Multiplicador de emisión: ×{mult:.2f}."
        ),
    }


def congestion_a_multiplicador(congestion_ratio: float) -> float:
    """
    Traduce el índice de congestión a un multiplicador de emisión.

    Fundamento: en condiciones de "stop-and-go" la emisión por kilómetro
    de PM y NOx puede duplicarse respecto al flujo libre, por ralentí
    prolongado y ciclos de aceleración. La relación es no lineal:

        ratio = 1.0 (libre)        -> mult ≈ 1.0
        ratio = 0.6 (moderado)     -> mult ≈ 1.3
        ratio = 0.4 (congestionado)-> mult ≈ 1.6
        ratio = 0.2 (saturado)     -> mult ≈ 2.1

    Se interpola con una curva suave acotada en [1.0, 2.3].
    """
    r = max(0.05, min(1.0, congestion_ratio))
    # Multiplicador inverso al ratio, con forma convexa.
    mult = 1.0 + 1.4 * (1.0 - r) ** 1.5
    return float(min(2.3, max(1.0, mult)))
