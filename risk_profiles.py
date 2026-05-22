"""
Perfiles de riesgo para el sistema de recomendaciones.
Define los grupos de población, factores de sensibilidad, y la matriz de recomendaciones
específicas por nivel de contaminación y perfil de riesgo.
"""
from __future__ import annotations

# =====================================================================
# DEFINICIÓN DE GRUPOS DE RIESGO
# =====================================================================

RISK_GROUPS = {
    "GENERAL": {
        "nombre": "Población general",
        "descripcion": "Adultos sin condiciones preexistentes",
        "icono": "👤",
        "factor_sensibilidad": 1.0,
        "color": "#4a90d9",
    },
    "RESPIRATORIO": {
        "nombre": "Sensibles respiratorios",
        "descripcion": "Asma, EPOC, rinitis alérgica, fibrosis",
        "icono": "🫁",
        "factor_sensibilidad": 1.8,
        "color": "#e74c3c",
    },
    "INFANTIL": {
        "nombre": "Niños y adolescentes",
        "descripcion": "Menores de 15 años (pulmones en desarrollo)",
        "icono": "👶",
        "factor_sensibilidad": 1.5,
        "color": "#f39c12",
    },
    "ADULTO_MAYOR": {
        "nombre": "Adultos mayores",
        "descripcion": "60+ años (capacidad pulmonar reducida)",
        "icono": "🧓",
        "factor_sensibilidad": 1.6,
        "color": "#8e44ad",
    },
    "EMBARAZO": {
        "nombre": "Embarazadas",
        "descripcion": "Riesgo para desarrollo fetal",
        "icono": "🤰",
        "factor_sensibilidad": 1.7,
        "color": "#e84393",
    },
    "DEPORTISTA": {
        "nombre": "Deportistas",
        "descripcion": "Actividad física intensa (mayor tasa de ventilación)",
        "icono": "🏃",
        "factor_sensibilidad": 1.4,
        "color": "#27ae60",
    },
}

# =====================================================================
# UMBRALES DE NIVELES (NOM-172)
# =====================================================================

NIVELES_ICA = [
    {"id": "BUENA", "max": 50, "nombre": "Buena", "color": "#00b894"},
    {"id": "ACEPTABLE", "max": 100, "nombre": "Aceptable", "color": "#ffd23f"},
    {"id": "MALA", "max": 150, "nombre": "Mala", "color": "#ff9f1c"},
    {"id": "MUY_MALA", "max": 200, "nombre": "Muy Mala", "color": "#e63946"},
    {"id": "EXTREMADAMENTE_MALA", "max": 300, "nombre": "Extremadamente Mala", "color": "#8e24aa"},
    {"id": "PELIGROSA", "max": 9999, "nombre": "Peligrosa", "color": "#5e0035"},
]

def ica_efectivo(ica_real: float, grupos: list[str]) -> float:
    """Calcula el ICA percibido tomando el factor de sensibilidad más alto de los grupos."""
    if not grupos:
        return ica_real
    max_factor = max([RISK_GROUPS[g]["factor_sensibilidad"] for g in grupos])
    return ica_real * max_factor

def nivel_para_ica(ica_val: float) -> dict:
    """Devuelve la información del nivel correspondiente a un valor de ICA."""
    for nivel in NIVELES_ICA:
        if ica_val <= nivel["max"]:
            return nivel
    return NIVELES_ICA[-1]

# =====================================================================
# MATRIZ DE RECOMENDACIONES
# =====================================================================
# Estructura: RISK_RECOMMENDATIONS[nivel_id][categoria]
# Hemos simplificado a definir las recomendaciones por Nivel de Riesgo (el ICA efectivo ya posiciona al grupo en un nivel).

RISK_RECOMMENDATIONS = {
    "BUENA": {
        "actividad_exterior": "Sin restricciones para actividades al aire libre.",
        "ejercicio": "Puede realizar cualquier tipo de ejercicio o deporte.",
        "cubrebocas": "No es necesario el uso de cubrebocas.",
        "ventilacion": "Abra ventanas y ventile libremente los espacios interiores.",
        "transporte": "Puede transitar por cualquier vía, caminar o usar bicicleta sin restricciones.",
        "alerta_medica": None,
    },
    "ACEPTABLE": {
        "actividad_exterior": "Puede realizar la mayoría de sus actividades normales.",
        "ejercicio": "Ejercicio permitido. Reducir intensidad si nota molestias menores.",
        "cubrebocas": "Opcional. Recomendado solo si percibe irritación.",
        "ventilacion": "Puede ventilar de manera regular.",
        "transporte": "Sin mayores restricciones. Precaución en avenidas con tráfico pesado.",
        "alerta_medica": "Si pertenece a un grupo sensible y presenta síntomas, limite su exposición.",
    },
    "MALA": {
        "actividad_exterior": "Limite el tiempo en exteriores. Prefiera actividades breves.",
        "ejercicio": "Evite ejercicio intenso prolongado al aire libre. Prefiera espacios interiores.",
        "cubrebocas": "Recomendable el uso de mascarilla KN95 si permanecerá en el exterior.",
        "ventilacion": "Ventile solo brevemente y preferentemente en horas de menor tráfico.",
        "transporte": "Evite rutas de alto tráfico o caminar junto a vías principales. Cierre ventanillas en autos.",
        "alerta_medica": "Personas sensibles: mantengan su medicación a la mano. Posibles síntomas respiratorios.",
    },
    "MUY_MALA": {
        "actividad_exterior": "Evite actividades al aire libre. Permanezca en interiores si es posible.",
        "ejercicio": "SUSPENDA el ejercicio al aire libre. Realice actividad física únicamente en interiores.",
        "cubrebocas": "Obligatorio uso de mascarilla N95/KN95 si es indispensable salir.",
        "ventilacion": "MANTENGA LAS VENTANAS CERRADAS. Use aire acondicionado en modo recirculación o purificadores.",
        "transporte": "Use modos de transporte cerrados. Evite la bicicleta o caminatas largas.",
        "alerta_medica": "Riesgo alto de agravamiento de afecciones. Acuda a servicios médicos si experimenta dificultad para respirar.",
    },
    "EXTREMADAMENTE_MALA": {
        "actividad_exterior": "PELIGRO: Quedese en interiores. Salidas estrictamente limitadas a emergencias.",
        "ejercicio": "PROHIBIDO cualquier esfuerzo físico al aire libre. Limite el esfuerzo físico incluso en interiores.",
        "cubrebocas": "N95 OBLIGATORIO Y ESTRICTO para cualquier salida de emergencia.",
        "ventilacion": "PROHIBIDO VENTILAR. Selle ranuras en ventanas y puertas si percibe humo o fuerte olor.",
        "transporte": "Evite traslados a menos que sea una emergencia.",
        "alerta_medica": "ALERTA MÁXIMA DE SALUD. Todos los grupos pueden experimentar efectos graves.",
    },
    "PELIGROSA": {
        "actividad_exterior": "EMERGENCIA SANITARIA. Prohibido salir.",
        "ejercicio": "Reposo recomendado. No realice esfuerzos físicos.",
        "cubrebocas": "Uso de N95 bien ajustado incluso en traslados de emergencia muy cortos.",
        "ventilacion": "Aislamiento total del exterior. Utilice purificadores de aire a máxima potencia.",
        "transporte": "Tránsito suspendido o desaconsejado fuertemente.",
        "alerta_medica": "EMERGENCIA. Alto riesgo de efectos severos y potencialmente fatales. Busque atención médica inmediata ante síntomas.",
    }
}

def get_recomendaciones(ica_real: float, grupos: list[str]) -> dict:
    """
    Obtiene el conjunto de recomendaciones cruzando el ICA real con el perfil de los grupos seleccionados.
    """
    ica_eff = ica_efectivo(ica_real, grupos)
    nivel = nivel_para_ica(ica_eff)
    recs = RISK_RECOMMENDATIONS[nivel["id"]]
    
    return {
        "ica_real": ica_real,
        "ica_efectivo": ica_eff,
        "nivel": nivel,
        "recomendaciones": recs
    }

def get_all_groups_summary(ica_real: float) -> list[dict]:
    """
    Genera un resumen del estado para todos los grupos disponibles.
    Útil para el dashboard/semáforo.
    """
    resumen = []
    for g_id, g_info in RISK_GROUPS.items():
        ica_eff = ica_efectivo(ica_real, [g_id])
        nivel = nivel_para_ica(ica_eff)
        resumen.append({
            "id": g_id,
            "nombre": g_info["nombre"],
            "icono": g_info["icono"],
            "ica_efectivo": ica_eff,
            "nivel": nivel
        })
    return resumen
