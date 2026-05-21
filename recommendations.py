"""
Sistema de recomendaciones derivado del mapa de contaminación.

Funcionalidades:
  - generate_alert      : nivel de alerta global a partir del ICA máximo/medio
  - mask_recommendation : recomendación de cubrebocas según ICA local
  - find_clean_route    : ruta de menor exposición (Dijkstra ponderado por ICA)
  - hourly_pollution_forecast : detecta picos de contaminación en próximas horas
"""
from __future__ import annotations
from heapq import heappush, heappop
import numpy as np


# ---------------------------------------------------------------------------
# Alertas globales
# ---------------------------------------------------------------------------

def generate_alert(max_ica: float, mean_ica: float) -> dict:
    """Alerta global a partir del ICA máximo y promedio."""
    if max_ica > 200:
        return {
            "nivel": "EMERGENCIA",
            "color": "#7E0023",
            "icono": "🚨",
            "mensaje": (
                "Calidad del aire peligrosa en zonas del campus. "
                "SUSPENDE actividades al aire libre. Cierra ventanas. "
                "Personas con asma, EPOC, embarazadas y adultos mayores: "
                "permanece en interiores."
            ),
        }
    if max_ica > 150:
        return {
            "nivel": "ALTO",
            "color": "#FF0000",
            "icono": "⚠️",
            "mensaje": (
                "Calidad del aire muy mala en parte del polígono. "
                "Evita actividades físicas exteriores intensas. "
                "Usa cubrebocas N95/KN95 si transitas por las zonas críticas."
            ),
        }
    if max_ica > 100:
        return {
            "nivel": "MODERADO",
            "color": "#FF7E00",
            "icono": "🟠",
            "mensaje": (
                "Calidad del aire mala en algunas áreas. Grupos sensibles "
                "(asmáticos, niños, adultos mayores) deben limitar la "
                "exposición prolongada en exteriores."
            ),
        }
    if max_ica > 50:
        return {
            "nivel": "ACEPTABLE",
            "color": "#FFFF00",
            "icono": "🟡",
            "mensaje": (
                "Calidad del aire aceptable. Personas extremadamente "
                "sensibles podrían experimentar molestias menores."
            ),
        }
    return {
        "nivel": "BUENA",
        "color": "#00E400",
        "icono": "✅",
        "mensaje": "Calidad del aire buena. Sin restricciones para actividades al aire libre.",
    }


# ---------------------------------------------------------------------------
# Recomendación de cubrebocas
# ---------------------------------------------------------------------------

def mask_recommendation(ica_local: float) -> str:
    """Devuelve recomendación de mascarilla en función del ICA local."""
    if ica_local > 200:
        return "🚨 Mascarilla N95 obligatoria. Idealmente, permanece en interiores."
    if ica_local > 150:
        return "⚠️ Mascarilla N95 recomendada en exteriores."
    if ica_local > 100:
        return "🟠 KN95 recomendada para grupos sensibles (asma, alergias, niños)."
    if ica_local > 50:
        return "🟡 Cubrebocas opcional, solo si presentas alergias o irritación."
    return "✅ No es necesario cubrebocas por calidad del aire."


# ---------------------------------------------------------------------------
# Recomendaciones detalladas (estado actual + contexto climático)
# ---------------------------------------------------------------------------

def _categoria_y_color(ica: float) -> tuple[str, str, str]:
    """Devuelve (nombre_categoría, color_hex, icono)."""
    if ica <= 50:
        return "Buena", "#00b894", "✅"
    if ica <= 100:
        return "Aceptable", "#ffd23f", "🟡"
    if ica <= 150:
        return "Mala", "#ff9f1c", "🟠"
    if ica <= 200:
        return "Muy Mala", "#e63946", "🔴"
    if ica <= 300:
        return "Extremadamente Mala", "#8e24aa", "🟣"
    return "Peligrosa", "#5e0035", "☠️"


def recomendaciones_detalladas(
    ica_max: float,
    ica_medio: float,
    contaminante: str,
    viento_ms: float,
    temperatura: float,
    presion: float,
    inversion: bool = False,
) -> dict:
    """
    Construye un panel completo de recomendaciones a partir del estado de
    calidad del aire y las condiciones meteorológicas.

    Returns:
        dict con secciones: categoria, color, icono, resumen, evita, puedes,
        cubrebocas, grupos_sensibles, ventilacion, contexto_clima.
    """
    categoria, color, icono = _categoria_y_color(ica_max)

    # --- Evita / Puedes ---
    if ica_max > 200:
        evita = [
            "Toda actividad física al aire libre",
            "Salir de casa salvo emergencia",
            "Abrir ventanas o ventilar interiores",
        ]
        puedes = [
            "Permanecer en interiores con aire filtrado",
            "Reducir actividad física también en interiores",
        ]
    elif ica_max > 150:
        evita = [
            "Ejercicio intenso al aire libre (correr, ciclismo)",
            "Actividades prolongadas en exteriores",
            "Trayectos a pie cerca de avenidas con mucho tráfico",
        ]
        puedes = [
            "Caminatas cortas usando mascarilla",
            "Actividades en interiores",
            "Tomar rutas alternativas (ver pestaña de Rutas)",
        ]
    elif ica_max > 100:
        evita = [
            "Ejercicio aeróbico intenso al aire libre",
            "Estar cerca de avenidas en hora pico sin protección",
        ]
        puedes = [
            "Actividades moderadas al aire libre",
            "Caminar por zonas verdes alejadas de vialidades",
            "Hacer ejercicio en interiores",
        ]
    elif ica_max > 50:
        evita = [
            "Esfuerzo prolongado si tienes asma o alergias",
        ]
        puedes = [
            "Casi todas las actividades habituales al aire libre",
            "Ejercicio moderado a intenso si no eres sensible",
        ]
    else:
        evita = ["—"]
        puedes = [
            "Todas las actividades al aire libre",
            "Ejercicio sin restricciones",
            "Ventilar interiores libremente",
        ]

    # --- Cubrebocas ---
    if ica_max > 200:
        cubrebocas = {
            "tipo": "N95 obligatorio",
            "color": "#b00020",
            "icono": "🚨",
            "detalle": "Toda la población debe usar N95 si sale; "
                       "considera quedarte en interiores.",
        }
    elif ica_max > 150:
        cubrebocas = {
            "tipo": "N95 recomendado",
            "color": "#e63946",
            "icono": "⚠️",
            "detalle": "N95 al transitar zonas con ICA alto; KN95 mínimo.",
        }
    elif ica_max > 100:
        cubrebocas = {
            "tipo": "KN95 para grupos sensibles",
            "color": "#ff9f1c",
            "icono": "🟠",
            "detalle": "Población general: opcional. Asma/alergias/niños/"
                       "adultos mayores: KN95 al salir.",
        }
    elif ica_max > 50:
        cubrebocas = {
            "tipo": "Opcional",
            "color": "#ffd23f",
            "icono": "🟡",
            "detalle": "Solo si presentas irritación o tienes alergias.",
        }
    else:
        cubrebocas = {
            "tipo": "No es necesario",
            "color": "#00b894",
            "icono": "✅",
            "detalle": "El aire está limpio.",
        }

    # --- Grupos sensibles (asma, EPOC, niños, adultos mayores, embarazadas) ---
    if ica_max > 150:
        grupos_sensibles = (
            "**Permanece en interiores.** Si tienes asma o EPOC, ten tu "
            "medicamento de rescate a la mano y evita esfuerzos."
        )
    elif ica_max > 100:
        grupos_sensibles = (
            "**Limita la exposición al aire libre.** Evita ejercicio fuera; "
            "usa KN95 si necesitas salir. Niños no deben hacer deporte exterior."
        )
    elif ica_max > 50:
        grupos_sensibles = (
            "**Precaución.** Si presentas síntomas (tos, ojos irritados, "
            "opresión en el pecho), reduce la exposición."
        )
    else:
        grupos_sensibles = "Sin precauciones especiales hoy."

    # --- Ventilación ---
    if ica_max > 150:
        ventilacion = (
            "🚪 **Mantén ventanas cerradas.** Usa purificador con filtro HEPA "
            "si tienes; evita encender ventilador hacia el exterior."
        )
    elif ica_max > 100:
        ventilacion = (
            "🪟 Ventila solo brevemente y en horas de mejor calidad del aire "
            "(madrugada o mediodía con viento)."
        )
    else:
        ventilacion = "🪟 Puedes ventilar normalmente."

    # --- Contexto climático: por qué está así, qué esperar ---
    razones = []
    if inversion:
        razones.append(
            "❄️ **Inversión térmica activa**: el aire frío atrapa los "
            "contaminantes cerca del suelo. Los niveles se mantendrán "
            "altos hasta que suba la temperatura (típicamente al mediodía)."
        )
    if viento_ms < 1.5:
        razones.append(
            f"🌬️ **Viento muy débil ({viento_ms:.1f} m/s)**: la pluma "
            "no se dispersa bien. Espera que los niveles bajen solo cuando "
            "aumente el viento."
        )
    elif viento_ms >= 5:
        razones.append(
            f"🌬️ **Viento fuerte ({viento_ms:.1f} m/s)**: buena dispersión, "
            "los niveles deberían mantenerse o mejorar."
        )
    if presion > 1018:
        razones.append(
            f"🔺 **Alta presión ({presion:.0f} hPa)**: atmósfera estable, "
            "menos mezcla vertical. Contribuye a mantener contaminación."
        )
    if temperatura < 10:
        razones.append(
            f"🥶 **Temperatura baja ({temperatura:.0f}°C)**: favorece "
            "estabilidad nocturna y mayor uso de calefacción."
        )

    if not razones:
        razones.append(
            "Condiciones meteorológicas favorables para la dispersión."
        )

    return {
        "ica_max": ica_max,
        "ica_medio": ica_medio,
        "categoria": categoria,
        "color": color,
        "icono": icono,
        "evita": evita,
        "puedes": puedes,
        "cubrebocas": cubrebocas,
        "grupos_sensibles": grupos_sensibles,
        "ventilacion": ventilacion,
        "contexto_clima": razones,
    }


def recomendaciones_pronostico(forecast_results: list[dict],
                               umbral: float = 100.0) -> list[dict]:
    """
    Convierte un pronóstico horario en recomendaciones por ventana de tiempo.
    Cada item incluye qué hacer durante esa hora.
    """
    out = []
    for r in forecast_results:
        ica = r.get("ica_max", 0)
        cat, color, icono = _categoria_y_color(ica)
        if ica > 150:
            accion = "Quédate en interiores"
        elif ica > 100:
            accion = "Limita exteriores · usa KN95 si sales"
        elif ica > 50:
            accion = "Actividades exteriores moderadas"
        else:
            accion = "Sin restricciones"
        out.append({
            "datetime": r["datetime"],
            "hora": r["hora"],
            "ica_max": ica,
            "categoria": cat,
            "color": color,
            "icono": icono,
            "accion": accion,
        })
    return out


# ---------------------------------------------------------------------------
# Factores ambientales: explicación de POR QUÉ el aire está como está
# ---------------------------------------------------------------------------

def _cardinal(deg: float) -> str:
    cardinales = ["N", "NE", "E", "SE", "S", "SO", "O", "NO"]
    return cardinales[int(((deg + 22.5) % 360) // 45)]


def factores_ambientales(viento_ms: float, viento_dir: float,
                         temperatura: float, presion: float,
                         hora: int, perfil_trafico_pct: float,
                         factor_trafico_efectivo: float,
                         inversion: bool) -> list[dict]:
    """
    Lista de factores que afectan la calidad del aire ahora, cada uno con:
       - icono, etiqueta, valor (texto), impacto ('bueno'|'neutro'|'malo'|'crítico'),
         mensaje explicativo.

    Pensado para mostrarse como tarjetas en la UI, así el usuario entiende
    POR QUÉ el aire está como está y qué cambiar para que mejore.
    """
    out: list[dict] = []

    # Viento
    if viento_ms >= 4.0:
        imp, msg = "bueno", "dispersión activa, los contaminantes son barridos rápidamente"
    elif viento_ms >= 2.0:
        imp, msg = "neutro", "dispersión moderada"
    else:
        imp, msg = "malo", "viento débil: los contaminantes se acumulan"
    out.append({
        "icono": "💨", "etiqueta": "Viento",
        "valor": f"{viento_ms:.1f} m/s desde {_cardinal(viento_dir)}",
        "impacto": imp, "mensaje": msg,
    })

    # Presión atmosférica
    if presion > 1020:
        imp, msg = "malo", "alta presión: atmósfera estable, poca mezcla vertical"
    elif presion < 1005:
        imp, msg = "bueno", "baja presión: mezcla atmosférica activa"
    else:
        imp, msg = "neutro", "presión normal"
    out.append({
        "icono": "🎚️", "etiqueta": "Presión",
        "valor": f"{presion:.0f} hPa", "impacto": imp, "mensaje": msg,
    })

    # Temperatura
    if temperatura < 8:
        imp, msg = "malo", "frío: favorece inversión térmica nocturna/matutina"
    elif temperatura > 30:
        imp, msg = "neutro", "calor: aumenta convección, mejora mezcla"
    else:
        imp, msg = "neutro", "temperatura templada"
    out.append({
        "icono": "🌡️", "etiqueta": "Temperatura",
        "valor": f"{temperatura:.1f}°C", "impacto": imp, "mensaje": msg,
    })

    # Tráfico
    if perfil_trafico_pct >= 90:
        imp, msg = "malo", "hora pico: emisión vehicular máxima"
    elif perfil_trafico_pct >= 50:
        imp, msg = "neutro", "tráfico moderado"
    else:
        imp, msg = "bueno", "tráfico ligero"
    if factor_trafico_efectivo > 1.25:
        msg += f" · congestión real (×{factor_trafico_efectivo:.2f})"
        if imp == "neutro":
            imp = "malo"
    out.append({
        "icono": "🚗", "etiqueta": "Tráfico",
        "valor": f"perfil {perfil_trafico_pct:.0f}% · {hora:02d}:00",
        "impacto": imp, "mensaje": msg,
    })

    # Inversión térmica (es el factor más crítico cuando ocurre)
    if inversion:
        out.append({
            "icono": "❄️", "etiqueta": "Inversión térmica",
            "valor": "ACTIVA", "impacto": "crítico",
            "mensaje": "concentra los contaminantes cerca del suelo (×2.2)",
        })
    else:
        out.append({
            "icono": "✅", "etiqueta": "Mezcla vertical",
            "valor": "normal", "impacto": "bueno",
            "mensaje": "los contaminantes se dispersan a las capas superiores",
        })

    return out


def recomendaciones_de_accion(max_ica: float, mean_ica: float,
                              factores: list[dict],
                              pico_proximas_horas: dict | None = None) -> list[str]:
    """
    Lista de recomendaciones de acción concretas para el usuario,
    derivadas del ICA actual + factores ambientales + pronóstico próximo.

    Returns:
        Lista de strings (cada uno una acción sugerida con su ícono).
    """
    recs: list[str] = []

    # Recomendación principal por nivel
    if max_ica > 200:
        recs.append("🚨 **Permanece en interiores** si es posible. Cierra "
                    "ventanas y evita ejercicio físico.")
    elif max_ica > 150:
        recs.append("⚠️ **Evita ejercicio al aire libre.** Si vas a salir, "
                    "usa mascarilla N95.")
    elif max_ica > 100:
        recs.append("🟠 **Personas sensibles** (asma, niños, mayores) deben "
                    "limitar el tiempo en exteriores.")
    elif max_ica > 50:
        recs.append("🟡 Calidad aceptable; toma precauciones solo si tienes "
                    "alergias o sensibilidad respiratoria.")
    else:
        recs.append("✅ Aire limpio: puedes realizar actividades al aire libre "
                    "sin restricciones.")

    # Recomendaciones basadas en factores específicos
    factores_malos = [f for f in factores if f["impacto"] in ("malo", "crítico")]
    if any(f["etiqueta"] == "Inversión térmica" for f in factores_malos):
        recs.append("❄️ La inversión térmica se rompe normalmente al subir "
                    "la temperatura: espera hasta media mañana para mejor calidad.")
    if any(f["etiqueta"] == "Viento" for f in factores_malos):
        recs.append("💨 El viento débil prolonga la contaminación. Las zonas "
                    "barlovento (hacia donde NO sopla) son las más afectadas.")
    if any(f["etiqueta"] == "Tráfico" for f in factores_malos):
        recs.append("🚗 Considera transitar por calles secundarias en lugar "
                    "de avenidas principales si caminas o vas en bicicleta.")

    # Recomendación basada en el pronóstico
    if pico_proximas_horas and pico_proximas_horas.get("hora_pico"):
        hp = pico_proximas_horas["hora_pico"]
        ica_pico = pico_proximas_horas["ica_pico"]
        if ica_pico > max_ica + 20:
            recs.append(
                f"📈 Se prevé un **empeoramiento** hacia las "
                f"**{hp:02d}:00** (ICA proyectado: {ica_pico:.0f}). "
                f"Si planeas salir, hazlo antes."
            )
        elif ica_pico + 20 < max_ica:
            recs.append(
                f"📉 La calidad **mejorará** hacia las **{hp:02d}:00** "
                f"(ICA proyectado: {ica_pico:.0f}). Si puedes posponer la "
                f"actividad, hazlo."
            )

    return recs


# ---------------------------------------------------------------------------
# Ruta de menor exposición (Dijkstra ponderado por ICA)
# ---------------------------------------------------------------------------

def find_clean_route(ica_map: np.ndarray,
                     mask: np.ndarray,
                     start_idx: tuple[int, int],
                     end_idx: tuple[int, int],
                     pollution_weight: float = 0.8) -> list[tuple[int, int]] | None:
    """
    Calcula una ruta entre `start_idx` y `end_idx` minimizando una función
    de costo:    costo_celda = (1 - w) · longitud + w · f(ICA)

    donde f(ICA) crece cuadráticamente con el ICA local. El parámetro
    `pollution_weight` ∈ [0,1] balancea distancia vs. limpieza:
       - 0.0 = ruta más corta (ignora contaminación)
       - 1.0 = ruta más limpia posible (puede ser larga)

    Args:
        ica_map: matriz de ICA (filas, columnas)
        mask:    máscara booleana del área transitable
        start_idx, end_idx: tuplas (i, j)
        pollution_weight: peso de la contaminación en el costo

    Returns:
        Lista de (i, j) que forma el camino, o None si no hay conexión.
    """
    rows, cols = ica_map.shape
    si, sj = start_idx
    ei, ej = end_idx

    if not (0 <= si < rows and 0 <= sj < cols and 0 <= ei < rows and 0 <= ej < cols):
        return None
    if not (mask[si, sj] and mask[ei, ej]):
        return None

    w = float(np.clip(pollution_weight, 0.0, 1.0))

    # Normalizamos ICA para que la penalización sea comparable a la unidad de paso
    ica_norm = ica_map / 100.0   # ICA=100 -> 1.0
    pollution_cost = (ica_norm ** 2) * 10.0   # crece rápido por encima de 100

    # Costos iniciales
    INF = np.float32(np.inf)
    dist = np.full((rows, cols), INF, dtype=np.float32)
    parent_i = np.full((rows, cols), -1, dtype=np.int32)
    parent_j = np.full((rows, cols), -1, dtype=np.int32)
    visited = np.zeros((rows, cols), dtype=bool)
    dist[si, sj] = 0.0

    # 8-conectividad (incluye diagonales)
    neighbors = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
                 (-1, -1, 1.4142), (-1, 1, 1.4142),
                 (1, -1, 1.4142), (1, 1, 1.4142)]

    heap: list[tuple[float, int, int]] = [(0.0, si, sj)]

    while heap:
        d, i, j = heappop(heap)
        if visited[i, j]:
            continue
        visited[i, j] = True
        if (i, j) == (ei, ej):
            break

        for di, dj, base in neighbors:
            ni, nj = i + di, j + dj
            if not (0 <= ni < rows and 0 <= nj < cols):
                continue
            if visited[ni, nj] or not mask[ni, nj]:
                continue
            step_cost = (1.0 - w) * base + w * float(pollution_cost[ni, nj]) * base
            nd = d + step_cost
            if nd < dist[ni, nj]:
                dist[ni, nj] = nd
                parent_i[ni, nj] = i
                parent_j[ni, nj] = j
                heappush(heap, (nd, ni, nj))

    if not np.isfinite(dist[ei, ej]):
        return None

    # Reconstruir camino
    path: list[tuple[int, int]] = []
    ci, cj = ei, ej
    while (ci, cj) != (si, sj):
        path.append((ci, cj))
        npi, npj = parent_i[ci, cj], parent_j[ci, cj]
        if npi < 0:
            return None
        ci, cj = int(npi), int(npj)
    path.append((si, sj))
    path.reverse()
    return path


def route_stats(path: list[tuple[int, int]], ica_map: np.ndarray,
                cell_size_m: float = 15.0) -> dict:
    """Estadísticas de una ruta: longitud, ICA promedio, máximo."""
    if not path:
        return {}
    icas = np.array([ica_map[i, j] for (i, j) in path])
    # Longitud aprox: suma de pasos con factor diagonal
    pasos = 0.0
    for k in range(1, len(path)):
        (a, b), (c, d) = path[k - 1], path[k]
        pasos += np.hypot(c - a, d - b)
    return {
        "longitud_m":  float(pasos * cell_size_m),
        "ica_medio":   float(icas.mean()),
        "ica_max":     float(icas.max()),
        "n_puntos":    len(path),
    }


# ---------------------------------------------------------------------------
# Pronóstico horario de contaminación (para alarmas anticipadas)
# ---------------------------------------------------------------------------

def hourly_pollution_forecast(simulate_fn, forecast_weather: list[dict],
                              contaminante: str = "PM2.5") -> list[dict]:
    """
    Ejecuta simulaciones rápidas (n_steps reducido) para cada hora del
    pronóstico y devuelve la evolución prevista del ICA.

    Args:
        simulate_fn: función f(hora, viento, dir, T, p) -> matriz ICA
        forecast_weather: lista de dicts (de weather.get_hourly_forecast)
        contaminante: identificador (solo informativo)

    Returns:
        Lista de dicts con timestamp e indicadores resumen.
    """
    results: list[dict] = []
    for w in forecast_weather:
        A = simulate_fn(
            hora=w["hora"],
            viento=w["velocidad_viento"],
            direccion=w["direccion_viento"],
            temperatura=w["temperatura"],
            presion=w["presion"],
        )
        results.append({
            "datetime":  w["datetime"],
            "hora":      w["hora"],
            "ica_max":   float(A.max()),
            "ica_medio": float(A.mean()),
            "ica_p95":   float(np.percentile(A, 95)),
            "contaminante": contaminante,
        })
    return results


def detectar_picos(forecast_results: list[dict], umbral: float = 100.0) -> list[dict]:
    """Identifica horas con ICA proyectado por encima del umbral."""
    picos = []
    for r in forecast_results:
        if r["ica_max"] >= umbral:
            picos.append({
                "datetime": r["datetime"],
                "hora": r["hora"],
                "ica_max": r["ica_max"],
                "severidad": "alta" if r["ica_max"] >= 150 else "moderada",
            })
    return picos
