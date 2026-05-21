# Simulador de Calidad del Aire — Ciudad Universitaria UANL

Motor de simulación predictiva de dispersión de contaminantes atmosféricos
sobre el polígono de Ciudad Universitaria (UANL, Monterrey), acoplado a un
**modelo de flujo vehicular** y con recomendaciones operativas para la
comunidad.

> **Equipo 11 · Brigada 003** · Modelado y Simulación de Sistemas Dinámicos

---

## ¿Qué hace?

1. **Discretiza** el área de estudio en una rejilla regular de **15 m × 15 m**
   (≈ 16 500 celdas).
2. Carga la **red vial** desde **OpenStreetMap** (Overpass API) o usa una red
   de respaldo curada de 11 vialidades.
3. Estima el **flujo vehicular** hora por hora a partir de la jerarquía OSM
   (`trunk`, `primary`, `secondary`…) y el perfil horario calibrado del Área
   Metropolitana de Monterrey.
4. Convierte el flujo a **emisiones** con factores g/km/vehículo estilo
   EMFAC/COPERT para una flota mexicana típica.
5. Añade la **fuente fija industrial** (planta Ternium) como área de emisión.
6. Obtiene **meteorología en tiempo real** vía Open-Meteo (temperatura,
   viento, presión).
7. Resuelve la **ecuación de advección–difusión 2D** con esquema explícito
   (upwind + laplaciano centrado), dt ajustado dinámicamente por CFL.
8. Calcula el **ICA** según NOM-172-SEMARNAT-2019 para PM2.5 / PM10 / NOx / SO2.
9. **Simula el día completo** (24 h) con **arrastre temporal**: el aire de la
   hora $t$ es el de la hora $t-1$ más nuevas emisiones, transportado y
   disipado.
10. Genera:
    - **Mapa de calor** del ICA sobre Folium
    - **Animación en tiempo casi real**: heatmap animado que muestra cómo
      se forma y se mueve la pluma frame a frame
    - **Evolución 24h** con slider horario y gráfico de doble eje
      (ICA + flujo vehicular)
    - **Pronóstico horario** (12 h) con detección de picos
    - **Rutas de menor exposición** (Dijkstra ponderado por ICA)
    - **Validación contra SIMA** (RMSE, sesgo, correlación, IOA)
    - **Alertas globales** y **recomendaciones de cubrebocas** por nivel
11. Integra opcionalmente **tráfico real de TomTom**: la congestión observada
    se traduce en un multiplicador de emisión.

---

## Arquitectura

```
simulator.py        Motor: grid, dispersión 2D, ICA, snapshot, evolución 24h,
                    animación de frames intermedios
traffic.py          Red vial OSM, flujo vehicular, emisiones (EMFAC)
tomtom.py           Tráfico real opcional (TomTom Traffic Flow API)
sima.py             Validación contra estaciones del SIMA Nuevo León
weather.py          Cliente Open-Meteo (tiempo real + pronóstico)
recommendations.py  Alertas, rutas (Dijkstra ponderado), cubrebocas
app.py              Interfaz Streamlit (7 tabs interactivos)
test_motor.py       Tests del motor base
test_trafico.py     Tests del modelo de tráfico + evolución 24h
test_avanzado.py    Tests de animación, TomTom y validación SIMA
requirements.txt    Dependencias
```

---

## Instalación y ejecución

```bash
python -m venv .venv
source .venv/bin/activate    # o .venv\Scripts\activate en Windows
pip install -r requirements.txt
streamlit run app.py
```

Para probar el motor sin la interfaz:

```bash
python test_motor.py     # tests de dispersión, ICA, rutas
python test_trafico.py   # tests de tráfico y evolución 24h
python test_avanzado.py  # tests de animación, TomTom y validación SIMA
```

---

## Animación en tiempo casi real

El módulo `run_dispersion_animated` ejecuta el mismo esquema numérico que
`run_dispersion` pero captura instantáneas cada cierto número de pasos. El
resultado es una secuencia de frames que la app convierte en un **heatmap
animado de Plotly** con botones de reproducción y slider temporal.

La animación parte de **aire limpio** y muestra cómo:
1. El contaminante se emite desde las vialidades y la industria.
2. El viento lo transporta (advección) — se ve la pluma desplazarse.
3. La difusión lo dispersa lateralmente.
4. El sistema tiende a un estado quasi-estacionario (típicamente en 5–10 min
   de tiempo simulado).

Para mantener el payload ligero, la rejilla se submuestrea ×2 en la
animación (de 121×136 a ~60×68), reduciendo el JSON de ~3.8 MB a ~1 MB sin
pérdida visual perceptible.

---

## Tráfico real con TomTom (opcional)

Con una clave gratuita de [TomTom Developer](https://developer.tomtom.com/)
(free tier: 2 500 peticiones/día), el simulador consulta la **Traffic Flow
Segment Data API** en varios puntos de las avenidas principales.

El cociente velocidad-actual / velocidad-en-flujo-libre es un índice directo
de congestión que se traduce a un **multiplicador de emisión**:

| Estado del tráfico | Ratio velocidad | Multiplicador emisión |
|---|---|---|
| Flujo libre | 1.00 | ×1.00 |
| Moderado | 0.70 | ×1.23 |
| Congestionado | 0.45 | ×1.57 |
| Saturado | 0.20 | ×2.00 |

Fundamento: en condiciones de *stop-and-go*, la emisión por kilómetro de PM
y NOx se duplica por ralentí prolongado y ciclos de aceleración.

Sin clave, el sistema funciona normalmente con el perfil horario sintético.

---

## Validación contra SIMA

El módulo `sima.py` permite contrastar el ciclo diario simulado con
observaciones reales del **Sistema Integral de Monitoreo Ambiental** de
Nuevo León. Calcula:

- **RMSE** y **MAE** — magnitud del error
- **Sesgo** — tendencia a sobre/subestimar
- **Correlación de Pearson** — concordancia en la forma temporal
- **Índice de concordancia de Willmott (IOA)** — 0 (nulo) a 1 (perfecto)
- **NMSE** — error cuadrático medio normalizado

Acepta un CSV exportado del portal SIMA o genera un conjunto sintético de
demostración. Nota metodológica: las estaciones del SIMA están fuera del
polígono de CU, por lo que la validación contrasta la **forma del ciclo
diario** y el orden de magnitud, no el valor absoluto puntual.

---

## Modelo físico

### Dispersión

$$
\frac{\partial C}{\partial t} =
\underbrace{-u\,\frac{\partial C}{\partial x} - v\,\frac{\partial C}{\partial y}}_{\text{advección}}
+ \underbrace{D\,\nabla^2 C}_{\text{difusión}}
+ \underbrace{S(x, y, t)}_{\text{fuentes}}
- \underbrace{k\,C}_{\text{pérdida}}
$$

| Símbolo | Significado | Origen |
| --- | --- | --- |
| $u, v$ | Viento (m/s) | Open-Meteo |
| $D$ | Difusión turbulenta (m²/s) | $D = 0.30 \cdot \lvert U \rvert \cdot \Delta x$ |
| $S$ | Emisiones (μg/m³/s) | Tráfico (EMFAC) + industria |
| $k$ | Pérdida efectiva (1/s) | Deposición + dispersión vertical (≈ 3×10⁻³) |

### Flujo vehicular → emisiones

Para cada celda atravesada por una vialidad:

$$Q_{\text{celda}}\;[\text{veh/h}] = \text{Cap}(\text{tipo}) \times n_{\text{carriles}} \times \phi(\text{hora}) \times \alpha_{\text{laboral}}$$

$$E_{\text{celda}}\;\left[\frac{\mu g}{m^3 \cdot s}\right] = \frac{Q \cdot FE\;[g/km] \cdot \Delta x\;[km] \cdot 10^6}{3600 \cdot \Delta x^2 \cdot H_{\text{mix}}}$$

Capacidades por categoría OSM (veh/h·carril):

| Categoría | Capacidad |
|---|---|
| `trunk` / `motorway` | 3 000 – 4 000 |
| `primary` | 2 000 |
| `secondary` | 1 200 |
| `tertiary` | 800 |
| `residential` | 250 |

Perfil horario (`HOURLY_PROFILE`): 2 % a las 3 AM, **100 % a las 8 AM** y
18 PM, 60–70 % entre 12 y 16 PM.

Factores de emisión (g/km/veh) para flota típica MTY:
PM2.5: 0.040 · PM10: 0.078 · NOx: 0.580 · SO2: 0.045.

### Resultados observados

| Escenario | ICA medio | ICA máx | Categoría |
|---|---|---|---|
| Madrugada (3 AM, viento 3 m/s) | 14 | 78 | Aceptable |
| Hora pico (8 AM, sin inversión) | 28 | 76 | Aceptable |
| Hora pico (8 AM, **con inversión**) | **76** | **168** | **Mala / Muy Mala** |
| Tarde (18 PM, viento 4 m/s) | 28 | 76 | Aceptable |

La diferencia hora pico con/sin inversión térmica (28 vs 76 ICA medio)
demuestra que la **estabilidad atmosférica** es tan importante como el
volumen de tráfico.

---

## Funcionalidades de recomendación

| Función | Ubicación | Algoritmo |
| --- | --- | --- |
| **Alertas globales** | `recommendations.generate_alert` | Umbral sobre ICA máximo |
| **Cubrebocas** | `recommendations.mask_recommendation` | Tabla por nivel ICA |
| **Rutas limpias** | `recommendations.find_clean_route` | Dijkstra 8-conectado con costo $w \cdot (\text{ICA}/100)^2 + (1-w)$ |
| **Pronóstico de picos** | `recommendations.detectar_picos` | Simulación rápida horaria + umbral 100 |

---

## Limitaciones reconocidas

- **2D**: la capa de mezcla se modela paramétricamente (toggle de inversión).
- **Factores de emisión calibrados** a partir de literatura general; para
  uso operativo requieren validación contra SIMA.
- **Perfil de tráfico genérico** del AMM. Para mayor precisión integrar:
  - TomTom Traffic Flow API (free tier 2 500 req/día)
  - Google Maps Distance Matrix API
- **Esquema upwind 1er orden** introduce algo de difusión numérica
  (aceptable para análisis de impacto, no para gradientes finos).

---

## Roadmap

- [x] Modelo de flujo vehicular con red OSM
- [x] Evolución 24h con arrastre temporal
- [x] Animación de la pluma en tiempo casi real
- [x] Integración con TomTom Traffic Flow API
- [x] Módulo de validación contra SIMA (métricas + comparación)
- [ ] Anidar el modelo en un dominio regional para validación absoluta
- [ ] Detección automática de inversión térmica desde perfil meteorológico
- [ ] Modelo 3D con capa de mezcla resuelta
- [ ] Sistema de suscripción y notificaciones push por zona

