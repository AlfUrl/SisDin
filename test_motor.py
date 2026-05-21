"""Test del motor de simulación sin depender de Streamlit."""
import sys
sys.modules['streamlit'] = type(sys)('streamlit')  # mock
import streamlit  # type: ignore
def _noop_decorator(*a, **kw):
    def deco(fn): return fn
    return deco
streamlit.cache_data = _noop_decorator  # type: ignore

import numpy as np
from simulator import (
    build_grid, build_mask, build_infrastructure,
    calculate_emissions, run_dispersion, calculate_ica,
    simular_escenario, INF_VIAL_PRINCIPAL, INF_FABRICA,
)
from recommendations import (
    generate_alert, mask_recommendation, find_clean_route, route_stats,
)

print("=" * 70)
print("TEST 1: Construcción de rejilla")
print("=" * 70)
grid = build_grid()
print(f"  filas x columnas: {grid['filas']} x {grid['columnas']}")
print(f"  celdas totales:   {grid['filas'] * grid['columnas']}")
print(f"  área aprox:       {grid['filas']*15} m x {grid['columnas']*15} m")

print("\n" + "=" * 70)
print("TEST 2: Máscara del polígono")
print("=" * 70)
mask = build_mask(grid)
print(f"  celdas dentro del polígono: {mask.sum()}")
print(f"  porcentaje del bounding box: {100*mask.mean():.1f}%")

print("\n" + "=" * 70)
print("TEST 3: Infraestructura")
print("=" * 70)
inf = build_infrastructure(grid)
print(f"  celdas vialidad: {(inf == INF_VIAL_PRINCIPAL).sum()}")
print(f"  celdas Ternium:  {(inf == INF_FABRICA).sum()}")

print("\n" + "=" * 70)
print("TEST 4: Emisiones (hora pico 8 AM, PM2.5)")
print("=" * 70)
E = calculate_emissions(inf, hora=8, contaminante="PM2.5")
print(f"  emisión máxima:  {E.max():.3f}")
print(f"  emisión media (donde >0): {E[E>0].mean():.3f}")
print(f"  celdas emisoras: {(E > 0).sum()}")

print("\n" + "=" * 70)
print("TEST 5: Dispersión (viento NE a 3 m/s, T=22°C, P=1013 hPa)")
print("=" * 70)
C = run_dispersion(
    grid, E,
    wind_speed_ms=3.0, wind_direction_deg=45.0,  # viento del NE
    temperatura_c=22.0, presion_hpa=1013.0,
    tiempo_simulado_s=900,
)
print(f"  concentración máxima:    {C.max():.2f} μg/m³")
print(f"  concentración media:     {C.mean():.2f} μg/m³")
print(f"  concentración media (dentro polígono): {C[mask].mean():.2f} μg/m³")
print(f"  % celdas > 12 μg/m³:     {100*(C > 12).mean():.1f}%")

print("\n" + "=" * 70)
print("TEST 6: ICA")
print("=" * 70)
A = calculate_ica(C, contaminante="PM2.5")
print(f"  ICA máximo:      {A.max():.1f}")
print(f"  ICA medio:       {A.mean():.1f}")
print(f"  ICA p95:         {np.percentile(A, 95):.1f}")

alert = generate_alert(A.max(), A.mean())
print(f"  Alerta global:   {alert['nivel']} {alert['icono']}")

print("\n" + "=" * 70)
print("TEST 7: Escenario - inversión térmica")
print("=" * 70)
res = simular_escenario(
    grid, inf, hora=7, contaminante="PM2.5",
    viento_ms=1.0, viento_dir=0, temperatura=5, presion=1022,
    inversion_termica=True,
)
print(f"  ICA máximo en inversión: {res['ica'].max():.1f}")
print(f"  ICA medio en inversión:  {res['ica'].mean():.1f}")
alert2 = generate_alert(res['ica'].max(), res['ica'].mean())
print(f"  Alerta:                  {alert2['nivel']}")

print("\n" + "=" * 70)
print("TEST 8: Ruta limpia entre dos puntos del polígono")
print("=" * 70)
# tomar dos puntos arbitrarios dentro de la máscara
indices = np.argwhere(mask)
start = tuple(indices[10])
end = tuple(indices[-10])
print(f"  desde {start} hasta {end}")
path_short = find_clean_route(A, mask, start, end, pollution_weight=0.0)
path_clean = find_clean_route(A, mask, start, end, pollution_weight=0.9)
if path_short and path_clean:
    s1 = route_stats(path_short, A)
    s2 = route_stats(path_clean, A)
    print(f"  ruta corta:  {s1['longitud_m']:.0f} m, ICA medio {s1['ica_medio']:.1f}, máx {s1['ica_max']:.1f}")
    print(f"  ruta limpia: {s2['longitud_m']:.0f} m, ICA medio {s2['ica_medio']:.1f}, máx {s2['ica_max']:.1f}")
    print(f"  Reducción ICA medio: {100*(1 - s2['ica_medio']/max(s1['ica_medio'],0.01)):.1f}%")
    print(f"  Costo extra distancia: +{100*(s2['longitud_m']/s1['longitud_m']-1):.1f}%")
else:
    print("  ❌ no se encontró ruta")

print("\n" + "=" * 70)
print("TEST 9: Cubrebocas")
print("=" * 70)
for ica_test in [30, 75, 120, 175, 250]:
    print(f"  ICA={ica_test}: {mask_recommendation(ica_test)}")

print("\n✅ Todos los tests ejecutaron sin errores.")
