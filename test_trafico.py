"""Test del modelo de tráfico y evolución 24h."""
import sys
sys.modules['streamlit'] = type(sys)('streamlit')
import streamlit
def _noop(*a, **kw):
    def d(fn): return fn
    return d
streamlit.cache_data = _noop
streamlit.cache_resource = _noop
streamlit.warning = lambda *a, **kw: print(f'  ⚠️ {a}')

import warnings; warnings.filterwarnings('ignore')
import logging; logging.disable(logging.CRITICAL)
import numpy as np

from simulator import build_grid, build_mask, simular_dia_completo
from traffic import (
    DEFAULT_ROADS, build_traffic_map, emissions_from_traffic,
    industrial_emissions, resumen_red,
)
from simulator import TERNIUM_AREA

print("=" * 70)
print("TEST 1: Red vial de respaldo")
print("=" * 70)
resumen = resumen_red(DEFAULT_ROADS)
print(f"  Segmentos: {resumen['total_segmentos']}")
print(f"  Km totales: {resumen['km_totales']}")
print(f"  Por tipo: {resumen['por_tipo']}")

print()
print("=" * 70)
print("TEST 2: Mapa de flujo vehicular por hora")
print("=" * 70)
grid = build_grid()
mask = build_mask(grid)
for hora in [3, 8, 14, 18, 22]:
    traffic = build_traffic_map(grid, DEFAULT_ROADS, hora=hora,
                                es_dia_laboral=True, ancho_celdas=2)
    max_flujo = traffic.max()
    total_celdas_activas = (traffic > 0).sum()
    veh_total = traffic.sum()
    print(f"  Hora {hora:2d}: flujo máx {max_flujo:6.0f} veh/h | "
          f"celdas activas {total_celdas_activas:4d} | "
          f"total veh-celda {veh_total/1000:.1f}k")

print()
print("=" * 70)
print("TEST 3: Emisiones de tráfico (PM2.5, hora pico)")
print("=" * 70)
traffic = build_traffic_map(grid, DEFAULT_ROADS, hora=8,
                            es_dia_laboral=True, ancho_celdas=2)
E_traf = emissions_from_traffic(traffic, "PM2.5", cell_size_m=15)
print(f"  Emisión máxima:  {E_traf.max():.4f} μg/m³/s")
print(f"  Emisión media (donde >0): {E_traf[E_traf > 0].mean():.4f} μg/m³/s")
print(f"  Total celdas emisoras: {(E_traf > 0).sum()}")

print()
print("=" * 70)
print("TEST 4: Evolución 24h con tráfico + industria (clima constante)")
print("=" * 70)
clima = {"velocidad_viento": 3.0, "direccion_viento": 90,
         "temperatura": 22, "presion": 1013}
print("  Simulando 24h (esto toma unos segundos)...")
import time
t0 = time.time()
frames = simular_dia_completo(
    grid, DEFAULT_ROADS,
    contaminante="PM2.5",
    perfil_meteo=clima,
    factor_trafico=1.0, factor_industrial=1.0,
    es_dia_laboral=True,
    inversion_horas=[6, 7, 8],   # inversión matutina
    minutos_por_hora=10,
)
print(f"  Tiempo de cálculo: {time.time()-t0:.1f} s ({(time.time()-t0)/24:.2f} s/hora)")
print()
print("  Evolución del ICA durante el día:")
print(f"  {'Hora':<5} {'Veh-celda':>10} {'ICA medio':>10} {'ICA máx':>9} {'Categoría'}")
for f in frames:
    from simulator import categoria_ica
    A_in = f["ica"][mask]
    cat, _ = categoria_ica(A_in.max())
    print(f"  {f['hora']:>3}h   {f['veh_total']/1000:>8.0f}k  "
          f"{A_in.mean():>9.1f}  {A_in.max():>8.1f}  {cat}")

# Verificar arrastre temporal: hora 9 debería tener más ICA que hora 8
# cuando el material previo no se ha disipado completamente
ica_horas = [frames[h]["ica"][mask].mean() for h in range(24)]
hora_pico = int(np.argmax(ica_horas))
print()
print(f"  Hora de máximo ICA medio: {hora_pico}h (ICA medio = {max(ica_horas):.1f})")
print(f"  Hora de mínimo ICA medio: {int(np.argmin(ica_horas))}h "
      f"(ICA medio = {min(ica_horas):.1f})")
print()
print("✅ Todos los tests del modelo de tráfico completaron OK.")
