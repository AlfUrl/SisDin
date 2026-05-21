"""
Test de integración: landuse.py + simulator.py (run_dispersion con landuse_map).
Ejecutar con: python SisDin/test_landuse_integration.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from simulator import build_grid, run_dispersion
from landuse import build_landuse_map, resumen_landuse, DEFAULT_LANDUSE

def main():
    grid = build_grid()

    # Rasterizar zonas hardcodeadas (sin llamada OSM)
    lm = build_landuse_map(grid, zonas=DEFAULT_LANDUSE)
    print(f"landuse_map shape: {lm.shape}")

    resumen = resumen_landuse(lm)
    print("Uso de suelo:")
    for nombre, data in resumen.items():
        print(f"  {nombre}: {data['celdas']} celdas ({data['pct']}%)")

    # Fuente ficticia en el centro de la rejilla
    E = np.zeros((grid["filas"], grid["columnas"]), dtype=np.float32)
    cx, cy = grid["filas"] // 2, grid["columnas"] // 2
    E[cx:cx+5, cy:cy+5] = 0.1

    # Con landuse (fricción diferencial)
    C = run_dispersion(
        grid, E,
        wind_speed_ms=3.0, wind_direction_deg=270,
        temperatura_c=25, presion_hpa=1013,
        tiempo_simulado_s=60,
        landuse_map=lm,
    )
    print(f"\nCon landuse_map  -> C.max={C.max():.4f}  C.mean={C.mean():.6f}")

    # Sin landuse (modo escalar original)
    C2 = run_dispersion(
        grid, E,
        wind_speed_ms=3.0, wind_direction_deg=270,
        temperatura_c=25, presion_hpa=1013,
        tiempo_simulado_s=60,
    )
    print(f"Sin landuse_map  -> C.max={C2.max():.4f}  C.mean={C2.mean():.6f}")

    assert C.max() > 0, "Error: concentración cero con landuse_map"
    assert C2.max() > 0, "Error: concentración cero sin landuse_map"
    print("\nTest PASADO OK")

if __name__ == "__main__":
    main()
