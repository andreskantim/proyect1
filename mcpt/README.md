# MCPT - Monte Carlo Permutation Test

Análisis estadístico de estrategias de trading usando Monte Carlo Permutation Test con paralelización mediante `multiprocessing`.

## Inicio Rápido

```bash
cd /home/ulc/cursos/curso396/proyect1/mcpt

# Análisis In-sample (1000 permutaciones, ~1 min con 64 cores)
python insample_donchian_simple.py

# Análisis Walk-forward (200 permutaciones, ~15 seg con 64 cores)
python walkforward_donchian_simple.py

# Especificar workers manualmente
N_WORKERS=32 python insample_donchian_simple.py
N_WORKERS=32 python walkforward_donchian_simple.py
```

## Estructura

```
mcpt/
├── data/                              # Datos de mercado
│   ├── bitcoin_hourly.csv
│   └── BTCUSD3600.pq
├── output/                            # Resultados
│   ├── plots/                        # Gráficos generados
│   └── logs/                         # Logs
├── insample_donchian_simple.py       # MCPT in-sample
├── walkforward_donchian_simple.py    # MCPT walk-forward
├── donchian.py                       # Estrategia Donchian
├── bar_permute.py                    # Permutación de barras
├── tree_strat.py                     # Estrategia con árboles
└── moving_average.py                 # Medias móviles
```

## Scripts Principales

### `insample_donchian_simple.py`

Análisis in-sample con MCPT usando `multiprocessing.Pool`.

**Características:**
- ✅ Simple y robusto
- ✅ Usa todos los cores del nodo
- ✅ Sin dependencias complejas (Dask)
- ✅ Progreso cada 5 segundos
- ✅ 1000 permutaciones por defecto

**Ejecución:**
```bash
python insample_donchian_simple.py
```

### `walkforward_donchian_simple.py`

Análisis walk-forward con MCPT usando `multiprocessing.Pool`.

**Características:**
- ✅ Walk-forward testing (entrenamiento continuo)
- ✅ Usa todos los cores del nodo
- ✅ Sin dependencias complejas (Dask)
- ✅ Progreso cada 5 segundos
- ✅ 200 permutaciones por defecto
- ✅ Período 2016-2020

**Ejecución:**
```bash
python walkforward_donchian_simple.py
```

**Salida esperada:**
```
======================================================================
MCPT - VERSIÓN SIMPLIFICADA (sin Dask)
======================================================================

Configuración:
  CPUs disponibles: 64
  Workers a usar:   64

Cargando datos...
✓ Datos cargados: 43824 filas

======================================================================
OPTIMIZACIÓN IN-SAMPLE
======================================================================
  Best Lookback:     49
  Best Profit Factor: 1.0661
======================================================================

Ejecutando MCPT con 1000 permutaciones usando 64 workers...

======================================================================
PROGRESO
======================================================================
  Inicio: 20:00:00
======================================================================

[████████████████████████████████████████] 999/999 (100.0%) | 15.2 tareas/s | Tiempo: 66s | ETA: 0s

======================================================================
RESULTADOS MCPT
======================================================================
  Permutaciones:     999
  Mejores que real:  45
  P-Value:           0.0450
  Tiempo total:      65.7s (1.1 min)
  Velocidad:         15.2 tareas/s
  ✅ Significativo (p < 0.05)
======================================================================

Generando gráfico...
✓ Gráfico guardado: .../output/plots/insample_mcpt_pval_0.0450.png

Generando gráfico de cumulative returns...
✓ Gráfico cumulative returns guardado: .../output/plots/insample_cumulative_returns_pval_0.0450.png

======================================================================
✓ ANÁLISIS COMPLETADO
======================================================================
```

## Configuración

### Número de Workers

Por defecto usa **todos los cores**:
```python
n_workers = cpu_count()  # 64 en tu sistema
```

Cambiar manualmente:
```bash
# Usar 32 cores
N_WORKERS=32 python insample_donchian_simple.py

# Usar 16 cores
N_WORKERS=16 python insample_donchian_simple.py
```

### Número de Permutaciones

**In-sample** - Editar línea 96 en `insample_donchian_simple.py`:
```python
n_permutations = 1000  # Cambiar a 100 para test rápido
```

**Walk-forward** - Editar línea 101 en `walkforward_donchian_simple.py`:
```python
n_permutations = 200  # Cambiar a 50 para test rápido
```

## Tiempos Estimados

### In-sample (1000 permutaciones)

| Workers | Tiempo Estimado |
|---------|-----------------|
| 16      | ~4 minutos      |
| 32      | ~2 minutos      |
| 64      | ~1 minuto       |

### Walk-forward (200 permutaciones)

| Workers | Tiempo Estimado |
|---------|-----------------|
| 16      | ~1 minuto       |
| 32      | ~30 segundos    |
| 64      | ~15 segundos    |

## Resultados

Cada script genera:

1. **Gráfico de Histograma** en `output/plots/`:
   - Histograma de profit factors de permutaciones
   - Línea del resultado real
   - P-value
   - Significancia estadística

2. **Gráfico de Cumulative Returns** en `output/plots/`:
   - Curvas de cumulative log returns de todas las permutaciones (blanco, transparente)
   - Curva de la estrategia optimizada real (rojo, destacado)
   - Comparación visual del rendimiento temporal

3. **Estadísticas en terminal**:
   - P-value
   - Número de permutaciones mejores
   - Tiempo total
   - Velocidad de procesamiento

## Por Qué Esta Versión

**Dask NO funcionó porque:**
- ❌ Diseñado para clusters distribuidos (múltiples máquinas)
- ❌ Overhead innecesario para 1 nodo
- ❌ Problemas de serialización con objetos complejos
- ❌ Errores en HPC con límites de procesos

**multiprocessing.Pool funciona porque:**
- ✅ Diseñado para 1 máquina, múltiples cores
- ✅ Simple y directo
- ✅ Usa fork() (copy-on-write, no serialización)
- ✅ Sin overhead de scheduler distribuido
- ✅ Compatible con cualquier entorno

## Requisitos

```bash
pip install pandas numpy matplotlib
```

## Otros Scripts

### `insample_tree_mcpt.py`
Análisis con decision trees (requiere actualización para usar multiprocessing simple)

### Scripts antiguos (legacy)
- `walkforward_donchian_mcpt.py` - Versión antigua secuencial (usar `walkforward_donchian_simple.py`)
- `insample_donchian_mcpt.py` - Versión antigua con Dask (DEPRECATED)

## Solución de Problemas

### Script muy lento
```bash
# Reducir permutaciones para test
# Editar línea 95: n_permutations = 100
python insample_donchian_simple.py
```

### Error de memoria
```bash
# Usar menos workers
N_WORKERS=16 python insample_donchian_simple.py
```

### Verificar cores disponibles
```bash
python -c "from multiprocessing import cpu_count; print(f'Cores: {cpu_count()}')"
```

## Documentación del Proyecto

- **[../config/README.md](../config/README.md)** - Sistema de rutas del proyecto
- **[../README.md](../README.md)** - Documentación general

## Conclusión

Esta versión usa **multiprocessing simple** de Python porque:
- Es la herramienta correcta para el caso (1 máquina, N cores)
- Funciona sin problemas en HPC
- No tiene overhead innecesario
- Es fácil de entender y modificar

**Para análisis distribuido real (múltiples nodos), Dask sería apropiado, pero no para este caso.**
