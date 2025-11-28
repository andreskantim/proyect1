# Proyecto 1 - Análisis Cuantitativo y Machine Learning

Este repositorio contiene múltiples subproyectos relacionados con análisis cuantitativo, trading algorítmico y machine learning aplicado a mercados financieros.

## Estructura del Proyecto

```
proyect1/
├── config/              # Configuración centralizada de rutas (NUEVO)
│   ├── path.py         # Definición de rutas del proyecto
│   ├── __init__.py     # Módulo Python
│   ├── README.md       # Documentación de configuración
│   └── example_usage.py # Ejemplos de uso
│
├── mcpt/               # Monte Carlo Permutation Test
│   ├── data/          # Datos de mercado
│   ├── output/        # Resultados y gráficos
│   └── *.py           # Scripts de análisis
│
├── dask/              # Utilidades de computación distribuida
│   └── dask_utils.py
│
├── LLM-T/             # Predicción con transformers
│   └── stock_predictor/
│
└── model_implemetation/ # Implementaciones de modelos
    ├── operativo/
    └── no_operativo/
```

## Sistema de Rutas Centralizado

Este proyecto utiliza un sistema moderno de paths relativos que permite:
- **Portabilidad total**: Mover el proyecto a cualquier máquina sin cambios
- **Organización clara**: Separación entre datos, código y resultados
- **Mantenibilidad**: Todas las rutas en un solo lugar (`config/path.py`)

### Uso del Sistema de Rutas

En cualquier script dentro de los subproyectos:

```python
import sys
from pathlib import Path

# Agregar proyect1 al path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Importar rutas configuradas
from config.path import BITCOIN_PARQUET, get_plot_path, ensure_directories

# Usar rutas
ensure_directories()
df = pd.read_parquet(BITCOIN_PARQUET)
plot_file = get_plot_path('mi_grafico.png')
plt.savefig(plot_file)
```

Para más detalles, consulta [config/README.md](config/README.md)

## Subproyectos

### MCPT (Monte Carlo Permutation Test)
Implementación de pruebas de permutación Monte Carlo para validación estadística de estrategias de trading.

- **Directorio**: `mcpt/`
- **Datos**: `mcpt/data/`
- **Salidas**: `mcpt/output/plots/` y `mcpt/output/logs/`

Scripts principales:
- `insample_donchian_mcpt.py` - Análisis in-sample con estrategia Donchian
- `walkforward_donchian_mcpt.py` - Análisis walk-forward
- `insample_tree_mcpt.py` - Análisis con decision trees

### LLM-T (Language Model Transformer)
Predicción de mercados financieros usando transformers.

- **Directorio**: `LLM-T/stock_predictor/`

### Dask Utilities
Utilidades para computación paralela y distribuida.

- **Directorio**: `dask/`
- **Uso**: Procesamiento de grandes volúmenes de datos

## Ventajas del Sistema de Rutas

1. **Portabilidad**: Copia `proyect1/` a cualquier ubicación y todo funciona
2. **Sin rutas hardcoded**: Todas las rutas se calculan relativamente
3. **Multiplataforma**: Funciona en Windows, Linux y macOS
4. **Organización automática**: Los directorios se crean automáticamente
5. **Fácil mantenimiento**: Cambios centralizados en `config/path.py`

## Migración a otra Máquina

Para mover el proyecto completo a otra máquina:

1. Copia el directorio completo `proyect1/`
2. Ejecuta cualquier script
3. ¡Eso es todo! No necesitas modificar ninguna ruta

El sistema detectará automáticamente la nueva ubicación y ajustará todas las rutas.

## Desarrollo

Cuando agregues nuevos scripts que necesiten acceder a datos o guardar resultados:

1. Importa las rutas desde `config.path`
2. Usa las funciones auxiliares (`ensure_directories()`, `get_plot_path()`, etc.)
3. Nunca uses rutas absolutas o hardcoded

## Documentación Adicional

- [Configuración de Rutas](config/README.md) - Detalles del sistema de paths
- [Ejemplo de Uso](config/example_usage.py) - Script de ejemplo completo
