# Configuración de Rutas del Proyecto

Este directorio contiene la configuración centralizada de rutas del proyecto MCPT.

## Estructura

```
config/
├── __init__.py       # Importaciones públicas del módulo
├── path.py           # Definición de todas las rutas del proyecto
└── README.md         # Este archivo
```

## Uso

### Importación básica

```python
from config.path import BITCOIN_PARQUET, get_plot_path, ensure_directories

# Asegurar que los directorios existen
ensure_directories()

# Cargar datos
df = pd.read_parquet(BITCOIN_PARQUET)

# Guardar un gráfico
plot_file = get_plot_path('my_plot.png')
plt.savefig(plot_file)
```

### Rutas disponibles

#### Directorios base
- `PROJECT_ROOT`: Raíz del proyecto
- `CONFIG_DIR`: Directorio de configuración
- `SCRIPTS_DIR`: Directorio de scripts
- `DATA_DIR`: Directorio de datos
- `OUTPUT_DIR`: Directorio de salidas
- `PLOTS_DIR`: Directorio de gráficos
- `LOGS_DIR`: Directorio de logs

#### Archivos de datos
- `BITCOIN_CSV`: Archivo CSV de Bitcoin
- `BITCOIN_PARQUET`: Archivo Parquet de Bitcoin

#### Directorios externos
- `DASK_UTILS_DIR`: Directorio de utilidades Dask

### Funciones auxiliares

#### `ensure_directories()`
Crea todos los directorios necesarios si no existen.

```python
from config.path import ensure_directories
ensure_directories()
```

#### `get_plot_path(filename)`
Genera una ruta completa para guardar un gráfico.

```python
from config.path import get_plot_path
plot_path = get_plot_path('insample_mcpt_result.png')
plt.savefig(plot_path)
```

#### `get_output_path(filename)`
Genera una ruta completa para guardar cualquier archivo de salida.

```python
from config.path import get_output_path
output_path = get_output_path('results.csv')
df.to_csv(output_path)
```

#### `print_paths()`
Imprime todas las rutas configuradas (útil para debugging).

```python
from config.path import print_paths
print_paths()
```

## Estructura del Proyecto

```
mcpt/
├── config/              # Configuración de rutas
│   ├── __init__.py
│   ├── path.py
│   └── README.md
├── data/                # Datos del proyecto
│   ├── bitcoin_hourly.csv
│   └── BTCUSD3600.pq
├── output/              # Resultados y salidas
│   ├── plots/          # Gráficos generados
│   └── logs/           # Archivos de log
├── bar_permute.py       # Scripts de análisis
├── donchian.py
├── insample_donchian_mcpt.py
└── ...
```

## Ventajas de esta estructura

1. **Portabilidad**: El proyecto puede moverse a cualquier máquina sin modificar las rutas
2. **Centralización**: Todas las rutas están definidas en un solo lugar
3. **Organización**: Separación clara entre datos, código y resultados
4. **Mantenibilidad**: Fácil de actualizar y mantener
5. **Compatibilidad**: Funciona en Windows, Linux y macOS usando `pathlib`

## Migración a otra máquina

Para mover el proyecto a otra máquina:

1. Copia toda la carpeta del proyecto
2. Ejecuta cualquier script - la estructura de rutas se adaptará automáticamente
3. No necesitas modificar ningún path en el código

¡Eso es todo! El sistema de rutas relativas se encargará de mantener la estructura.
