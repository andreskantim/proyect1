"""
Configuración de rutas del proyecto MCPT (Monte Carlo Permutation Test)

Este módulo centraliza todas las rutas clave del proyecto usando paths relativos,
lo que permite mover el proyecto a cualquier máquina manteniendo su estructura.
"""

from pathlib import Path

# ====================================================================
# RUTAS BASE DEL PROYECTO
# ====================================================================

# Raíz del proyecto (directorio que contiene config/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Directorio de configuración
CONFIG_DIR = PROJECT_ROOT / "config"

# Directorio MCPT (subproyecto principal)
MCPT_DIR = PROJECT_ROOT / "mcpt"

# Directorio de scripts/código fuente
SCRIPTS_DIR = MCPT_DIR

# Directorio de datos
DATA_DIR = MCPT_DIR / "data"

# Directorio de salidas/resultados
OUTPUT_DIR = MCPT_DIR / "output"

# Directorio de gráficos
PLOTS_DIR = OUTPUT_DIR / "plots"

# Directorio de logs
LOGS_DIR = OUTPUT_DIR / "logs"

# ====================================================================
# RUTAS DE DATOS
# ====================================================================

# Archivos de datos principales
BITCOIN_CSV = DATA_DIR / "bitcoin_hourly.csv"
BITCOIN_PARQUET = DATA_DIR / "BTCUSD3600.pq"

# ====================================================================
# RUTAS EXTERNAS
# ====================================================================

# Directorio de utilidades Dask
DASK_UTILS_DIR = PROJECT_ROOT / "dask"

# ====================================================================
# FUNCIONES AUXILIARES
# ====================================================================

def ensure_directories():
    """
    Crea los directorios necesarios si no existen.
    Llamar esta función al inicio de los scripts principales.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)


def get_plot_path(filename):
    """
    Genera una ruta completa para guardar un gráfico.

    Args:
        filename (str): Nombre del archivo (puede incluir subdirectorios)

    Returns:
        Path: Ruta completa al archivo de gráfico

    Example:
        >>> plot_path = get_plot_path('insample_mcpt_pval_0.0450.png')
        >>> plt.savefig(plot_path)
    """
    return PLOTS_DIR / filename


def get_output_path(filename):
    """
    Genera una ruta completa para guardar cualquier archivo de salida.

    Args:
        filename (str): Nombre del archivo

    Returns:
        Path: Ruta completa al archivo de salida
    """
    return OUTPUT_DIR / filename


# ====================================================================
# INFORMACIÓN DEL PROYECTO
# ====================================================================

def print_paths():
    """
    Imprime todas las rutas configuradas (útil para debugging).
    """
    print("=" * 70)
    print("CONFIGURACIÓN DE RUTAS DEL PROYECTO")
    print("=" * 70)
    print(f"PROJECT_ROOT:    {PROJECT_ROOT}")
    print(f"CONFIG_DIR:      {CONFIG_DIR}")
    print(f"MCPT_DIR:        {MCPT_DIR}")
    print(f"SCRIPTS_DIR:     {SCRIPTS_DIR}")
    print(f"DATA_DIR:        {DATA_DIR}")
    print(f"OUTPUT_DIR:      {OUTPUT_DIR}")
    print(f"PLOTS_DIR:       {PLOTS_DIR}")
    print(f"LOGS_DIR:        {LOGS_DIR}")
    print()
    print("Archivos de datos:")
    print(f"  BITCOIN_CSV:     {BITCOIN_CSV}")
    print(f"  BITCOIN_PARQUET: {BITCOIN_PARQUET}")
    print()
    print("Directorios externos:")
    print(f"  DASK_UTILS_DIR:  {DASK_UTILS_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    # Si se ejecuta directamente, muestra las rutas configuradas
    print_paths()

    # Crear directorios si no existen
    print("\nCreando directorios necesarios...")
    ensure_directories()
    print("Directorios creados correctamente.")
