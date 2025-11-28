"""
Módulo de configuración del proyecto MCPT.

Importa las rutas principales para facilitar su uso en otros scripts.
"""

from config.path import (
    PROJECT_ROOT,
    CONFIG_DIR,
    MCPT_DIR,
    SCRIPTS_DIR,
    DATA_DIR,
    OUTPUT_DIR,
    PLOTS_DIR,
    LOGS_DIR,
    BITCOIN_CSV,
    BITCOIN_PARQUET,
    DASK_UTILS_DIR,
    ensure_directories,
    get_plot_path,
    get_output_path,
    print_paths
)

__all__ = [
    'PROJECT_ROOT',
    'CONFIG_DIR',
    'MCPT_DIR',
    'SCRIPTS_DIR',
    'DATA_DIR',
    'OUTPUT_DIR',
    'PLOTS_DIR',
    'LOGS_DIR',
    'BITCOIN_CSV',
    'BITCOIN_PARQUET',
    'DASK_UTILS_DIR',
    'ensure_directories',
    'get_plot_path',
    'get_output_path',
    'print_paths'
]
