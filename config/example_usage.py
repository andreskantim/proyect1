"""
Ejemplo de uso del sistema de configuración de rutas.

Este script demuestra cómo usar las rutas configuradas en tus propios scripts.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importar rutas desde la configuración (importación relativa)
from path import (
    BITCOIN_PARQUET,
    BITCOIN_CSV,
    get_plot_path,
    get_output_path,
    ensure_directories,
    print_paths
)


def example_load_data():
    """Ejemplo: Cargar datos usando rutas configuradas"""
    print("\n=== Ejemplo 1: Cargar datos ===")

    # Cargar datos desde el archivo Parquet
    if BITCOIN_PARQUET.exists():
        print(f"Cargando datos desde: {BITCOIN_PARQUET}")
        df = pd.read_parquet(BITCOIN_PARQUET)
        print(f"Datos cargados: {len(df)} filas")
        return df
    else:
        print(f"Archivo no encontrado: {BITCOIN_PARQUET}")
        return None


def example_save_plot(df):
    """Ejemplo: Guardar un gráfico usando rutas configuradas"""
    print("\n=== Ejemplo 2: Guardar gráfico ===")

    if df is None:
        print("No hay datos para graficar")
        return

    # Crear un gráfico simple
    plt.figure(figsize=(12, 6))
    plt.plot(df.index[:100], df['close'][:100])
    plt.title('Bitcoin Price (First 100 hours)')
    plt.xlabel('Time')
    plt.ylabel('Price (USD)')
    plt.grid(True)

    # Guardar usando get_plot_path
    plot_file = get_plot_path('example_bitcoin_plot.png')
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"Gráfico guardado en: {plot_file}")
    plt.close()


def example_save_csv(df):
    """Ejemplo: Guardar resultados usando rutas configuradas"""
    print("\n=== Ejemplo 3: Guardar resultados ===")

    if df is None:
        print("No hay datos para guardar")
        return

    # Calcular algunas estadísticas
    stats = df.describe()

    # Guardar usando get_output_path
    output_file = get_output_path('bitcoin_statistics.csv')
    stats.to_csv(output_file)
    print(f"Estadísticas guardadas en: {output_file}")


def main():
    """Función principal de ejemplo"""
    print("="*70)
    print("EJEMPLO DE USO DEL SISTEMA DE RUTAS")
    print("="*70)

    # 1. Mostrar configuración de rutas
    print_paths()

    # 2. Asegurar que existen los directorios necesarios
    print("\nCreando directorios necesarios...")
    ensure_directories()
    print("Directorios verificados")

    # 3. Cargar datos
    df = example_load_data()

    # 4. Guardar gráfico
    if df is not None:
        example_save_plot(df)

    # 5. Guardar resultados
    if df is not None:
        example_save_csv(df)

    print("\n" + "="*70)
    print("EJEMPLO COMPLETADO")
    print("="*70)
    print("\nVentajas de usar este sistema:")
    print("  - Las rutas son relativas al proyecto")
    print("  - Puedes mover el proyecto a cualquier ubicación")
    print("  - No necesitas modificar rutas hardcoded")
    print("  - Todos los archivos se organizan automáticamente")
    print("="*70)


if __name__ == "__main__":
    main()
