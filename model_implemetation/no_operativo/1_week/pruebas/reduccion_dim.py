#!/usr/bin/env python3
"""
Análisis de Complejidad por Semana usando PCA

Mide cuántas componentes PCA necesita cada semana
para representar ventanas deslizantes de 48h
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import matplotlib.pyplot as plt
import sys
import webbrowser
import time

sys.path.append(str(Path(__file__).parent.parent / 'src'))
from dask_utils import get_dask_client, close_dask_client
import dask.bag as db
from dask.distributed import progress

def analyze_week_complexity(week_data, window_size=48, variance_ratio=0.95):
    """
    Analiza una semana creando múltiples sub-samples
    
    week_data: 168 horas de datos OHLCV
    window_size: tamaño de ventana deslizante (horas)
    
    Returns: número de componentes PCA necesarias
    """
    n_features = 5  # OHLCV (sin duplicar volume)
    samples = []
    
    # Crear ventanas deslizantes DENTRO de la semana
    for i in range(len(week_data) - window_size + 1):
        window = week_data.iloc[i:i+window_size]
        features = window[['open', 'high', 'low', 'close', 'volume']].values.flatten()
        samples.append(features)
    
    if len(samples) < 10:  # Mínimo samples para PCA
        return None, None
    
    X = np.array(samples)
    
    # Normalizar
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # PCA
    try:
        pca = PCA(n_components=variance_ratio)
        pca.fit(X_scaled)
        
        n_components = pca.n_components_
        variance_explained = pca.explained_variance_ratio_.sum()
        
        return n_components, variance_explained
    except:
        return None, None


def main():
    # Iniciar Dask con dashboard
    print("Iniciando Dask cluster...")
    client = get_dask_client(n_workers=None, threads_per_worker=2)

    # Abrir dashboard en navegador
    dashboard_url = client.dashboard_link
    print(f"\n{'='*70}")
    print(f"Dashboard Dask: {dashboard_url}")
    print(f"{'='*70}")
    print("\nAbriendo dashboard en navegador...")

    time.sleep(1)  # Esperar a que el dashboard esté listo
    webbrowser.open(dashboard_url)

    print("\n✓ Dashboard abierto")
    print("✓ Puedes monitorear la ejecución en tiempo real")
    print(f"\nPresiona Ctrl+C para detener\n")

    # Cargar datos
    data_path = '../data/raw/bitcoin_hourly.csv'
    print(f"Cargando datos desde: {data_path}")
    df = pd.read_csv(data_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    print(f"Total horas disponibles: {len(df)}")
    
    # Parámetros
    week_size = 168  # 7 días
    window_size = 48  # Ventana deslizante de 48h para crear samples
    variance_ratio = 0.95
    
    n_features_per_window = 5 * window_size  # 240 features
    
    print(f"\nParámetros:")
    print(f"  Semana: {week_size} horas")
    print(f"  Ventana para samples: {window_size} horas")
    print(f"  Features por sample: {n_features_per_window}")
    print(f"  Varianza PCA: {variance_ratio}")
    
    # Preparar tareas para Dask
    print(f"\nPreparando {(len(df) - week_size) // week_size} semanas para análisis...")

    def analyze_week_task(args):
        week_idx, week_start_idx, df_subset, window_size, variance_ratio, n_features_per_window = args
        week_data = df_subset.iloc[week_start_idx:week_start_idx + week_size]

        n_comp, var_exp = analyze_week_complexity(
            week_data,
            window_size=window_size,
            variance_ratio=variance_ratio
        )

        if n_comp is not None:
            week_start = week_data.iloc[0]['timestamp']
            return {
                'week_idx': week_idx,
                'week_start': week_start,
                'n_components': n_comp,
                'variance_explained': var_exp,
                'original_features': n_features_per_window,
                'reduction_pct': (1 - n_comp/n_features_per_window) * 100
            }
        return None

    # Crear lista de tareas
    tasks = []
    week_counter = 0
    for week_start_idx in range(0, len(df) - week_size, week_size):
        tasks.append((
            week_counter,
            week_start_idx,
            df,
            window_size,
            variance_ratio,
            n_features_per_window
        ))
        week_counter += 1

    print(f"\n🚀 Ejecutando {len(tasks)} análisis en paralelo con Dask...")
    print(f"👀 Observa el dashboard en tu navegador para ver el progreso\n")

    # Ejecutar con Dask
    bag = db.from_sequence(tasks, partition_size=5)
    futures = bag.map(analyze_week_task).persist()

    # Mostrar progreso
    progress(futures)

    # Obtener resultados
    results = [r for r in futures.compute() if r is not None]
    
    if len(results) == 0:
        print("ERROR: No se pudieron analizar semanas")
        return
    
    df_results = pd.DataFrame(results)
    
    # Estadísticas
    print(f"\n{'='*70}")
    print(f"RESULTADOS - Componentes PCA por Semana")
    print(f"{'='*70}")
    print(f"Semanas analizadas:     {len(df_results)}")
    print(f"Features originales:    {n_features_per_window}")
    print(f"\nComponentes necesarias:")
    print(f"  Máximo:    {df_results['n_components'].max()}")
    print(f"  Media:     {df_results['n_components'].mean():.2f}")
    print(f"  Mediana:   {df_results['n_components'].median():.0f}")
    print(f"  Mínimo:    {df_results['n_components'].min()}")
    print(f"  Std:       {df_results['n_components'].std():.2f}")
    print(f"\nReducción dimensional:")
    print(f"  Media:     {df_results['reduction_pct'].mean():.1f}%")
    print(f"  Mediana:   {df_results['reduction_pct'].median():.1f}%")
    print(f"{'='*70}\n")
    
    # Análisis de estabilidad
    coef_variation = df_results['n_components'].std() / df_results['n_components'].mean()
    
    print(f"ANÁLISIS DE ESTABILIDAD:")
    print(f"  Coeficiente de variación: {coef_variation:.3f}")
    
    if coef_variation < 0.1:
        print(f"  ✅ MUY ESTABLE - Datos consistentes entre semanas")
        months_needed = "6-12 meses"
    elif coef_variation < 0.2:
        print(f"  ✅ ESTABLE - Variabilidad moderada")
        months_needed = "12-18 meses"
    elif coef_variation < 0.3:
        print(f"  ⚠️ INESTABLE - Alta variabilidad")
        months_needed = "18-24 meses"
    else:
        print(f"  ❌ MUY INESTABLE - Datos muy variables")
        months_needed = "24-36 meses"
    
    print(f"\n  Datos recomendados: {months_needed}")
    
    # Visualizaciones
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Componentes por semana
    axes[0, 0].plot(df_results['week_idx'], df_results['n_components'], marker='o', markersize=3)
    axes[0, 0].axhline(df_results['n_components'].mean(), color='r', linestyle='--', label='Media')
    axes[0, 0].set_xlabel('Semana')
    axes[0, 0].set_ylabel('Componentes PCA')
    axes[0, 0].set_title('Componentes Necesarias por Semana')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Histograma
    axes[0, 1].hist(df_results['n_components'], bins=30, edgecolor='black', alpha=0.7)
    axes[0, 1].axvline(df_results['n_components'].mean(), color='r', linestyle='--', label='Media')
    axes[0, 1].axvline(df_results['n_components'].median(), color='g', linestyle='--', label='Mediana')
    axes[0, 1].set_xlabel('Número de Componentes')
    axes[0, 1].set_ylabel('Frecuencia')
    axes[0, 1].set_title('Distribución de Componentes')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Reducción porcentual
    axes[1, 0].plot(df_results['week_idx'], df_results['reduction_pct'], marker='o', markersize=3, color='green')
    axes[1, 0].axhline(df_results['reduction_pct'].mean(), color='r', linestyle='--', label='Media')
    axes[1, 0].set_xlabel('Semana')
    axes[1, 0].set_ylabel('Reducción (%)')
    axes[1, 0].set_title('Reducción Dimensional por Semana')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Evolución temporal con fecha
    if len(df_results) > 1:
        axes[1, 1].plot(df_results['week_start'], df_results['n_components'], marker='o', markersize=3)
        axes[1, 1].axhline(df_results['n_components'].mean(), color='r', linestyle='--', label='Media')
        axes[1, 1].set_xlabel('Fecha')
        axes[1, 1].set_ylabel('Componentes PCA')
        axes[1, 1].set_title('Evolución Temporal')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('pca_weekly_analysis.png', dpi=150, bbox_inches='tight')
    print(f"\nGráfico guardado: pca_weekly_analysis.png")
    
    # Guardar CSV
    output_path = 'pca_weekly_results.csv'
    df_results.to_csv(output_path, index=False)
    print(f"Resultados guardados: {output_path}")
    
    # Resumen
    summary_path = 'pca_weekly_summary.txt'
    with open(summary_path, 'w') as f:
        f.write(f"PCA Weekly Complexity Analysis\n")
        f.write(f"{'='*70}\n\n")
        f.write(f"Configuración:\n")
        f.write(f"  Ventana por sample: {window_size} horas\n")
        f.write(f"  Features por sample: {n_features_per_window}\n")
        f.write(f"  Varianza PCA: {variance_ratio}\n\n")
        f.write(f"Resultados:\n")
        f.write(f"  Semanas analizadas: {len(df_results)}\n")
        f.write(f"  Componentes (media): {df_results['n_components'].mean():.2f}\n")
        f.write(f"  Componentes (mediana): {df_results['n_components'].median():.0f}\n")
        f.write(f"  Componentes (std): {df_results['n_components'].std():.2f}\n")
        f.write(f"  Reducción media: {df_results['reduction_pct'].mean():.1f}%\n\n")
        f.write(f"Estabilidad:\n")
        f.write(f"  Coef. variación: {coef_variation:.3f}\n")
        f.write(f"  Datos recomendados: {months_needed}\n")

    print(f"Resumen guardado: {summary_path}\n")

    # Cerrar Dask
    print("Cerrando cluster Dask...")
    close_dask_client(client)
    print("✓ Cluster cerrado")

if __name__ == "__main__":
    main()