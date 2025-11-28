"""
Versión SIMPLIFICADA del análisis MCPT sin Dask.
Usa multiprocessing básico de Python para mayor compatibilidad.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path
import time
from multiprocessing import Pool, cpu_count

# Agregar directorio padre al path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Importar configuración de rutas
from config.path import (
    BITCOIN_CSV, BITCOIN_PARQUET,
    get_plot_path, ensure_directories
)

from donchian import optimize_donchian
from bar_permute import get_permutation


# Función para procesar una permutación (debe estar en nivel módulo para pickle)
def process_permutation_simple(args):
    """Procesa una permutación individual"""
    perm_i, train_df_dict, best_real_pf = args

    # Reconstruir DataFrame
    train_df = pd.DataFrame(
        train_df_dict['data'],
        index=pd.DatetimeIndex(train_df_dict['index']),
        columns=train_df_dict['columns']
    )

    # Ejecutar permutación
    train_perm = get_permutation(train_df, seed=perm_i)
    best_lookback, best_perm_pf = optimize_donchian(train_perm)

    # Calcular cumulative returns para esta permutación
    from donchian import donchian_breakout
    signal = donchian_breakout(train_perm, best_lookback)
    r = np.log(train_perm['close']).diff().shift(-1)
    perm_rets = signal * r
    cum_rets = perm_rets.cumsum().values  # Convertir a numpy array

    is_better = 1 if best_perm_pf >= best_real_pf else 0
    return best_perm_pf, is_better, cum_rets


if __name__ == '__main__':
    print("\n" + "="*70)
    print("MCPT - VERSIÓN SIMPLIFICADA (sin Dask)")
    print("="*70 + "\n")

    # Asegurar directorios
    ensure_directories()

    # Configuración de workers - usar TODOS los cores
    total_cpus = cpu_count()
    n_workers = int(os.getenv('N_WORKERS', total_cpus))

    print(f"Configuración:")
    print(f"  CPUs disponibles: {total_cpus}")
    print(f"  Workers a usar:   {n_workers}")
    print(f"  (Cambiar con: N_WORKERS=4 python {Path(__file__).name})")
    print("="*70 + "\n")
    sys.stdout.flush()

    # Cargar datos
    print("Cargando datos...")
    if BITCOIN_PARQUET.exists():
        df = pd.read_parquet(BITCOIN_PARQUET)
        df.index = df.index.astype('datetime64[s]')
    else:
        print(f"Convirtiendo CSV a Parquet...")
        df = pd.read_csv(BITCOIN_CSV, parse_dates=["timestamp"])
        df = df.set_index("timestamp")
        df.to_parquet(BITCOIN_PARQUET)

    print(f"✓ Datos cargados: {len(df)} filas\n")

    # Análisis in-sample
    print("="*70)
    print("OPTIMIZACIÓN IN-SAMPLE")
    print("="*70)

    train_df = df[(df.index.year >= 2016) & (df.index.year < 2020)]
    best_lookback, best_real_pf = optimize_donchian(train_df)

    # Calcular cumulative returns de la estrategia real
    from donchian import donchian_breakout
    real_signal = donchian_breakout(train_df, best_lookback)
    real_r = np.log(train_df['close']).diff().shift(-1)
    real_rets = real_signal * real_r
    real_cum_rets = real_rets.cumsum()

    print(f"  Best Lookback:     {best_lookback}")
    print(f"  Best Profit Factor: {best_real_pf:.4f}")
    print("="*70 + "\n")
    sys.stdout.flush()

    # MCPT
    n_permutations = 1000
    print(f"Ejecutando MCPT con {n_permutations} permutaciones usando {n_workers} workers...")
    print()
    sys.stdout.flush()

    # Preparar datos
    train_df_dict = {
        'data': train_df.values,
        'index': train_df.index.values,
        'columns': train_df.columns.tolist()
    }

    # Crear argumentos
    args_list = [(i, train_df_dict, best_real_pf) for i in range(n_permutations)]

    # Procesar con multiprocessing
    start_time = time.time()
    last_update = start_time
    update_interval = 1

    print("="*70)
    print("PROGRESO")
    print("="*70)
    print(f"  Inicio: {time.strftime('%H:%M:%S')}")
    print("="*70 + "\n")
    sys.stdout.flush()

    results = []
    with Pool(processes=n_workers) as pool:
        for i, result in enumerate(pool.imap_unordered(process_permutation_simple, args_list), 1):
            results.append(result)

            # Actualizar progreso
            current_time = time.time()
            if current_time - last_update >= update_interval or i == len(args_list):
                elapsed = current_time - start_time
                completed = i
                total = len(args_list)
                percentage = (completed / total) * 100
                speed = completed / elapsed if elapsed > 0 else 0
                eta = (total - completed) / speed if speed > 0 else 0

                # Barra visual
                bar_width = 40
                filled = int(bar_width * completed / total)
                bar = '█' * filled + '░' * (bar_width - filled)

                print(f"\r[{bar}] {completed}/{total} ({percentage:5.1f}%) | "
                      f"{speed:.1f} tareas/s | "
                      f"Tiempo: {elapsed:.0f}s | "
                      f"ETA: {eta:.0f}s", end='')
                sys.stdout.flush()
                last_update = current_time

    print("\n")
    total_time = time.time() - start_time

    # Análisis de resultados
    permuted_pfs = [pf for pf, _, _ in results]
    perm_better_count = 1 + sum(is_better for _, is_better, _ in results)
    perm_cum_rets = [cum_rets for _, _, cum_rets in results]
    insample_mcpt_pval = perm_better_count / n_permutations

    print("="*70)
    print("RESULTADOS MCPT")
    print("="*70)
    print(f"  Permutaciones:     {len(results)}")
    print(f"  Mejores que real:  {perm_better_count}")
    print(f"  P-Value:           {insample_mcpt_pval:.4f}")
    print(f"  Tiempo total:      {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  Velocidad:         {len(results)/total_time:.1f} tareas/s")

    if insample_mcpt_pval < 0.05:
        print(f"  ✅ Significativo (p < 0.05)")
    else:
        print(f"  ⚠️  NO significativo (p >= 0.05)")

    print("="*70 + "\n")
    sys.stdout.flush()

    # Generar gráfico
    print("Generando gráfico...")
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 7))

    ax.hist(permuted_pfs, bins=50, color='steelblue',
            alpha=0.7, edgecolor='white', label='Permutations')

    ax.axvline(best_real_pf, color='red', linestyle='--',
               linewidth=2.5, label=f'Real PF: {best_real_pf:.4f}')

    mean_perm = np.mean(permuted_pfs)
    ax.axvline(mean_perm, color='yellow', linestyle=':',
               linewidth=2, alpha=0.7, label=f'Mean Perm: {mean_perm:.4f}')

    ax.set_xlabel("Profit Factor", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title(f"In-sample MCPT | P-Value: {insample_mcpt_pval:.4f} | "
                 f"{'Significant' if insample_mcpt_pval < 0.05 else 'Not Significant'}",
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc='upper right')

    plt.tight_layout()

    output_filename = f'insample_mcpt_pval_{insample_mcpt_pval:.4f}.png'
    output_file = get_plot_path(output_filename)
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    print(f"✓ Gráfico guardado: {output_file}\n")
    plt.close()

    # Generar gráfico de cumulative returns
    print("Generando gráfico de cumulative returns...")
    fig, ax = plt.subplots(figsize=(14, 8))

    # Graficar todas las permutaciones en blanco con transparencia
    for perm_cum in perm_cum_rets:
        ax.plot(train_df.index, perm_cum, color='white', alpha=0.02, linewidth=0.5)

    # Graficar la estrategia real en rojo
    ax.plot(train_df.index, real_cum_rets, color='red', linewidth=2.5,
            label=f'Real Strategy (PF={best_real_pf:.4f})', zorder=100)

    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel("Cumulative Log Return", fontsize=12)
    ax.set_title(f"In-sample Cumulative Returns | Real vs {len(perm_cum_rets)} Permutations",
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc='upper left')

    plt.tight_layout()

    output_filename_cum = f'insample_cumulative_returns_pval_{insample_mcpt_pval:.4f}.png'
    output_file_cum = get_plot_path(output_filename_cum)
    plt.savefig(output_file_cum, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    print(f"✓ Gráfico cumulative returns guardado: {output_file_cum}\n")
    plt.close()

    print("="*70)
    print("✓ ANÁLISIS COMPLETADO")
    print("="*70)
