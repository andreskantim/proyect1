"""
Versión SIMPLIFICADA del análisis Walk-Forward MCPT sin Dask.
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
    BITCOIN_PARQUET,
    get_plot_path, ensure_directories
)

from donchian import walkforward_donch
from bar_permute import get_permutation


# Función para procesar una permutación (debe estar en nivel módulo para pickle)
def process_walkforward_permutation(args):
    """Procesa una permutación individual de walk-forward"""
    perm_i, df_dict, train_window, real_wf_pf = args

    # Reconstruir DataFrame
    df_perm = pd.DataFrame(
        df_dict['data'],
        index=pd.DatetimeIndex(df_dict['index']),
        columns=df_dict['columns']
    )

    # Ejecutar permutación
    wf_perm = get_permutation(df_perm, start_index=train_window)

    # Calcular returns y señales
    wf_perm['r'] = np.log(wf_perm['close']).diff().shift(-1)
    wf_perm_sig = walkforward_donch(wf_perm, train_lookback=train_window)
    perm_rets = wf_perm['r'] * wf_perm_sig

    # Calcular profit factor
    pos = perm_rets[perm_rets > 0].sum()
    neg = perm_rets[perm_rets < 0].abs().sum()
    if neg == 0:
        perm_pf = np.inf if pos > 0 else 0.0
    else:
        perm_pf = pos / neg

    # Calcular cumulative returns
    cum_rets = perm_rets.cumsum().values  # Convertir a numpy array

    is_better = 1 if perm_pf >= real_wf_pf else 0
    return perm_pf, is_better, cum_rets


if __name__ == '__main__':
    print("\n" + "="*70)
    print("WALK-FORWARD MCPT - VERSIÓN SIMPLIFICADA (sin Dask)")
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
    df = pd.read_parquet(BITCOIN_PARQUET)
    df.index = df.index.astype('datetime64[s]')
    df = df[(df.index.year >= 2016) & (df.index.year < 2021)]
    print(f"✓ Datos cargados: {len(df)} filas (2016-2020)\n")

    # Configuración walk-forward
    # Ajustar train_window para que sea máximo 60% de los datos disponibles
    max_train_window = int(len(df) * 0.6)
    desired_train_window = 24 * 365 * 4  # 4 years of hourly data

    if desired_train_window > max_train_window:
        train_window = max_train_window
        print(f"⚠️  Train window ajustado de {desired_train_window} a {train_window} (datos insuficientes)\n")
    else:
        train_window = desired_train_window

    print("="*70)
    print("ANÁLISIS WALK-FORWARD")
    print("="*70)
    print(f"  Datos totales: {len(df)} períodos ({len(df)/24/365:.1f} años)")
    print(f"  Train window: {train_window} períodos ({train_window/24/365:.1f} años)")

    # Calcular estrategia real
    df['r'] = np.log(df['close']).diff().shift(-1)
    df['donch_wf_signal'] = walkforward_donch(df, train_lookback=train_window)
    donch_rets = df['donch_wf_signal'] * df['r']
    real_wf_pf = donch_rets[donch_rets > 0].sum() / donch_rets[donch_rets < 0].abs().sum()
    real_cum_rets = donch_rets.cumsum()

    print(f"  Real Profit Factor: {real_wf_pf:.4f}")
    print("="*70 + "\n")
    sys.stdout.flush()

    # MCPT
    n_permutations = 1000
    print(f"Ejecutando Walk-Forward MCPT con {n_permutations} permutaciones usando {n_workers} workers...")
    print()
    sys.stdout.flush()

    # Preparar datos
    df_dict = {
        'data': df.values,
        'index': df.index.values,
        'columns': df.columns.tolist()
    }

    # Crear argumentos
    args_list = [(i, df_dict, train_window, real_wf_pf)
                 for i in range(n_permutations)]

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
        for i, result in enumerate(pool.imap_unordered(process_walkforward_permutation, args_list), 1):
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
    walkforward_mcpt_pval = perm_better_count / n_permutations

    print("="*70)
    print("RESULTADOS MCPT")
    print("="*70)
    print(f"  Permutaciones:     {len(results)}")
    print(f"  Mejores que real:  {perm_better_count}")
    print(f"  P-Value:           {walkforward_mcpt_pval:.4f}")
    print(f"  Tiempo total:      {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  Velocidad:         {len(results)/total_time:.1f} tareas/s")

    if walkforward_mcpt_pval < 0.05:
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

    ax.axvline(real_wf_pf, color='red', linestyle='--',
               linewidth=2.5, label=f'Real PF: {real_wf_pf:.4f}')

    mean_perm = np.mean(permuted_pfs)
    ax.axvline(mean_perm, color='yellow', linestyle=':',
               linewidth=2, alpha=0.7, label=f'Mean Perm: {mean_perm:.4f}')

    ax.set_xlabel("Profit Factor", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title(f"Walk-Forward MCPT | P-Value: {walkforward_mcpt_pval:.4f} | "
                 f"{'Significant' if walkforward_mcpt_pval < 0.05 else 'Not Significant'}",
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc='upper right')

    plt.tight_layout()

    output_filename = f'walkforward_mcpt_pval_{walkforward_mcpt_pval:.4f}.png'
    output_file = get_plot_path(output_filename)
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    print(f"✓ Gráfico guardado: {output_file}\n")
    plt.close()

    # Generar gráfico de cumulative returns
    print("Generando gráfico de cumulative returns...")
    fig, ax = plt.subplots(figsize=(14, 8))

    # Graficar todas las permutaciones en blanco con transparencia
    for perm_cum in perm_cum_rets:
        ax.plot(df.index, perm_cum, color='white', alpha=0.05, linewidth=0.5)

    # Graficar la estrategia real en rojo
    ax.plot(df.index, real_cum_rets, color='red', linewidth=2.5,
            label=f'Real Strategy (PF={real_wf_pf:.4f})', zorder=100)

    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel("Cumulative Log Return", fontsize=12)
    ax.set_title(f"Walk-Forward Cumulative Returns | Real vs {len(perm_cum_rets)} Permutations",
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc='upper left')

    plt.tight_layout()

    output_filename_cum = f'walkforward_cumulative_returns_pval_{walkforward_mcpt_pval:.4f}.png'
    output_file_cum = get_plot_path(output_filename_cum)
    plt.savefig(output_file_cum, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    print(f"✓ Gráfico cumulative returns guardado: {output_file_cum}\n")
    plt.close()

    print("="*70)
    print("✓ ANÁLISIS WALK-FORWARD COMPLETADO")
    print("="*70)
