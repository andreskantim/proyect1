# Pruebas - Análisis de Datos

## reduccion_dim.py

Analiza cuántos componentes PCA (variance=0.95) se necesitan por semana.

**Uso:**
```bash
python reduccion_dim.py --data ../data/raw/bitcoin_hourly.csv --output results/pca_results.csv
```

**Output:**
- `pca_results.csv`: Componentes por semana
- `pca_results_summary.txt`: Estadísticas (max, medio, mediana, min)
