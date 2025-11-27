# Modelo de Predicción Bitcoin - 1 Semana

## Descripción General

Sistema de predicción de precios de Bitcoin utilizando velas horarias para predecir las próximas 24 horas (1 día).

## Especificaciones del Modelo

### Datos de Entrada
- **Input**: 168 velas horarias (7 días × 24 horas)
- **Output**: 24 valores predichos (siguiente día)
- **Features por vela**: Open, High, Low, Close, Volume

### Estructura de Datos

#### Training Set (Año 1)
- Walk-forward diario durante 1 año
- Cada día usa la semana anterior (168 horas) para predecir el día siguiente (24 horas)
- ~365 iteraciones de entrenamiento

#### Validation Set (Año 2)
- Evaluación del mejor modelo del año 1
- Mismo proceso walk-forward
- **IMPORTANTE**: Gap de 1 día entre training y validation para evitar overlap

#### Confidence Set (Año 3)
- Conjunto independiente para intervalos de confianza
- No se usa en entrenamiento ni selección de modelo

### Gestión de Overlap

Según Masters (2018), página 17-18, para evitar future leak:
- **Lookback length**: 168 horas (7 días)
- **Look-ahead length**: 24 horas (1 día)
- **Gap mínimo necesario**: min(168, 24) - 1 = 23 horas

Por tanto, usamos **1 día de gap (24 horas)** entre datasets, que es suficiente (24h > 23h requerido).

Esto significa que se elimina el último día del año de entrenamiento y el primer día del año de validación.

## Modelos a Entrenar

### 1. Support Vector Regression (SVR) - Kernel Gaussiano
```python
Parámetros a probar:
- C: [0.1, 1, 10, 100]
- gamma: ['scale', 'auto', 0.001, 0.01]
- epsilon: [0.01, 0.1, 0.5]
```

### 2. Random Forest Regressor
```python
Parámetros a probar:
- n_estimators: [100, 200, 500]
- max_depth: [10, 20, None]
- min_samples_split: [2, 5, 10]
- min_samples_leaf: [1, 2, 4]
```

### 3. Gradient Boosting Regressor
```python
Parámetros a probar:
- n_estimators: [100, 200, 500]
- learning_rate: [0.01, 0.05, 0.1]
- max_depth: [3, 5, 7]
- subsample: [0.8, 1.0]
```

### 4. Multi-Layer Perceptron (MLP) Regressor
```python
Parámetros a probar:
- hidden_layer_sizes: [(100,), (100, 50), (200, 100, 50)]
- activation: ['relu', 'tanh']
- alpha: [0.0001, 0.001, 0.01]
- learning_rate_init: [0.001, 0.01]
```

## Métricas de Evaluación

Según el PDF (Capítulo 1), usaremos:

1. **MSE** (Mean Squared Error) - Métrica base
2. **RMS Error** - Interpretable en unidades de precio
3. **Success Ratio** - Específico para trading
   - Success Factor = Σ(ganancias) / |Σ(pérdidas)|
4. **Profit Factor** - Para evaluación financiera
5. **Spearman Correlation** - Orden relativo de predicciones

## Estructura de Archivos

```
1_week/
├── README.md                    # Este archivo
├── data/                        # Datos de Bitcoin (crear)
│   ├── raw/                     # Datos crudos
│   └── processed/               # Datos procesados
├── src/                         # Código fuente
│   ├── data_loader.py          # Carga de datos
│   ├── feature_engineering.py   # Creación de features
│   ├── models.py                # Definición de modelos
│   ├── walk_forward.py          # Walk-forward testing
│   ├── evaluation.py            # Métricas de evaluación
│   └── utils.py                 # Utilidades
├── scripts/                     # Scripts de ejecución
│   ├── train_models.py          # Script principal
│   └── run_slurm.sh            # Script SLURM
├── results/                     # Resultados (se crea automáticamente)
│   ├── models/                  # Modelos entrenados
│   ├── predictions/             # Predicciones
│   └── metrics/                 # Métricas calculadas
└── configs/                     # Configuraciones
    └── model_configs.yaml       # Configuración de modelos
```

## Ejecución en SLURM

### Requisitos
- 64 cores
- Memoria estimada: 32GB
- Tiempo estimado: 48-72 horas

### Comando
```bash
cd 1_week/scripts
sbatch run_slurm.sh
```

## Pipeline de Ejecución

1. **Preparación de datos**
   - Cargar velas horarias de Bitcoin
   - Dividir en 3 años (training, validation, confidence)
   - Aplicar gaps de 9 días entre datasets

2. **Feature Engineering**
   - Features básicos: OHLCV
   - Features técnicos: RSI, MACD, Bollinger Bands
   - Features temporales: hora del día, día de la semana
   - Normalización

3. **Walk-Forward Training** (Año 1)
   - Para cada día del año 1:
     - Tomar 168 horas previas como input
     - Predecir 24 horas siguientes
     - Entrenar todos los modelos en paralelo
   - Seleccionar mejor modelo/parámetros

4. **Validation** (Año 2)
   - Aplicar mejor modelo del año 1
   - Walk-forward en año 2
   - Calcular métricas de rendimiento

5. **Confidence Intervals** (Año 3)
   - Usar año 3 para calcular intervalos de confianza
   - Métodos empíricos (quantiles)

## Consideraciones Importantes

### No-Estacionariedad
El mercado de Bitcoin es altamente no-estacionario. Según el PDF (página 18-19):
- Evaluar degradación de rendimiento con fold sizes crecientes
- Re-entrenar periódicamente si es necesario

### Consistencia (Stratification)
- Dividir el año de entrenamiento en estratos mensuales
- Optimizar para rendimiento consistente, no solo promedio
- Evita modelos que funcionan bien en algunos periodos y mal en otros

### Success Ratios
Para trading, más importante que MSE:
- Minimizar pérdidas grandes
- Maximizar ratio ganancias/pérdidas
- Considerar costos de transacción

## Validación Estadística (CRÍTICO)

**IMPORTANTE**: Después del entrenamiento, es OBLIGATORIO ejecutar tests de validación estadística para confirmar que los resultados no son producto del azar.

### Pipeline de Validación (4 Tests)

Ubicado en `validation/`, implementa el framework completo de Masters y neurotrader888:

1. **In-Sample Excellence Test**: ¿El modelo funciona en training data?
2. **In-Sample Permutation Test**: ¿El rendimiento in-sample es significativo? (p-value)
3. **Walk-Forward Test**: ¿El modelo generaliza a datos out-of-sample?
4. **Walk-Forward Permutation Test**: ¿El rendimiento walk-forward es significativo? (p-value)

### Ejecutar Validación Completa

```bash
cd validation
./run_validation_example.sh
```

O manualmente:
```bash
python full_validation_pipeline.py \
    --data ../data/raw/bitcoin_hourly.csv \
    --model-type RandomForestRegressor \
    --n-permutations 1000 \
    --metric sharpe_ratio \
    --output-dir results/full_validation/
```

### Criterios de Aceptación

El modelo es válido SOLO si:
- ✅ Test 1 (In-Sample Excellence): PASS
- ✅ Test 2 (In-Sample Permutation): p < 0.05
- ✅ Test 3 (Walk-Forward): PASS
- ✅ Test 4 (Walk-Forward Permutation): **p < 0.05** ← CRÍTICO

Si Test 4 falla (p ≥ 0.05) → **Rendimiento es SUERTE, NO usar el modelo**

Ver documentación completa en `validation/README.md`

## Próximos Pasos

1. Preparar datos de Bitcoin (velas horarias, 3 años)
2. Ejecutar pipeline de entrenamiento
3. **VALIDACIÓN ESTADÍSTICA (OBLIGATORIO)**
4. Analizar resultados
5. Seleccionar mejor modelo
6. Calcular intervalos de confianza
7. Evaluar estabilidad temporal

## Referencias

- **Masters, T. (2018)**. "Assessing and Improving Prediction and Classification"
  - Capítulo 1: Assessment of Numeric Predictions
  - Walk-forward testing: páginas 14-15
  - Overlap considerations: páginas 16-18
  - Performance measures: páginas 21-28
  - Permutation tests: Capítulo sobre validación estadística

- **neurotrader888** (2025). [Monte Carlo Permutation Tests](https://github.com/neurotrader888/mcpt)
  - Implementación práctica de MCPT para trading strategies
  - Framework de validación estadística
  - MIT License
