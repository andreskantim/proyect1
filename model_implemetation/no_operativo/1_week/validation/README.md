# Validation Module - Monte Carlo Permutation Tests

Este módulo implementa tests estadísticos rigurosos para validar si el rendimiento de los modelos de predicción es estadísticamente significativo o simplemente resultado del azar.

## Metodología

Basado en:
- **Timothy Masters**: "Assessing and Improving Prediction and Classification"
- **neurotrader888's MCPT**: [GitHub Repository](https://github.com/neurotrader888/mcpt)
- **White's Reality Check**: Metodología para testing múltiple

## Componentes

### 1. Permutation Tests (`permutation_tests/`)

**Propósito**: Determinar si el rendimiento del modelo es estadísticamente significativo.

**Cómo funciona**:
1. Calcula métrica de rendimiento en datos reales
2. Permuta los datos N veces (destruyendo patrones temporales)
3. Calcula métrica en cada permutación
4. Calcula p-value: ¿cuántas permutaciones superan el rendimiento real?
5. Si p < 0.05, el rendimiento es estadísticamente significativo

**Ventajas**:
- No asume distribuciones específicas
- Robusto estadísticamente
- Detecta overfitting

**Uso**:
```bash
python permutation_tests/permutation_test.py \
    --model-path ../results/models/best_model.pkl \
    --data ../data/processed/validation_data.csv \
    --n-permutations 1000 \
    --metric sharpe_ratio \
    --output results/permutation_results.json \
    --plot results/permutation_distribution.png
```

**Interpretación**:
- **p < 0.01**: Altamente significativo (***) - Rendimiento real, muy poco probable por azar
- **p < 0.05**: Significativo (**) - Rendimiento probablemente real
- **p < 0.10**: Marginalmente significativo (*) - Evidencia débil
- **p ≥ 0.10**: No significativo - Puede ser suerte

### 2. Monte Carlo Simulation (`monte_carlo/`)

**Propósito**: Estimar distribución de resultados posibles y calcular intervalos de confianza.

**Tipos de simulación**:

#### a) Trade Resampling
- Reordena secuencia de trades aleatoriamente
- Pregunta: "¿Importa el orden de los trades?"
- Útil para: Estimar variabilidad de rendimiento

#### b) Error Bootstrapping
- Remuestra errores de predicción
- Reconstruye predicciones con errores bootstrapped
- Útil para: Intervalos de confianza de métricas

#### c) Random Entry
- Simula entradas/salidas aleatorias
- Baseline de comparación
- Pregunta: "¿El modelo supera timing aleatorio?"

**Uso**:
```bash
python monte_carlo/monte_carlo_test.py \
    --predictions predictions.csv \
    --actual actual.csv \
    --n-simulations 10000 \
    --confidence-level 0.95 \
    --simulation-type all \
    --output results/monte_carlo_results.json
```

**Interpretación**:
- **Intervalos de Confianza**: Rango esperado de rendimiento
- **Worst/Best Case**: Escenarios extremos
- **Comparación con Random**: Si modelo ≈ random entry → no funciona

## Estructura de Archivos

```
validation/
├── __init__.py                           # Módulo principal
├── README.md                             # Esta documentación
├── utils.py                              # Utilidades compartidas
│
├── permutation_tests/                    # Tests de permutación
│   ├── __init__.py
│   └── permutation_test.py              # Script principal
│
├── monte_carlo/                          # Simulaciones Monte Carlo
│   ├── __init__.py
│   └── monte_carlo_test.py              # Script principal
│
└── results/                              # Resultados (se crea automáticamente)
    ├── permutation_results.json
    ├── monte_carlo_results.json
    └── plots/
```

## Pipeline Completo de Validación

### Paso 1: Entrenar Modelos
```bash
cd ../scripts
python train_models.py
```

### Paso 2: Permutation Test
```bash
cd ../validation/permutation_tests
python permutation_test.py \
    --model-path ../../results/models/best_model.pkl \
    --data ../../data/processed/validation_data.csv \
    --n-permutations 1000 \
    --metric sharpe_ratio \
    --output ../results/permutation_results.json \
    --plot ../results/plots/permutation_dist.png
```

### Paso 3: Monte Carlo Simulation
```bash
cd ../monte_carlo
python monte_carlo_test.py \
    --predictions ../../results/predictions/predictions.csv \
    --actual ../../results/predictions/actual.csv \
    --n-simulations 10000 \
    --output ../results/monte_carlo_results.json
```

### Paso 4: Analizar Resultados
```python
import json

# Cargar resultados de permutation test
with open('results/permutation_results.json') as f:
    perm_results = json.load(f)

print(f"p-value: {perm_results['p_value']}")
print(f"Significativo: {perm_results['is_significant']['p_0.05']}")

# Cargar resultados de Monte Carlo
with open('results/monte_carlo_results.json') as f:
    mc_results = json.load(f)

sharpe_ci = mc_results['simulations']['error_bootstrap']['metrics']['sharpe_ratio']
print(f"Sharpe Ratio 95% CI: [{sharpe_ci['ci_lower']}, {sharpe_ci['ci_upper']}]")
```

## Criterios de Validación

Un modelo se considera **válido y robusto** si cumple:

### Criterios Mínimos:
1. ✅ **Permutation Test**: p < 0.05 (significativo)
2. ✅ **Sharpe Ratio**: > 1.0 (preferible > 1.5)
3. ✅ **Profit Factor**: > 1.5
4. ✅ **MC Confidence**: Límite inferior del CI > 0

### Criterios Óptimos:
1. ⭐ **Permutation Test**: p < 0.01 (altamente significativo)
2. ⭐ **Sharpe Ratio**: > 2.0
3. ⭐ **Profit Factor**: > 2.0
4. ⭐ **Consistencia**: Rendimiento estable en todos los folds
5. ⭐ **Superioridad**: Rendimiento >> random entry baseline

## Métricas Disponibles

### Para Permutation Tests:
- `sharpe_ratio`: Ratio de Sharpe (retorno/riesgo)
- `profit_factor`: Factor de ganancias/pérdidas
- `rmse`: Error cuadrático medio
- `mae`: Error absoluto medio
- `r2`: Coeficiente de determinación
- `directional_accuracy`: Precisión direccional

### Para Monte Carlo:
- `total_return`: Retorno total acumulado
- `sharpe_ratio`: Ratio de Sharpe
- `max_drawdown`: Máxima caída
- `win_rate`: Tasa de trades ganadores
- `mse/rmse`: Errores de predicción

## Interpretación de Resultados

### Ejemplo de Resultado Bueno:
```json
{
  "p_value": 0.008,              // Altamente significativo
  "percentile_rank": 99.2,       // Top 1% de permutaciones
  "real_metric": 2.34,           // Sharpe > 2
  "permuted_metrics": {
    "mean": 0.12,                // Promedio de permutaciones mucho menor
    "max": 1.45                  // Incluso el mejor es peor que real
  }
}
```

### Ejemplo de Resultado Malo:
```json
{
  "p_value": 0.523,              // NO significativo
  "percentile_rank": 47.7,       // ~50% - igual que azar
  "real_metric": 0.89,           // Sharpe < 1
  "permuted_metrics": {
    "mean": 0.85,                // Casi igual que real
    "max": 2.31                  // Algunas permutaciones superan real!
  }
}
```

## Advertencias Importantes

### ⚠️ Sobreajuste (Overfitting)
Si el modelo pasa validation set pero falla permutation test:
- Probablemente overfitting a patrones espurios
- No usar en producción

### ⚠️ Data Snooping
- No optimizar parámetros en el validation set
- No ejecutar múltiples tests sin corrección Bonferroni
- Usar confidence set para validación final

### ⚠️ Costo Computacional
- Permutation tests: ~1000 entrenamientos del modelo
- Monte Carlo: ~10000 simulaciones
- Usar SLURM con múltiples cores
- Tiempo estimado: 4-12 horas según modelo

## Referencias

1. **Masters, T. (2018)**. "Assessing and Improving Prediction and Classification"
   - Capítulo sobre Permutation Tests y Monte Carlo validation

2. **White, H. (2000)**. "A Reality Check for Data Snooping"
   - Econometrica, metodología para testing múltiple

3. **neurotrader888** (2025). [Monte Carlo Permutation Tests](https://github.com/neurotrader888/mcpt)
   - Implementación práctica para trading strategies

4. **Efron, B. & Tibshirani, R. (1993)**. "An Introduction to the Bootstrap"
   - Fundamentos de bootstrap y permutation tests

## Preguntas Frecuentes

**P: ¿Cuántas permutaciones necesito?**
R: 1000 es estándar, 10000 para mayor precisión. Más de 10000 rara vez es necesario.

**P: ¿Qué métrica usar?**
R: Para trading, usa `sharpe_ratio` o `profit_factor`. Para predicción pura, usa `rmse` o `mae`.

**P: ¿Qué hacer si p > 0.05?**
R: El modelo NO es estadísticamente significativo. No usar en producción. Revisar features, modelo, o aceptar que no hay señal predictiva.

**P: ¿Diferencia entre permutation y Monte Carlo?**
R: Permutation test responde "¿es significativo?". Monte Carlo responde "¿qué tan variable es el rendimiento?".

**P: ¿Puedo usar esto para optimizar hiperparámetros?**
R: NO. Esto es solo para validación final. Optimiza en training set, valida aquí solo una vez.
