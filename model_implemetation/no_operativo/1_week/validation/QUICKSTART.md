# Validation Quick Start Guide

## ¿Qué es esto?

Un sistema completo de validación estadística para determinar si tu modelo de predicción de Bitcoin funciona **DE VERDAD** o es solo **SUERTE**.

## Los 4 Tests Críticos

```
┌─────────────────────────────────────────────────────────────┐
│  TEST 1: In-Sample Excellence                               │
│  ¿El modelo aprende algo en training data?                  │
│  → Si NO pasa: modelo inútil                                │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  TEST 2: In-Sample Permutation (p-value)                    │
│  ¿El rendimiento in-sample es real o ruido?                 │
│  → Si p ≥ 0.05: overfitting a ruido                         │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  TEST 3: Walk-Forward                                       │
│  ¿El modelo funciona en datos nuevos (out-of-sample)?       │
│  → Si NO pasa: no generaliza                                │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  TEST 4: Walk-Forward Permutation (p-value) ★ CRÍTICO ★     │
│  ¿El rendimiento out-of-sample es suerte o real?            │
│  → Si p ≥ 0.05: ES SUERTE, NO USAR EL MODELO               │
└─────────────────────────────────────────────────────────────┘
```

## Uso Rápido

### Opción 1: Script Automático (Recomendado)

```bash
cd validation
./run_validation_example.sh
```

### Opción 2: Manual

```bash
cd validation

python full_validation_pipeline.py \
    --data ../data/raw/bitcoin_hourly.csv \
    --model-type RandomForestRegressor \
    --n-permutations 1000 \
    --metric sharpe_ratio \
    --output-dir results/full_validation/ \
    --n-jobs -1
```

### Parámetros Disponibles

- `--model-type`: RandomForestRegressor, GradientBoostingRegressor, SVR, MLPRegressor
- `--metric`: sharpe_ratio, profit_factor, rmse, r2
- `--n-permutations`: Número de permutaciones (1000 = estándar, 10000 = muy preciso)
- `--train-ratio`: Ratio train/test (default: 0.7)
- `--n-jobs`: Cores a usar (-1 = todos)

## Interpretación de Resultados

### ✅ RESULTADO BUENO (Modelo Válido)

```
FINAL VALIDATION SUMMARY
================================================================================
Test Results:
  Test 1 (In-Sample Excellence):        ✓ PASS
  Test 2 (In-Sample Permutation p<0.05): ✓ PASS
  Test 2 (In-Sample Permutation p<0.01): ✓ PASS
  Test 3 (Walk-Forward):                 ✓ PASS
  Test 4 (Walk-Forward Perm p<0.05):     ✓ PASS
  Test 4 (Walk-Forward Perm p<0.01):     ✓ PASS

Overall Status:
  ✓✓✓ ALL TESTS PASSED (strict)

Recommendation:
  EXCELLENT - Model is highly significant and not due to luck. Safe for production.

Key p-values:
  In-Sample Permutation:   0.003240
  Walk-Forward Permutation: 0.007850  ← p < 0.01 = EXCELENTE!
================================================================================
```

**Interpretación**: Tu modelo es estadísticamente significativo. El rendimiento NO es suerte.

### ❌ RESULTADO MALO (Modelo Inválido)

```
FINAL VALIDATION SUMMARY
================================================================================
Test Results:
  Test 1 (In-Sample Excellence):        ✓ PASS
  Test 2 (In-Sample Permutation p<0.05): ✓ PASS
  Test 3 (Walk-Forward):                 ✓ PASS
  Test 4 (Walk-Forward Perm p<0.05):     ✗ FAIL  ← PROBLEMA
  Test 4 (Walk-Forward Perm p<0.01):     ✗ FAIL

Overall Status:
  ✗ FAILED

Recommendation:
  REJECT - Performance likely due to luck or overfitting. Do NOT use in production.

Key p-values:
  In-Sample Permutation:   0.021000
  Walk-Forward Permutation: 0.347000  ← p > 0.05 = SUERTE!
================================================================================
```

**Interpretación**: El rendimiento walk-forward NO es significativo. Es SUERTE. NO usar.

## ¿Qué Significa el p-value?

El **p-value** responde: "¿Cuál es la probabilidad de obtener este rendimiento por puro azar?"

- **p < 0.01** (1%): Altamente significativo ✓✓✓ - Solo 1% de probabilidad de ser suerte
- **p < 0.05** (5%): Significativo ✓✓ - Estándar científico, 5% probabilidad de suerte
- **p < 0.10** (10%): Marginalmente significativo ✓ - Evidencia débil
- **p ≥ 0.10**: NO significativo ✗ - Muy probablemente es SUERTE

### Ejemplo Visual

Imagina que ejecutas 1000 permutaciones (datos aleatorios):

```
Distribución de Sharpe Ratio en datos aleatorios:

     │                  ╭─╮
 200 │              ╭───╯ ╰───╮
 150 │          ╭───╯         ╰───╮
 100 │      ╭───╯                 ╰───╮
  50 │  ╭───╯                         ╰───╮
   0 └──┴────┴────┴────┴────┴────┴────┴────┴──
     -1.0  -0.5  0.0  0.5  1.0  1.5  2.0  2.5
                                          ↑
                                      Tu modelo = 2.1
                                      Percentil 99.2%
                                      p-value = 0.008
                                      ✓ SIGNIFICATIVO!
```

Si tu modelo está en el top 5% (p < 0.05) → NO es suerte.
Si está en el centro (p > 0.10) → ES suerte.

## Flujo de Trabajo Completo

```bash
# 1. Descargar datos
cd ../scripts
python download_bitcoin_data.py --output ../data/raw/bitcoin_hourly.csv

# 2. Entrenar modelos (si tienes pipeline de training)
python train_models.py

# 3. VALIDACIÓN (CRÍTICO)
cd ../validation
python full_validation_pipeline.py \
    --data ../data/raw/bitcoin_hourly.csv \
    --model-type RandomForestRegressor \
    --n-permutations 1000 \
    --metric sharpe_ratio \
    --output-dir results/

# 4. Revisar resultados
cat results/full_validation/validation_results_*.json | jq '.final_verdict'

# 5. Si p-value < 0.05 → continuar
# 6. Si p-value ≥ 0.05 → DESCARTAR modelo, probar otros
```

## Tests Individuales

Si solo quieres ejecutar tests específicos:

### Solo Permutation Test

```bash
cd permutation_tests

python permutation_test.py \
    --model-path ../../results/models/best_model.pkl \
    --data ../../data/processed/validation_data.csv \
    --n-permutations 1000 \
    --metric sharpe_ratio \
    --output ../results/permutation_results.json \
    --plot ../results/permutation_plot.png
```

### Solo Monte Carlo Simulation

```bash
cd monte_carlo

python monte_carlo_test.py \
    --predictions ../../results/predictions/predictions.csv \
    --actual ../../results/predictions/actual.csv \
    --n-simulations 10000 \
    --output ../results/monte_carlo_results.json
```

## Preguntas Frecuentes

**P: ¿Por qué son necesarios estos tests?**
R: Porque es EXTREMADAMENTE fácil crear un modelo que funcione "bien" en backtesting pero que sea **completamente suerte**. Estos tests son la única forma rigurosa de distinguir suerte de habilidad real.

**P: ¿Cuánto tiempo toma?**
R: Con 1000 permutaciones y -1 cores (todos), aproximadamente:
- Random Forest: 30-60 minutos
- Gradient Boosting: 1-2 horas
- SVR: 2-4 horas
- MLP: 1-3 horas

**P: ¿Puedo reducir el número de permutaciones para ir más rápido?**
R: Sí, pero con menor precisión:
- 100 permutations: Test rápido (~5-10 min), precisión baja
- 1000 permutations: Estándar recomendado (~1-2 horas), buena precisión
- 10000 permutations: Alta precisión (~10-20 horas), p-values muy precisos

**P: Mi modelo pasó Test 1 y 2 pero falló Test 4. ¿Qué significa?**
R: Significa **overfitting**. El modelo aprendió patrones específicos del training set que no se generalizan. NO usar.

**P: ¿Qué hacer si mi modelo falla Test 4?**
R: Opciones:
1. Probar diferentes features
2. Probar diferentes modelos
3. Reducir complejidad del modelo (regularización)
4. Aumentar datos de training
5. Aceptar que quizás no hay señal predictiva en los datos

**P: ¿Puedo optimizar hiperparámetros usando estos tests?**
R: **NO**. Estos tests son SOLO para validación final de UN modelo. Si optimizas con estos tests, estás haciendo data snooping y los p-values serán inválidos.

## Documentación Completa

Ver `README.md` en este directorio para:
- Explicación detallada de metodología
- Matemáticas detrás de permutation tests
- Interpretación avanzada de resultados
- Referencias bibliográficas
- Ejemplos de código

## Soporte

Si tienes preguntas o problemas:
1. Lee `README.md` completo
2. Revisa los ejemplos en los scripts
3. Verifica que instalaste todas las dependencias: `pip install -r ../requirements.txt`

## Cita

Si usas este framework en investigación, cita:

```bibtex
@book{masters2018assessing,
  title={Assessing and Improving Prediction and Classification},
  author={Masters, Timothy},
  year={2018},
  publisher={Independently published}
}

@software{neurotrader888_mcpt,
  author = {neurotrader888},
  title = {Monte Carlo Permutation Tests},
  year = {2025},
  url = {https://github.com/neurotrader888/mcpt},
  license = {MIT}
}
```
