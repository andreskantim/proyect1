# CONTEXTO DEL PROYECTO: MarketGPT Trading System

> **Propósito**: Este archivo proporciona contexto completo del proyecto para sesiones de Claude.
> **Uso**: Ejecutar `/context` al iniciar una nueva sesión.

---

## 📋 RESUMEN EJECUTIVO

**Nombre**: MarketGPT - Sistema de Trading con IA
**Objetivo**: Sistema completo de predicción de mercados financieros y generación de señales de trading (LONG/SHORT) de alta confianza
**Tecnología**: Transformers (réplicas de LLM tipo GPT) entrenados con datos de bolsa + PyTorch + Multi-GPU (2×A100)
**Ubicación**: `/mnt/netapp2/Home_FT2/home/ulc/cursos/curso396/LLM-T/stock_predictor/`

**Concepto Clave**: Entrenamos 3 modelos base (réplicas de arquitectura LLM) con datos históricos de mercados. Cada modelo aprende patrones de precios OHLC tokenizados. Luego especializamos por mercado y generamos señales de trading validadas con benchmarks.

---

## 🏗️ ARQUITECTURA DEL SISTEMA (3 FASES)

### FASE 1: Pre-entrenamiento General ✅ IMPLEMENTADO

**Objetivo**: Entrenar 3 modelos base (réplicas de LLM) con datos de bolsa

**Concepto**: Entrenamos transformers estilo GPT con datos OHLC tokenizados de diferentes conjuntos de tickers. Cada modelo es una réplica de arquitectura LLM adaptada para series temporales financieras.

**Cases (3 Modelos Base)**:
- **Case A**: 600 tickers (US + EU + EM + Commodities + Crypto) - 7-10 días
  - Modelo universal con máxima cobertura
- **Case B**: 100 tickers (curated multi-market) - 3-5 días
  - Modelo baseline con activos de alta calidad
- **Case C**: 20 tickers crypto (prototipo rápido) - 1-2 días
  - Modelo especializado en criptomonedas

**Datos por Case**:
- Split temporal: Train 70% / Val 15% / Test 15%
- Periodo: 2014-2025 (11 años de historia)
- NO división aleatoria (preserva causalidad temporal)
- Tickers específicos por case (definidos en cada data loader)

**Proceso de Entrenamiento**:
1. **Train set** (70%) → Entrena el modelo (actualiza pesos)
2. **Val set** (15%) → Early stopping + Model selection
3. **Test set** (15%) → Evaluación final ÚNICA (NO usado durante entrenamiento)

**Hardware por Case**: 2×A100 GPUs, 64 CPUs, 128GB RAM

**Output**: `best_model.pt` (modelo general pre-entrenado para cada case)

---

### FASE 2: Fine-Tuning por Mercado ❌ PENDIENTE

**Objetivo**: Dentro de cada case (A, B, C), generar modelos especializados por tipo de mercado

**Proceso**: Partir del modelo general pre-entrenado de cada case y especializarlo para mercados específicos usando tickers propios de cada mercado.

**Mercados Target (dentro de cada Case)**:
1. **US Stocks**: Tickers específicos de acciones estadounidenses
2. **EU Stocks**: Tickers específicos de acciones europeas
3. **Commodities**: Tickers específicos de materias primas
4. **Crypto**: Tickers específicos de criptomonedas

**Importante**: Los tickers usados en fine-tuning son ESPECÍFICOS de cada mercado y diferentes de los que se usarán en benchmarks.

**Método**: Walk-Forward Analysis

```
Train Window: 24 meses
Val Window: 6 meses
Step Forward: 3 meses

|----Train----|Val|
    |----Train----|Val|
        |----Train----|Val|
```

**Prevención Look-Forward Bias**:
- ✅ Nunca usar datos futuros
- ✅ Validar en out-of-sample
- ✅ Ventanas temporales no solapadas
- ✅ Simula trading real

**Output**: 4 modelos especializados × N folds

---

### FASE 3: Generación de Señales y Validación ❌ PENDIENTE

**Objetivo**: Generar señales de trading (LONG/SHORT) de alta confianza y validarlas con benchmarks

#### 3.1 Generación de Señales

**Tipos de Señales**:
- **LONG**: Señales de compra (predicción de subida)
- **SHORT**: Señales de venta (predicción de bajada)

**Condiciones para Señal LONG (TODAS deben cumplirse)**:

1. **Prob Día**: `P(subida_próxima_vela) > 90%`
2. **Prob Horizonte**: `P(subida_horizonte) > 90%`
3. **Rango Bollinger**: `expected_return_up > 2 × σ_Bollinger`

**Condiciones para Señal SHORT (TODAS deben cumplirse)**:

1. **Prob Día**: `P(bajada_próxima_vela) > 90%`
2. **Prob Horizonte**: `P(bajada_horizonte) > 90%`
3. **Rango Bollinger**: `expected_return_down > 2 × σ_Bollinger`

**Horizontes Temporales**:

| Tipo | Horizonte | Días | Uso |
|------|-----------|------|-----|
| Corto | 1 semana | 5 | Day/Swing trading |
| Medio-Corto | 2 semanas | 10 | Swing trading |
| Medio | 1 mes | 22 | Position trading |
| Medio-Largo | 2 meses | 44 | Investment |

**Cálculo Bollinger Bands**:
- Calculadas sobre cada horizonte temporal
- Solo datos históricos (evita look-forward bias)
- 2 desviaciones estándar (ancho de banda similar a Bollinger tradicional)

#### 3.2 Validación con Benchmarks

**Importante**: Los benchmarks usan ÍNDICES de cada mercado, diferentes de los tickers usados en fine-tuning.

**Benchmarks por Mercado**:
1. **US Stocks**: Índices como SPY, QQQ, DIA (diferentes de tickers de tuning)
2. **EU Stocks**: Índices como EWU, EWG, EWQ (diferentes de tickers de tuning)
3. **Commodities**: Índices como GLD, USO, DBA (diferentes de tickers de tuning)
4. **Crypto**: Índices como BTC, ETH (si no están en tuning)

**Proceso de Validación**:
1. Modelo especializado genera señales (LONG/SHORT) en índices benchmark
2. Se ejecutan entradas y salidas según señales
3. Se mide performance real de la estrategia

#### 3.3 Métricas

**Métricas de Entrenamiento (durante generación de señales)**:
- **Tasa de acierto de señales**: % de señales que resultan correctas
- **Precisión LONG**: % de señales LONG correctas
- **Precisión SHORT**: % de señales SHORT correctas
- **False Positive Rate**: % de señales incorrectas
- **Signal Frequency**: Número de señales generadas por periodo

**Métricas de Benchmark (en índices)**:
- **Win Rate**: % de operaciones ganadoras
- **Profit Factor**: Ganancia total / Pérdida total
- **Sharpe Ratio**: Retorno ajustado por riesgo
- **Max Drawdown**: Pérdida máxima desde pico
- **Average Return per Trade**: Retorno promedio por operación
- **Total Return**: Retorno acumulado del periodo

---

### FASE 4: Optimización y Monitoreo ❌ FUTURO

**Objetivo**: Ajustar el sistema de benchmarking y detectar deterioro de la estrategia

#### 4.1 Ajuste del Benchmark

**Pendiente para implementación posterior**:
- Optimización de parámetros de entrada/salida
- Refinamiento de umbrales de probabilidad
- Ajuste de gestión de riesgo (stop-loss, take-profit)
- Testing de diferentes horizontes temporales

#### 4.2 Detección de Deterioro

**Sistema de Monitoreo Temporal**:
- Métricas de performance por periodo
- Detección de cambios de régimen de mercado
- Alertas de degradación de modelo
- Comparación de performance actual vs histórica

**Indicadores de Deterioro**:
- **Win Rate Drift**: Caída sostenida del % de acierto
- **Sharpe Ratio Decline**: Reducción del retorno ajustado por riesgo
- **Max Drawdown Increase**: Aumento de pérdidas máximas
- **Signal Frequency Change**: Cambio significativo en número de señales

**Acciones ante Deterioro**:
1. Re-entrenamiento con datos más recientes
2. Re-ajuste de walk-forward windows
3. Revisión de tickers/índices de benchmark
4. Evaluación de cambio estructural del mercado

---

## 💾 ESTADO ACTUAL DEL PROYECTO

### ✅ COMPLETADO

**Código**:
- [x] Arquitectura MarketGPT (transformer GPT-style)
- [x] MarketGPTMultiAsset (multi-asset con embeddings)
- [x] OHLCTokenizer (quantile-based, 4 canales)
- [x] Data loaders para 3 cases
- [x] Scripts entrenamiento multi-GPU (2×A100)
- [x] Train/Val/Test splits temporales
- [x] Evaluación final en test set ✅ **AÑADIDO HOY**
- [x] Distributed training (DataParallel)

**Documentación**:
- [x] SYSTEM_ARCHITECTURE.md (arquitectura completa)
- [x] QUICK_REFERENCE.md (referencia rápida)
- [x] TRAINING_GUIDE.md (guía multi-GPU)
- [x] READMEs por case (A, B, C)
- [x] CLAUDE_CONTEXT.md (este archivo)

**Bugs Corregidos**:
- [x] Case C: DatetimeArray.sort() error → `sorted()`
- [x] Case C: CUDA module error → removido
- [x] Cases A/B/C: Test set no se usaba → añadida evaluación final
- [x] Case B: Faltaba --num-gpus → añadido

### ❌ PENDIENTE

**Fase 2: Fine-Tuning por Mercado**:
- [ ] Scripts de fine-tuning por mercado (US, EU, Commodities, Crypto)
- [ ] Pipeline walk-forward analysis completo
- [ ] Gestión de múltiples folds
- [ ] Ensemble de modelos por fold
- [ ] Definir tickers específicos por mercado para tuning

**Fase 3: Generación de Señales y Validación**:
- [ ] Motor de generación de señales (LONG/SHORT)
- [ ] Cálculo multi-horizonte Bollinger Bands (1w, 2w, 1m, 2m)
- [ ] Sistema de predicción de probabilidades por horizonte
- [ ] Definir índices de benchmark por mercado (diferentes de tuning)
- [ ] Sistema de backtesting en benchmarks
- [ ] Métricas de entrenamiento (tasa acierto, precision LONG/SHORT)
- [ ] Métricas de benchmark (win rate, profit factor, sharpe, drawdown)
- [ ] Logging de señales generadas

**Fase 4: Optimización y Monitoreo (Futuro)**:
- [ ] Sistema de ajuste de parámetros de benchmark
- [ ] Detección de deterioro temporal
- [ ] Dashboard de métricas de deterioro
- [ ] Sistema de re-entrenamiento automático

**Infraestructura General**:
- [ ] Dashboard de monitoreo en tiempo real
- [ ] API de señales
- [ ] Base de datos de señales históricas
- [ ] Sistema de alertas

---

## 📁 ESTRUCTURA DEL PROYECTO

```
stock_predictor/
├── 📄 CLAUDE_CONTEXT.md           ← Este archivo
├── 📄 SYSTEM_ARCHITECTURE.md      ← Arquitectura completa
├── 📄 QUICK_REFERENCE.md          ← Referencia rápida
├── 📄 TRAINING_GUIDE.md           ← Guía entrenamiento
│
├── 📂 common/                     ← Módulos compartidos
│   ├── market_gpt.py              ← Transformer base
│   ├── market_gpt_multi.py        ← Multi-asset model
│   ├── tokenizer.py               ← OHLC tokenizer
│   ├── training_monitor.py        ← Monitor entrenamiento
│   └── distributed_utils.py       ← Multi-GPU utils
│
├── 📂 case_a_full_market/         ← 600 assets
│   ├── universal_loader.py
│   ├── train_full.py              ← Training script (2 GPUs)
│   ├── configs/full_market_config.json
│   ├── slurm_scripts/train_full_a100.sh
│   └── README.md
│
├── 📂 case_b_reduced/             ← 100 assets
│   ├── multi_market_loader.py
│   ├── train_reduced.py           ← Training script (2 GPUs)
│   ├── configs/reduced_config.json
│   ├── slurm_scripts/train_reduced_a100.sh
│   └── README.md
│
├── 📂 case_c_crypto/              ← 20 cryptos
│   ├── crypto_data_loader.py
│   ├── train_crypto.py            ← Training script (2 GPUs)
│   ├── configs/crypto_prototype.json
│   ├── slurm_scripts/train_crypto_a100.sh
│   └── README.md
│
└── 📂 checkpoints/                ← Modelos entrenados
    ├── case_a_full_market/
    ├── case_b_reduced/
    └── case_c_crypto/
```

---

## 🔑 CONCEPTOS CLAVE

### 1. Tokenización OHLC
```python
# Convierte precios OHLC a tokens discretos
# Método: Quantile-based binning
# Canales: 4 (Open, High, Low, Close)
# Vocab size: 1024-4096 bins

OHLCTokenizer.encode(ohlc) → token_ids
```

### 2. Multi-Asset Embeddings
```python
# El modelo aprende embeddings por:
# - Asset ID (identificador único del activo)
# - Category ID (US stocks, EU stocks, crypto, etc.)

model(tokens, asset_id, category_id) → logits
```

### 3. Walk-Forward Analysis
```
Evita look-forward bias:
- Entrena en ventana pasada (2 años)
- Valida en ventana siguiente (6 meses)
- Avanza 3 meses
- Repite

Simula trading real en el tiempo
```

### 4. Prevención de Look-Forward Bias
```
✓ División temporal (no aleatoria)
✓ Test set solo UNA vez
✓ Bollinger solo datos históricos
✓ Tokenizer fit solo en train
✓ Sin información futura en features
✓ Timestamps estrictos
```

### 5. Señales de Alta Confianza
```
Solo genera señal si:
  P(subida_día) > 90%
  AND
  P(subida_horizonte) > 90%
  AND
  expected_return > 2σ_Bollinger

→ Triple filtro de confianza
```

---

## 🚀 COMANDOS FRECUENTES

### Entrenar Models

```bash
# Case C (más rápido, 1-2 días)
cd case_c_crypto/slurm_scripts
sbatch train_crypto_a100.sh

# Case B (medio, 3-5 días)
cd case_b_reduced/slurm_scripts
sbatch train_reduced_a100.sh

# Case A (completo, 7-10 días)
cd case_a_full_market/slurm_scripts
sbatch train_full_a100.sh
```

### Monitorear

```bash
# Ver jobs
squeue -u $(whoami)

# Logs en tiempo real
tail -f case_c_crypto/logs/crypto_*.out

# Ver GPUs
ssh <node>
nvidia-smi
```

### Checkpoints

```
checkpoints/case_X/<experiment>/
├── best_model.pt          ← Mejor modelo (val_loss)
├── test_results.json      ← Métricas en test
├── training_log.json      ← Historial entrenamiento
├── tokenizer.pkl          ← Tokenizer fitted
└── asset_info.json        ← Mapeo assets
```

---

## 📊 DETALLES TÉCNICOS

### Hardware por Case
- **GPUs**: 2×A100-40GB (un nodo, DataParallel)
- **CPUs**: 64 (32 por GPU)
- **RAM**: 128GB (64GB por GPU)
- **Batch size**: 32 base → 64 efectivo (32×2 GPUs)

### Model Architecture (Case A - Full)
```python
vocab_size = 4096
context_length = 512
d_model = 768
num_layers = 12
num_heads = 12
d_ff = 3072
dropout = 0.1
asset_embed_dim = 64
category_embed_dim = 32

Total params: ~85M
```

### Model Architecture (Case B - Reduced)
```python
vocab_size = 2048
context_length = 256
d_model = 512
num_layers = 8
num_heads = 8
d_ff = 2048

Total params: ~45M
```

### Model Architecture (Case C - Crypto)
```python
vocab_size = 1024
context_length = 128
d_model = 256
num_layers = 6
num_heads = 8
d_ff = 1024

Total params: ~25M
```

---

## 🎯 WORKFLOW DE TRABAJO TÍPICO

### Nueva Sesión

1. **Cargar contexto**: `/context` (este comando)
2. **Revisar estado**: Ver sección "Estado Actual"
3. **Identificar tarea**: Consultar sección "Pendiente"
4. **Consultar docs**: Ver SYSTEM_ARCHITECTURE.md si es necesario

### Implementar Nueva Feature

1. **Consultar arquitectura**: SYSTEM_ARCHITECTURE.md
2. **Ver ejemplos**: Revisar code existente en common/
3. **Implementar**: Seguir patrones del proyecto
4. **Testear**: Usar datos de validación
5. **Documentar**: Actualizar READMEs y este archivo

### Debugging

1. **Revisar logs**: `case_*/logs/*.err`
2. **Verificar GPU**: `nvidia-smi` en nodo
3. **Consultar fixes**: Sección "Bugs Corregidos"
4. **Checkear paths**: Todo en `/mnt/netapp2/...`

---

## 📖 REFERENCIAS RÁPIDAS

### Documentos Principales
- **Arquitectura completa**: `SYSTEM_ARCHITECTURE.md`
- **Referencia rápida**: `QUICK_REFERENCE.md`
- **Guía entrenamiento**: `TRAINING_GUIDE.md`

### Secciones Importantes de SYSTEM_ARCHITECTURE.md
- Sección 2: Fase 1 (Pre-entrenamiento)
- Sección 3: Fase 2 (Fine-tuning + Walk-forward)
- Sección 4: Fase 3 (Generación de señales)
- Sección 5: Prevención look-forward bias
- Sección 6: Pipeline producción

### Preguntas Frecuentes

**P: ¿Cómo funcionan las señales?**
→ Ver SYSTEM_ARCHITECTURE.md sección 4

**P: ¿Qué es walk-forward?**
→ Ver SYSTEM_ARCHITECTURE.md sección 3.3

**P: ¿Cómo entrenar un case?**
→ Ver TRAINING_GUIDE.md

**P: ¿Cómo evitar look-forward bias?**
→ Ver SYSTEM_ARCHITECTURE.md sección 5

**P: ¿Estado del proyecto?**
→ Ver este archivo, sección "Estado Actual"

---

## 🔄 ÚLTIMAS ACTUALIZACIONES

### 2025-01-06 (Hoy - Actualización 2)

**Aclaraciones Arquitectura**:
- ✅ Clarificado: Modelos son "réplicas de LLM" entrenadas con datos de bolsa
- ✅ Añadido: Señales LONG y SHORT (no solo compra)
- ✅ Especificado: Benchmarks usan ÍNDICES diferentes de tickers de tuning
- ✅ Definido: Métricas de entrenamiento Y de benchmark
- ✅ Añadido: FASE 4 (optimización benchmark + detección deterioro)

**Documentación Actualizada**:
- CLAUDE_CONTEXT.md con aclaraciones completas
- FASE 3 expandida con señales LONG/SHORT
- Sección de benchmarks con índices específicos
- Métricas detalladas (entrenamiento + benchmark)
- Plan futuro de detección de deterioro

**Conceptos Clave Aclarados**:
- 3 modelos base (Cases A, B, C) = réplicas LLM con datos financieros
- Fine-tuning usa tickers específicos por mercado
- Benchmarks usan índices DIFERENTES (SPY, QQQ, GLD, etc.)
- Señales LONG (subida) y SHORT (bajada) con 90% confianza
- Walk-forward evita look-forward bias

### 2025-01-06 (Hoy - Actualización 1)

**Añadido**:
- ✅ Evaluación en test set para Cases A, B, C
- ✅ Fix bug Case C (DatetimeArray.sort)
- ✅ Fix bug Case C (CUDA module)
- ✅ Añadido --num-gpus a Case B
- ✅ Documentación completa (3 documentos principales)
- ✅ Sistema de contexto con /context y /ctx

**Corregido**:
- Test set se creaba pero NO se usaba → Ahora se evalúa al final
- Cases A/B/C ahora reportan test_loss y test_accuracy
- Multi-GPU configurado correctamente (2 GPUs por case)

**Documentado**:
- Sistema de fases completo
- Walk-forward analysis en detalle
- Sistema de señales
- Prevención de look-forward bias

---

## 💡 NOTAS IMPORTANTES

### Al Iniciar Sesión
1. **Siempre** ejecutar `/context` al inicio
2. Revisar "Estado Actual" para saber qué está hecho
3. Consultar "Pendiente" para próximas tareas
4. Usar SYSTEM_ARCHITECTURE.md como referencia técnica

### Al Implementar
- Seguir patrones existentes en `common/`
- Mantener consistencia con nomenclatura
- Documentar cambios en este archivo
- Actualizar "Estado Actual" si se completa algo

### Al Hacer Cambios
- Actualizar sección "Últimas Actualizaciones"
- Si afecta arquitectura → actualizar SYSTEM_ARCHITECTURE.md
- Si es nuevo comando → actualizar QUICK_REFERENCE.md
- Si es bug → añadir a "Bugs Corregidos"

---

## 🎓 FILOSOFÍA DEL PROYECTO

1. **No Look-Forward Bias**: Nunca usar información futura
2. **Alta Confianza**: Solo señales con >90% probabilidad
3. **Validación Rigurosa**: Walk-forward en múltiples periodos
4. **Especialización**: Modelos específicos por mercado
5. **Documentación**: Todo debe estar documentado
6. **Reproducibilidad**: Scripts SLURM versionados

---

**FIN DEL CONTEXTO**

> Al leer este archivo tienes contexto completo del proyecto MarketGPT.
> Para detalles técnicos, consulta SYSTEM_ARCHITECTURE.md
> Para comandos rápidos, consulta QUICK_REFERENCE.md
