# MarketGPT: Sistema Completo de Trading con IA

## Arquitectura General del Sistema

Este documento describe la arquitectura completa del sistema de trading basado en MarketGPT, desde el entrenamiento inicial hasta la generación de señales de compra.

---

## 1. Visión General del Flujo de Trabajo

```
┌─────────────────────────────────────────────────────────────────────┐
│                     FASE 1: PRE-ENTRENAMIENTO                       │
│                     (3 Réplicas de LLM con datos financieros)       │
│                                                                     │
│  Case A (600 assets) ──> LLM Replica A (Multi-mercado completo)    │
│  Case B (100 assets) ──> LLM Replica B (Curated baseline)          │
│  Case C (20 cryptos) ──> LLM Replica C (Crypto especializado)      │
│                                                                     │
│  Train/Val/Test: 70%/15%/15%                                       │
│  Objetivo: 3 modelos base entrenados con datos OHLC tokenizados   │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│                  FASE 2: FINE-TUNING POR MERCADO                    │
│          (Dentro de cada case → especialización por mercado)        │
│                                                                     │
│  Cada LLM replica ──┬──> Fine-tune: US Stocks (tickers específicos)│
│  (A, B, C)          ├──> Fine-tune: EU Stocks (tickers específicos)│
│                     ├──> Fine-tune: Commodities (tickers específicos)│
│                     └──> Fine-tune: Crypto (tickers específicos)   │
│                                                                     │
│  Walk-Forward Analysis para evitar Look-Forward Bias               │
│  Tickers de TUNING: Activos específicos por mercado                │
│  Resultado: 4 modelos especializados × N folds (por cada case)     │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│           FASE 3: GENERACIÓN SEÑALES + VALIDACIÓN BENCHMARKS        │
│                                                                     │
│  A. GENERACIÓN DE SEÑALES (LONG/SHORT) cuando:                     │
│     ✓ Condición 1: P(subida/bajada_1_día) > 90%                   │
│     ✓ Condición 2: P(subida/bajada_horizonte) > 90%               │
│     ✓ Condición 3: Rango_esperado > 2σ Bollinger Bands            │
│                                                                     │
│  B. VALIDACIÓN CON BENCHMARKS (índices ≠ tickers tuning):          │
│     • US Stocks: SPY, QQQ, DIA                                     │
│     • EU Stocks: EWU (FTSE), EWG (DAX)                             │
│     • Commodities: GLD (oro), USO (petróleo)                       │
│     • Crypto: BTC, ETH                                             │
│                                                                     │
│  MÉTRICAS:                                                          │
│  - Entrenamiento: Tasa acierto, Precision LONG, Precision SHORT    │
│  - Benchmark: Win rate, Profit factor, Sharpe, Max drawdown        │
│                                                                     │
│  4 horizontes × 2 tipos: Corto, Medio-Corto, Medio, Medio-Largo   │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│          FASE 4: OPTIMIZACIÓN Y MONITOREO (Futuro)                  │
│                                                                     │
│  • Optimización de parámetros de benchmark                         │
│  • Detección de deterioro temporal de la estrategia                │
│  • Re-entrenamiento automático cuando deterioro > threshold        │
│  • Dashboard de monitoreo continuo                                 │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. FASE 1: Pre-entrenamiento General

### 2.1 Objetivo

Entrenar **3 réplicas de LLM** (transformers estilo GPT) con datos financieros OHLC tokenizados. Cada case entrena una réplica independiente del mismo modelo de lenguaje, pero con diferentes datasets:

- **LLM Replica A**: Entrenada con 600 activos multi-mercado
- **LLM Replica B**: Entrenada con 100 activos curated
- **LLM Replica C**: Entrenada con 20 criptomonedas

Cada réplica aprende patrones de comportamiento de precios en su respectivo dataset.

### 2.2 Datasets

| Case | Assets | Mercados | Objetivo | LLM Replica |
|------|--------|----------|----------|-------------|
| **A** | 600 | EEUU, Europa, EM, Commodities, Crypto | Modelo universal | LLM A |
| **B** | 100 | Curated multi-mercado | Baseline rápido | LLM B |
| **C** | 20 | Solo criptomonedas | Especializado crypto | LLM C |

### 2.3 División de Datos

```python
# División temporal (NO aleatoria)
train_ratio = 0.70  # 2014-2021
val_ratio   = 0.15  # 2021-2023
test_ratio  = 0.15  # 2023-2025
```

**Importante**: La división es **temporal** para preservar causalidad.

### 2.4 Proceso de Entrenamiento

```
1. Train set (70%):
   - Entrena el modelo
   - Actualiza pesos

2. Validation set (15%):
   - Evalúa cada época
   - Early stopping
   - Model selection (guarda best_val_loss)

3. Test set (15%):
   - Evaluación final ÚNICA
   - Reporta métricas objetivas
   - NO se usa durante entrenamiento
```

### 2.5 Output

Cada case genera su propia réplica de LLM pre-entrenada:

```
checkpoints/
├── case_a/
│   ├── best_model.pt              # LLM Replica A (600 assets)
│   ├── tokenizer.pkl              # Tokenizer fitted
│   ├── asset_info.json            # Mapeo de assets
│   ├── test_results.json          # Métricas finales
│   └── training_log.json          # Historial completo
├── case_b/
│   ├── best_model.pt              # LLM Replica B (100 assets)
│   └── ...
└── case_c/
    ├── best_model.pt              # LLM Replica C (20 cryptos)
    └── ...
```

**Resultado**: 3 modelos base independientes, cada uno especializado en su dataset.

---

## 3. FASE 2: Fine-Tuning Especializado por Mercado

### 3.1 Objetivo

Crear **modelos especializados** para cada mercado, partiendo de cada LLM replica pre-entrenada (A, B, C). Cada case se fine-tunea independientemente en sus mercados objetivo.

**Importante**:
- Los tickers usados para **fine-tuning** son activos específicos por mercado
- Los tickers usados para **validación benchmark** son DIFERENTES (índices de mercado)

### 3.2 Mercados Target y Tickers de Tuning

**Para Fine-Tuning** (tickers específicos):

1. **Acciones EEUU** (US Stocks)
   - Tickers individuales: AAPL, MSFT, GOOGL, TSLA, AMZN, etc.
   - S&P 500, NASDAQ, NYSE
   - ~300 activos principales

2. **Acciones Europa** (EU Stocks)
   - Tickers individuales: SAP.DE, SAN.MC, RR.L, AIR.PA, etc.
   - FTSE, DAX, CAC, IBEX
   - ~150 activos principales

3. **Commodities**
   - Futuros específicos: GC=F (oro), CL=F (petróleo), SI=F (plata)
   - Metales: Gold, Silver, Copper
   - Energía: Oil, Gas
   - Agricultura: Wheat, Corn
   - ~30 activos

4. **Criptomonedas**
   - Tickers individuales: BTC-USD, ETH-USD, SOL-USD, ADA-USD, etc.
   - ~70 activos

### 3.3 Benchmarks de Validación (DIFERENTES de tickers de tuning)

**Para Validación** (índices de mercado, NO usados en tuning):

1. **US Stocks**: SPY (S&P 500), QQQ (NASDAQ), DIA (Dow Jones)
2. **EU Stocks**: EWU (FTSE 100), EWG (DAX), EWQ (CAC 40)
3. **Commodities**: GLD (oro), USO (petróleo), SLV (plata)
4. **Crypto**: BTC (índice), ETH (índice)

### 3.4 Walk-Forward Analysis

**Objetivo**: Evitar **look-forward bias** usando ventanas móviles.

#### 3.4.1 Metodología

```
Ventana de entrenamiento: 2 años
Ventana de validación: 6 meses
Step forward: 3 meses

Timeline Example:
|----Train----|Val|    Step
|        |----Train----|Val|    Step
|            |----Train----|Val|    Step
```

#### 3.4.2 Proceso

```python
# Pseudocódigo walk-forward
train_window = 24  # meses
val_window = 6     # meses
step_forward = 3   # meses

for start in range(0, total_months - train_window - val_window, step_forward):
    # 1. Define ventanas
    train_start = start
    train_end = start + train_window
    val_start = train_end
    val_end = val_start + val_window

    # 2. Carga modelo pre-entrenado
    model = load_pretrained_model("best_model.pt")

    # 3. Fine-tune en ventana de train
    finetune(model, data[train_start:train_end])

    # 4. Valida en ventana siguiente
    metrics = validate(model, data[val_start:val_end])

    # 5. Guarda modelo si es mejor
    if metrics['val_loss'] < best_val_loss:
        save_checkpoint(model, f"best_model_fold_{fold}.pt")

    # 6. Avanza la ventana
    fold += 1
```

#### 3.4.3 Ventajas

- ✅ **Sin look-forward bias**: Nunca entrena con datos futuros
- ✅ **Realista**: Simula trading en tiempo real
- ✅ **Robusto**: Múltiples validaciones en diferentes periodos
- ✅ **Adaptativo**: Capta cambios de régimen de mercado

### 3.4 Fine-Tuning Technique

```python
# Estrategia de fine-tuning
1. Cargar modelo pre-entrenado general
2. Congelar capas base (opcional)
3. Descongelar últimas capas
4. Entrenar con learning rate bajo
5. Validar con walk-forward

# Hyperparameters típicos
learning_rate = 1e-5  # 10x menor que pre-training
epochs = 10-20        # Menos que pre-training
batch_size = 32       # Ajustar según mercado
```

### 3.5 Output

```
fine_tuned_models/
├── us_stocks/
│   ├── fold_0_best_model.pt
│   ├── fold_1_best_model.pt
│   ├── fold_N_best_model.pt
│   └── ensemble_config.json
├── eu_stocks/
│   └── ...
├── commodities/
│   └── ...
└── crypto/
    └── ...
```

### 3.6 Ensemble (Opcional)

Combinar múltiples folds para mejorar robustez:

```python
# Ensemble de walk-forward folds
predictions = []
for fold in folds:
    model = load_model(f"fold_{fold}_best_model.pt")
    pred = model.predict(data)
    predictions.append(pred)

# Promedio ponderado
final_prediction = weighted_average(predictions, weights=fold_performances)
```

---

## 4. FASE 3: Generación de Señales y Validación con Benchmarks

### 4.1 Objetivo

Generar **señales de trading (LONG y SHORT) de alta confianza** basadas en múltiples condiciones probabilísticas, y **validar** su efectividad usando benchmarks independientes (índices de mercado).

**Dos tipos de señales**:
- **LONG**: Señales de compra (predicción de subida)
- **SHORT**: Señales de venta (predicción de bajada)

**Dos tipos de métricas**:
- **Métricas de Entrenamiento**: Evaluadas en tickers de tuning
- **Métricas de Benchmark**: Evaluadas en índices independientes (NO usados en tuning)

### 4.2 Arquitectura de Señales

```
Modelo Especializado por Mercado
              ↓
        Predicción
              ↓
      ┌───────────────┐
      │ Dirección:    │
      │ LONG o SHORT  │
      └───────────────┘
              ↓
┌─────────────────────────────────────────────────────┐
│  CONDICIÓN 1: Probabilidad Día                      │
│  • LONG: P(subida_mañana) > 90%                     │
│  • SHORT: P(bajada_mañana) > 90%                    │
└─────────────────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────────────┐
│  CONDICIÓN 2: Probabilidad Horizonte                │
│  • LONG: P(subida_horizonte) > 90%                  │
│  • SHORT: P(bajada_horizonte) > 90%                 │
│  Horizontes: 1w, 2w, 1m, 2m                         │
└─────────────────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────────────┐
│  CONDICIÓN 3: Rango Esperado                        │
│  • LONG: Expected_return > +2σ Bollinger            │
│  • SHORT: Expected_return < -2σ Bollinger           │
│  Calculado sobre cada horizonte                     │
└─────────────────────────────────────────────────────┘
              ↓
        SEÑAL GENERADA ✓
              ↓
┌─────────────────────────────────────────────────────┐
│  EVALUACIÓN:                                         │
│                                                      │
│  A. Métricas de Entrenamiento (tickers tuning):     │
│     • Tasa de acierto general                       │
│     • Precision LONG (% señales LONG correctas)     │
│     • Precision SHORT (% señales SHORT correctas)   │
│                                                      │
│  B. Métricas de Benchmark (índices mercado):        │
│     • Win rate (% operaciones ganadoras)            │
│     • Profit factor (ganancia/pérdida)              │
│     • Sharpe ratio (retorno/riesgo)                 │
│     • Max drawdown (pérdida máxima)                 │
└─────────────────────────────────────────────────────┘
```

### 4.3 Condiciones Detalladas

#### 4.3.1 Condición 1: Probabilidad Próximo Día (LONG o SHORT)

```python
# Predicción del modelo
logits_next_day = model.predict(current_ohlc, horizon=1)

# Convertir a probabilidades
prob_up = softmax(logits_next_day)[UP_CLASS]
prob_down = softmax(logits_next_day)[DOWN_CLASS]

# Criterio LONG
if prob_up > 0.90:
    signal_direction = "LONG"
    condition_1 = True

# Criterio SHORT
elif prob_down > 0.90:
    signal_direction = "SHORT"
    condition_1 = True
else:
    condition_1 = False
```

**Interpretación**:
- **LONG**: El modelo está >90% seguro de que el precio subirá mañana
- **SHORT**: El modelo está >90% seguro de que el precio bajará mañana

#### 4.3.2 Condición 2: Probabilidad en Horizonte (LONG o SHORT)

```python
# Horizontes temporales
HORIZONS = {
    '1week': 5,    # días de trading
    '2weeks': 10,
    '1month': 22,
    '2months': 44
}

# Para cada horizonte
for horizon_name, days in HORIZONS.items():
    # Predicción multi-step
    logits_horizon = model.predict(current_ohlc, horizon=days)

    # Probabilidades de subida/bajada
    prob_up_horizon = calculate_prob_positive_return(logits_horizon)
    prob_down_horizon = calculate_prob_negative_return(logits_horizon)

    # Criterio LONG
    if signal_direction == "LONG" and prob_up_horizon > 0.90:
        condition_2[horizon_name] = True

    # Criterio SHORT
    elif signal_direction == "SHORT" and prob_down_horizon > 0.90:
        condition_2[horizon_name] = True
    else:
        condition_2[horizon_name] = False
```

**Interpretación**:
- **LONG**: El modelo está >90% seguro de que el precio será mayor al final del horizonte
- **SHORT**: El modelo está >90% seguro de que el precio será menor al final del horizonte

#### 4.3.3 Condición 3: Rango Esperado vs Bollinger Bands (LONG o SHORT)

```python
# Calcular Bollinger Bands sobre el horizonte
bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(
    prices=historical_prices,
    window=horizon_days,
    num_std=2.0
)

# Banda superior = media + 2σ
# Banda inferior = media - 2σ
sigma = (bb_upper - bb_middle) / 2.0

# Predicción de precio esperado
expected_price = model.predict_price(current_ohlc, horizon=horizon_days)
current_price = current_ohlc[-1]['Close']

# Rango esperado (puede ser positivo o negativo)
expected_return = (expected_price - current_price) / current_price

# Criterio LONG
if signal_direction == "LONG" and expected_return > 2 * sigma:
    condition_3 = True

# Criterio SHORT
elif signal_direction == "SHORT" and expected_return < -2 * sigma:
    condition_3 = True
else:
    condition_3 = False
```

**Interpretación**:
- **LONG**: El rango de subida esperado supera +2σ (movimiento significativo alcista)
- **SHORT**: El rango de bajada esperado supera -2σ (movimiento significativo bajista)

### 4.4 Tipos de Señales

Cada horizonte temporal genera un **tipo de señal** diferente, con dos direcciones posibles:

| Tipo | Horizonte | Días | Objetivo | Dirección | Uso |
|------|-----------|------|----------|-----------|-----|
| **Señal Corto** | 1 semana | 5 | ±2σ en 5 días | LONG/SHORT | Day trading, scalping |
| **Señal Medio-Corto** | 2 semanas | 10 | ±2σ en 10 días | LONG/SHORT | Swing trading |
| **Señal Medio** | 1 mes | 22 | ±2σ en 22 días | LONG/SHORT | Position trading |
| **Señal Medio-Largo** | 2 meses | 44 | ±2σ en 44 días | LONG/SHORT | Investment, largo plazo |

**Total**: 4 horizontes × 2 direcciones = **8 tipos de señales posibles** por asset

### 4.5 Generación de Señal Completa

```python
def generate_trading_signal(model, asset, current_ohlc, horizon_name):
    """
    Genera señal de trading (LONG o SHORT) si se cumplen TODAS las condiciones.

    Returns:
        signal: dict con información de la señal o None
    """
    horizon_days = HORIZONS[horizon_name]

    # CONDICIÓN 1: Determinar dirección (LONG o SHORT)
    prob_up_next_day = model.predict_prob_up(current_ohlc, horizon=1)
    prob_down_next_day = model.predict_prob_down(current_ohlc, horizon=1)

    if prob_up_next_day > 0.90:
        signal_direction = "LONG"
        prob_next_day = prob_up_next_day
    elif prob_down_next_day > 0.90:
        signal_direction = "SHORT"
        prob_next_day = prob_down_next_day
    else:
        return None  # No cumple condición 1

    # CONDICIÓN 2: Prob horizonte > 90%
    if signal_direction == "LONG":
        prob_horizon = model.predict_prob_up(current_ohlc, horizon=horizon_days)
    else:  # SHORT
        prob_horizon = model.predict_prob_down(current_ohlc, horizon=horizon_days)

    if prob_horizon < 0.90:
        return None  # No cumple condición 2

    # CONDICIÓN 3: Rango > 2σ Bollinger
    bb_sigma = calculate_bollinger_sigma(asset.history, horizon_days)
    expected_return = model.predict_return(current_ohlc, horizon=horizon_days)

    if signal_direction == "LONG" and expected_return < 2 * bb_sigma:
        return None  # No cumple condición 3 para LONG
    elif signal_direction == "SHORT" and expected_return > -2 * bb_sigma:
        return None  # No cumple condición 3 para SHORT

    # ✅ TODAS las condiciones cumplidas
    signal = {
        'asset': asset.symbol,
        'timestamp': current_time,
        'direction': signal_direction,  # "LONG" o "SHORT"
        'type': horizon_name,
        'horizon_days': horizon_days,
        'conditions': {
            'prob_next_day': prob_next_day,
            'prob_horizon': prob_horizon,
            'expected_return': expected_return,
            'bollinger_sigma': bb_sigma,
            'threshold': 2 * bb_sigma if signal_direction == "LONG" else -2 * bb_sigma
        },
        'current_price': current_ohlc[-1]['Close'],
        'expected_price': current_ohlc[-1]['Close'] * (1 + expected_return),
        'expected_return_pct': expected_return * 100,
        'confidence': min(prob_next_day, prob_horizon)  # Confianza mínima
    }

    return signal
```

### 4.6 Output de Señales

**Ejemplo LONG**:

```json
{
  "signal_id": "AAPL_2025-01-15_1month_LONG",
  "asset": "AAPL",
  "market": "us_stocks",
  "direction": "LONG",
  "timestamp": "2025-01-15T14:30:00Z",
  "signal_type": "1month",
  "horizon_days": 22,
  "conditions": {
    "prob_next_day": 0.94,
    "prob_horizon": 0.92,
    "expected_return": 0.085,
    "bollinger_sigma": 0.035,
    "threshold": 0.070
  },
  "current_price": 150.25,
  "expected_price": 163.52,
  "expected_return_pct": 8.5,
  "confidence": 0.92,
  "model_version": "us_stocks_v1.2_fold_5"
}
```

**Ejemplo SHORT**:

```json
{
  "signal_id": "TSLA_2025-01-15_2weeks_SHORT",
  "asset": "TSLA",
  "market": "us_stocks",
  "direction": "SHORT",
  "timestamp": "2025-01-15T14:30:00Z",
  "signal_type": "2weeks",
  "horizon_days": 10,
  "conditions": {
    "prob_next_day": 0.91,
    "prob_horizon": 0.93,
    "expected_return": -0.062,
    "bollinger_sigma": 0.025,
    "threshold": -0.050
  },
  "current_price": 245.80,
  "expected_price": 230.55,
  "expected_return_pct": -6.2,
  "confidence": 0.91,
  "model_version": "us_stocks_v1.2_fold_5"
}
```

### 4.7 Métricas de Evaluación

El sistema utiliza **dos tipos de métricas** para evaluar el rendimiento:

#### 4.7.1 Métricas de Entrenamiento (sobre tickers de tuning)

Evaluadas sobre los activos específicos usados durante el fine-tuning:

```python
# Métricas generales
- Tasa de acierto: % de señales que resultaron correctas
- Frecuencia de señales: Número de señales generadas por periodo

# Métricas por dirección
- Precision LONG: % de señales LONG que resultaron en ganancia
- Precision SHORT: % de señales SHORT que resultaron en ganancia
- Recall LONG: % de oportunidades LONG capturadas
- Recall SHORT: % de oportunidades SHORT capturadas
```

**Ejemplo**:
```
Tickers de tuning: AAPL, MSFT, GOOGL, TSLA, AMZN (US Stocks)

Métricas:
- Tasa de acierto: 72.5%
- Precision LONG: 75.2%
- Precision SHORT: 68.8%
- Frecuencia: 12 señales/semana
```

#### 4.7.2 Métricas de Benchmark (sobre índices de mercado)

Evaluadas sobre **índices independientes** (NO usados en tuning) mediante backtesting:

```python
# Métricas de trading real
- Win rate: % de operaciones ganadoras
- Profit factor: Ganancia total / Pérdida total
- Sharpe ratio: Retorno ajustado por riesgo
- Max drawdown: Pérdida máxima desde peak
- Average return per trade: Retorno promedio por operación
- Total return: Retorno acumulado
```

**Ejemplo**:
```
Benchmark: SPY (S&P 500 ETF) - NO usado en tuning

Métricas:
- Win rate: 68.3%
- Profit factor: 2.14
- Sharpe ratio: 1.85
- Max drawdown: -12.5%
- Avg return/trade: +3.2%
- Total return: +45.7% (12 meses)
```

#### 4.7.3 Benchmarks por Mercado

| Mercado | Tickers de Tuning (ejemplos) | Benchmarks de Validación |
|---------|------------------------------|--------------------------|
| **US Stocks** | AAPL, MSFT, GOOGL, TSLA, NVDA | SPY, QQQ, DIA |
| **EU Stocks** | SAP.DE, SAN.MC, RR.L, AIR.PA | EWU, EWG, EWQ |
| **Commodities** | GC=F, CL=F, SI=F, HG=F | GLD, USO, SLV |
| **Crypto** | BTC-USD, ETH-USD, SOL-USD | BTC (índice), ETH (índice) |

**Importante**: Los benchmarks son activos DIFERENTES a los usados en tuning, lo que garantiza una validación objetiva e independiente.

---

## 5. Prevención de Look-Forward Bias

### 5.1 Principios Clave

1. **Datos históricos solo hasta T**: Nunca usar datos futuros a la fecha de decisión
2. **Walk-forward obligatorio**: Siempre validar en datos out-of-sample
3. **Re-entrenamiento periódico**: Actualizar modelos con nuevos datos
4. **Ventanas móviles**: Simular decisiones en tiempo real

### 5.2 Checkpoints Anti-Bias

```
✓ División temporal (no aleatoria)
✓ Test set solo se usa UNA VEZ al final
✓ Walk-forward con ventanas no solapadas en validación
✓ Bollinger Bands calculadas solo con datos históricos
✓ Re-fitting de tokenizer solo en train set
✓ No usar información futura en features
✓ Timestamps estrictos en backtesting
```

### 5.3 Validación de Señales

```python
def validate_no_lookahead_bias(signal, historical_data):
    """
    Valida que la señal no use información futura.
    """
    signal_timestamp = signal['timestamp']

    # Check 1: Datos usados no son futuros
    assert all(candle.timestamp <= signal_timestamp
               for candle in signal['input_data'])

    # Check 2: Bollinger calculadas solo con pasado
    bb_data = signal['bollinger_data']
    assert bb_data['last_timestamp'] <= signal_timestamp

    # Check 3: Modelo entrenado solo con pasado
    model_train_end = signal['model_metadata']['train_end_date']
    assert model_train_end <= signal_timestamp

    return True
```

---

## 6. Pipeline Completo de Producción

### 6.1 Workflow Diario

```
1. Recopilar datos actualizados (EOD o intraday)
   ↓
2. Preprocesar y tokenizar
   ↓
3. Cargar modelos especializados por mercado
   ↓
4. Para cada asset en cada mercado:
   ├── Calcular probabilidades (1 día + horizontes)
   ├── Calcular Bollinger Bands
   └── Evaluar condiciones de señal
   ↓
5. Filtrar señales válidas (todas condiciones = True)
   ↓
6. Rankear por confianza
   ↓
7. Generar reporte de señales
   ↓
8. Almacenar en base de datos
   ↓
9. Notificar señales de alta prioridad
```

### 6.2 Re-entrenamiento Periódico

```
Frecuencia: Mensual o trimestral

1. Descargar datos actualizados
2. Extender ventana de walk-forward
3. Fine-tune modelos con nuevos datos
4. Validar performance out-of-sample
5. Si performance > threshold:
   └── Deploy nuevo modelo
   Else:
   └── Mantener modelo actual
6. Archivar modelos antiguos (versionado)
```

---

## 7. Métricas y Monitoreo

### 7.1 Métricas de Modelo

- **Accuracy**: % predicciones correctas
- **Precision**: De las señales generadas, % correctas
- **Recall**: De las oportunidades reales, % capturadas
- **F1-Score**: Balance precision/recall
- **Sharpe Ratio**: Return vs riesgo
- **Max Drawdown**: Pérdida máxima

### 7.2 Métricas de Señales

- **Signal Win Rate**: % señales que resultaron en ganancia
- **Average Return per Signal**: Retorno promedio
- **Signal Frequency**: Señales generadas por día/semana
- **False Positive Rate**: Señales incorrectas
- **Time to Target**: Tiempo promedio hasta alcanzar objetivo

### 7.3 Dashboard de Monitoreo

```
- Performance actual vs histórico
- Degradación de modelo (drift detection)
- Distribución de señales por mercado
- Heat map de oportunidades
- Backtesting continuo
- Alertas de anomalías
```

---

## 8. FASE 4: Optimización y Detección de Deterioro (Futuro)

### 8.1 Objetivo

Monitorear continuamente el rendimiento de los modelos en benchmarks y detectar **deterioro temporal** de la estrategia. Cuando el rendimiento cae por debajo de umbrales definidos, activar re-entrenamiento automático.

### 8.2 Optimización de Parámetros de Benchmark

#### 8.2.1 Parámetros Optimizables

```python
# Parámetros del sistema de señales
PROBABILITY_THRESHOLD = 0.90      # Umbral de probabilidad
BOLLINGER_MULTIPLIER = 2.0        # Multiplicador de σ Bollinger
HORIZONS = [5, 10, 22, 44]        # Horizontes en días

# Parámetros de gestión de riesgo
STOP_LOSS = 0.02                  # -2% stop loss
TAKE_PROFIT = 0.10                # +10% take profit
MAX_POSITION_SIZE = 0.05          # 5% del capital
```

#### 8.2.2 Proceso de Optimización

```
1. Definir rango de valores para cada parámetro
2. Ejecutar backtesting con diferentes combinaciones
3. Evaluar métricas en benchmarks:
   • Win rate
   • Profit factor
   • Sharpe ratio
   • Max drawdown
4. Seleccionar combinación óptima
5. Validar en periodo out-of-sample
6. Desplegar si mejora > threshold
```

### 8.3 Detección de Deterioro Temporal

#### 8.3.1 Métricas de Deterioro

El sistema monitorea continuamente:

```python
# Ventana de evaluación
EVALUATION_WINDOW = 30  # días

# Métricas críticas
critical_metrics = {
    'win_rate': {
        'current': calculate_win_rate(last_30_days),
        'baseline': baseline_win_rate,
        'threshold': 0.85,  # 85% del baseline
        'status': 'OK' if current >= baseline * threshold else 'DETERIORO'
    },
    'sharpe_ratio': {
        'current': calculate_sharpe(last_30_days),
        'baseline': baseline_sharpe,
        'threshold': 0.80,  # 80% del baseline
        'status': 'OK' if current >= baseline * threshold else 'DETERIORO'
    },
    'profit_factor': {
        'current': calculate_profit_factor(last_30_days),
        'baseline': baseline_profit_factor,
        'threshold': 0.75,  # 75% del baseline
        'status': 'OK' if current >= baseline * threshold else 'DETERIORO'
    }
}
```

#### 8.3.2 Causas Comunes de Deterioro

1. **Cambio de régimen de mercado**
   - Transición bull → bear o viceversa
   - Aumento de volatilidad estructural
   - Cambios en correlaciones entre activos

2. **Drift de datos**
   - Distribución de precios cambia con el tiempo
   - Nuevos patrones no capturados en entrenamiento

3. **Sobreajuste temporal**
   - Modelo optimizado para periodo específico
   - No generaliza a condiciones actuales

4. **Eventos exógenos**
   - Cambios regulatorios
   - Crisis económicas
   - Shocks geopolíticos

#### 8.3.3 Sistema de Alertas

```python
def monitor_deterioration(benchmark_results, baseline_metrics):
    """
    Monitorea deterioro y genera alertas.
    """
    alerts = []

    # Calcular métricas actuales
    current_metrics = calculate_metrics(benchmark_results[-30:])

    # Comparar con baseline
    for metric_name, baseline_value in baseline_metrics.items():
        current_value = current_metrics[metric_name]
        threshold = DETERIORATION_THRESHOLDS[metric_name]

        ratio = current_value / baseline_value

        if ratio < threshold:
            alert = {
                'severity': 'HIGH' if ratio < 0.7 else 'MEDIUM',
                'metric': metric_name,
                'current': current_value,
                'baseline': baseline_value,
                'deterioration_pct': (1 - ratio) * 100,
                'action': 'RETRAIN' if ratio < 0.7 else 'MONITOR',
                'timestamp': datetime.now()
            }
            alerts.append(alert)

    return alerts
```

### 8.4 Re-entrenamiento Automático

#### 8.4.1 Criterios de Activación

El re-entrenamiento se activa cuando:

```python
# Condiciones de re-entrenamiento
retrain_conditions = {
    # Condición 1: Deterioro severo en múltiples métricas
    'multiple_deterioration': (
        num_metrics_below_threshold >= 2 and
        worst_deterioration > 0.30  # >30% caída
    ),

    # Condición 2: Deterioro prolongado
    'sustained_deterioration': (
        days_below_threshold >= 14 and
        deterioration_trend == 'WORSENING'
    ),

    # Condición 3: Evento específico
    'specific_event': (
        profit_factor < 1.0 or  # Perdiendo dinero
        max_drawdown > 0.25     # >25% drawdown
    )
}

# Activar si CUALQUIERA se cumple
trigger_retrain = any(retrain_conditions.values())
```

#### 8.4.2 Pipeline de Re-entrenamiento

```
1. DETECCIÓN
   └── Monitor detecta deterioro > threshold

2. DIAGNÓSTICO
   ├── Analizar métricas detalladas
   ├── Identificar causa raíz
   └── Determinar acción (retrain, ajuste parámetros, etc.)

3. RE-ENTRENAMIENTO
   ├── Extender datos con periodo reciente
   ├── Actualizar ventanas de walk-forward
   ├── Fine-tune modelos con nuevos datos
   └── Validar en out-of-sample

4. VALIDACIÓN
   ├── Backtest en benchmark
   ├── Comparar métricas: nuevo vs actual
   └── Decidir deployment

5. DEPLOYMENT
   ├── Si nuevo modelo > actual: deploy
   ├── Si nuevo modelo < actual: mantener actual + investigar
   └── Archivar versión anterior

6. MONITOREO POST-DEPLOYMENT
   └── Seguimiento intensivo primeros 7 días
```

#### 8.4.3 Dashboard de Deterioro

```
┌─────────────────────────────────────────────────────────┐
│           SISTEMA DE MONITOREO DE DETERIORO             │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Benchmark: SPY (S&P 500)                              │
│  Periodo: Últimos 30 días                              │
│                                                         │
│  ┌─────────────────────────────────────────┐           │
│  │ Métrica         │ Current │ Baseline │ Status │     │
│  ├─────────────────┼─────────┼──────────┼────────┤     │
│  │ Win Rate        │  62.3%  │  68.3%   │  ⚠️     │     │
│  │ Profit Factor   │  1.85   │  2.14    │  ⚠️     │     │
│  │ Sharpe Ratio    │  1.52   │  1.85    │  ⚠️     │     │
│  │ Max Drawdown    │ -15.2%  │ -12.5%   │  ❌     │     │
│  └─────────────────────────────────────────┘           │
│                                                         │
│  ⚠️  ALERTA: Deterioro detectado en 4 métricas         │
│  📊 Deterioro promedio: -18.7%                          │
│  📅 Días bajo threshold: 12 días                        │
│                                                         │
│  🔧 ACCIÓN RECOMENDADA: Monitoreo intensivo             │
│     Si persiste >14 días → Re-entrenamiento             │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 8.5 Registro de Deterioro y Re-entrenamientos

```json
{
  "deterioration_log": {
    "event_id": "DETERIORO_2025-01-15",
    "detection_date": "2025-01-15",
    "benchmark": "SPY",
    "model_version": "us_stocks_v1.2_fold_5",
    "metrics": {
      "win_rate": {"current": 0.623, "baseline": 0.683, "deterioration": 0.088},
      "sharpe": {"current": 1.52, "baseline": 1.85, "deterioration": 0.178}
    },
    "cause": "Market regime change (bull to sideways)",
    "action_taken": "Retrain triggered",
    "retrain_id": "RETRAIN_2025-01-20",
    "outcome": "New model deployed with improved metrics"
  }
}
```

---

## 9. Consideraciones de Riesgo

### 9.1 Limitaciones

- **Probabilidad ≠ Certeza**: 90% NO es 100%
- **Black Swan Events**: Eventos imprevistos no capturados
- **Cambios de régimen**: Mercados pueden cambiar fundamentalmente
- **Correlaciones dinámicas**: Relaciones entre assets cambian

### 9.2 Gestión de Riesgo

```python
# Reglas de gestión de riesgo
MAX_POSITIONS = 10  # Máximo posiciones simultáneas
MAX_ALLOCATION_PER_SIGNAL = 0.05  # 5% del capital por señal
STOP_LOSS = 0.02  # -2% stop loss
TAKE_PROFIT = 0.10  # +10% take profit (ajustar según horizonte)

# Diversificación
MAX_PER_MARKET = 0.30  # Máximo 30% en un mercado
MAX_PER_ASSET = 0.10   # Máximo 10% en un asset
```

---

## 10. Resumen Ejecutivo

### 10.1 Fases del Sistema

| Fase | Nombre | Objetivo | Estado | Output |
|------|--------|----------|--------|--------|
| **1** | Pre-entrenamiento | 3 réplicas de LLM con datos OHLC | ✅ COMPLETADO | 3 × `best_model.pt` (A, B, C) |
| **2** | Fine-tuning por mercado | Especialización con walk-forward | ❌ PENDIENTE | 4 modelos × N folds por case |
| **3** | Señales + Benchmarks | Trading signals LONG/SHORT + validación | ❌ PENDIENTE | Señales alta confianza + métricas |
| **4** | Optimización + Deterioro | Monitoreo y re-entrenamiento | ❌ FUTURO | Sistema adaptativo |

### 10.2 Condiciones de Señal (LONG/SHORT)

✅ **TODAS deben cumplirse**:
1. **LONG**: P(subida_mañana) > 90% | **SHORT**: P(bajada_mañana) > 90%
2. **LONG**: P(subida_horizonte) > 90% | **SHORT**: P(bajada_horizonte) > 90%
3. **LONG**: Rango > +2σ Bollinger | **SHORT**: Rango < -2σ Bollinger

### 10.3 Tipos de Señales

Cada horizonte admite 2 direcciones (LONG/SHORT):

- **Corto** (1 semana): Day/swing trading
- **Medio-Corto** (2 semanas): Swing trading
- **Medio** (1 mes): Position trading
- **Medio-Largo** (2 meses): Investment

**Total**: 4 horizontes × 2 direcciones = 8 tipos de señales

### 10.4 Métricas Clave

**Entrenamiento** (sobre tickers de tuning):
- Tasa de acierto, Precision LONG, Precision SHORT

**Benchmark** (sobre índices independientes):
- Win rate, Profit factor, Sharpe ratio, Max drawdown

---

## 11. Próximos Pasos

### 11.1 Fase 2 - Fine-Tuning (PENDIENTE)

- [ ] Scripts de fine-tuning por mercado (US, EU, Commodities, Crypto)
- [ ] Pipeline de walk-forward analysis
- [ ] Definir tickers específicos para tuning por mercado
- [ ] Sistema de multi-fold management
- [ ] Ensemble de folds

### 11.2 Fase 3 - Señales y Benchmarks (PENDIENTE)

- [ ] Motor de generación de señales LONG/SHORT
- [ ] Cálculo de Bollinger Bands multi-horizonte
- [ ] Sistema de predicción de probabilidades por horizonte
- [ ] Definir benchmarks por mercado (índices independientes)
- [ ] Sistema de backtesting sobre benchmarks
- [ ] Métricas de entrenamiento (tasa acierto, precision LONG/SHORT)
- [ ] Métricas de benchmark (win rate, profit factor, sharpe, drawdown)
- [ ] Sistema de logging de señales
- [ ] API de señales en tiempo real

### 11.3 Fase 4 - Optimización y Deterioro (FUTURO)

- [ ] Sistema de optimización de parámetros de benchmark
- [ ] Monitor de deterioro temporal
- [ ] Dashboard de métricas de deterioro
- [ ] Sistema de re-entrenamiento automático
- [ ] Alertas de deterioro
- [ ] Registro de eventos de deterioro y re-entrenamientos

### 11.4 Mejoras Generales

- [ ] Multi-model ensemble (combinación de Cases A, B, C)
- [ ] Incorporar sentiment analysis
- [ ] Incorporar indicadores técnicos adicionales
- [ ] Reinforcement learning para timing óptimo
- [ ] Dashboard de monitoreo general

---

**Versión**: 2.0
**Fecha**: 2025-01-06
**Autor**: Sistema MarketGPT
**Status**: Arquitectura completa con 4 fases (Pre-entrenamiento, Fine-tuning, Señales/Benchmarks, Deterioro)
**Última actualización**: Añadidas señales LONG/SHORT, benchmarks independientes, y Fase 4 de deterioro


---

## 12. APÉNDICE: Multi-Task Architecture Approach

> **Nota**: Este es un enfoque alternativo experimental para predicción multi-tarea.

### 12.1 Overview

El enfoque multi-task reemplaza la predicción de tokens OHLC exactos por tareas de clasificación más simples:

**Enfoque original (token prediction)**:
- Predice tokens OHLC exactos (2048 classes)
- Complejo para datos limitados
- No alineado directamente con objetivos de trading

**Enfoque multi-task (experimental)**:
- Predice **dirección** de siguiente vela (3 clases: DOWN/FLAT/UP)
- Predice **magnitud** en múltiples horizontes (4 tareas binarias)
- Alineado con sistema de trading

### 12.2 Architecture Details

#### Task 1: Direction Prediction (3 classes)

Predice la dirección de la **próxima vela**:

- **DOWN (0)**: Precio cierra más de threshold% abajo (default 0.5%)
- **FLAT (1)**: Cambio de precio dentro de ±threshold%
- **UP (2)**: Precio cierra más de threshold% arriba

**Propósito**: Generar señales de entrada con el movimiento del mercado.

#### Task 2: Magnitude Prediction (4 binary tasks)

Para cada horizonte, predice si precio excederá **2σ Bollinger bands**:

| Horizonte | Trading Days | Clasificación Binaria |
|-----------|--------------|----------------------|
| 1 semana  | 5 días       | ¿Precio excede 2σ? (0/1) |
| 2 semanas | 10 días      | ¿Precio excede 2σ? (0/1) |
| 1 mes | 20 días      | ¿Precio excede 2σ? (0/1) |
| 2 meses | 40 días      | ¿Precio excede 2σ? (0/1) |

**Propósito**: Identificar "buenos movimientos" - cambios de precio significativos que exceden 2 desviaciones estándar.

### 12.3 Model Architecture

**Input**:
- Secuencias OHLC: `(batch, seq_len, 4)`
- Sequence length default: 128 velas

**Transformer Encoder**:
- Layers: 4
- d_model: 128
- Heads: 4
- Feed-forward: 512
- Parameters: ~811K (vs 12M en enfoque original)

**Output Heads**:

1. **Direction Head**
   - Output: `(batch, 3)` logits
   - Loss: CrossEntropyLoss
   - Metric: Classification accuracy

2. **Magnitude Head**
   - Output: `(batch, 4)` logits (uno por horizonte)
   - Loss: BCEWithLogitsLoss
   - Metric: Binary accuracy por horizonte

**Combined Loss**:
```
Total Loss = w_direction × Direction Loss + w_magnitude × Magnitude Loss
```

### 12.4 Key Advantages

1. **Tarea más simple**:
   - Antes: 2,048 clases OHLC
   - Ahora: 3 direcciones + 4 binarias = 7 outputs totales
   - Mucho más fácil de aprender con datos limitados

2. **Alineado con objetivos de trading**:
   - Predicción de dirección → Señales de entrada
   - Predicción de magnitud → Señales de calidad (threshold 2σ)
   - Directamente usable para decisiones LONG/SHORT

3. **Mejor eficiencia de datos**:
   - Anterior: 481 sequences : 12M params = 1:24,900 ratio
   - Actual: 481 sequences : 811K params = 1:1,685 ratio
   - ~15× mejor ratio datos-a-parámetros

4. **Outputs interpretables**:
   - Dirección: "¿Debería ir LONG o SHORT?"
   - Magnitud: "¿Será un movimiento significativo worth trading?"
   - Fácil de integrar en lógica de trading

### 12.5 Integration with Trading System

**Direction Signal** (3 clases):
```python
if direction == UP:
    consider_LONG()
elif direction == DOWN:
    consider_SHORT()
else:
    stay_neutral()
```

**Magnitude Signals** (4 binarias por horizonte):
```python
if magnitude_1w == 1 and magnitude_1m == 1:
    # Movimiento significativo esperado en corto y largo plazo
    high_confidence_trade()
elif sum(magnitude) >= 2:
    # Movimiento esperado en al menos 2 horizontes
    moderate_confidence_trade()
else:
    # Movimiento limitado esperado
    skip_trade()
```

**Estrategia combinada**:
```python
# Criterio para trade de alta calidad:
if direction == UP and sum(magnitude) >= 3:
    # Fuerte movimiento alcista esperado en múltiples horizontes
    enter_LONG_with_confidence()

elif direction == DOWN and sum(magnitude) >= 3:
    # Fuerte movimiento bajista esperado en múltiples horizontes
    enter_SHORT_with_confidence()
```

### 12.6 Status

**Estado**: Enfoque experimental
**Implementación**: Archivos en `common/multitask_*.py`
**Testing**: Pendiente validación en producción
**Comparación**: Requiere benchmarking vs enfoque token-based

---

**Fin de ARCHITECTURE.md**

