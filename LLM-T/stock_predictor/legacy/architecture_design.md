# Arquitectura Market GPT - Predictor de Series Temporales Financieras

## Resumen
Modelo tipo GPT optimizado para predicción de precios de mercado usando velas de 1 minuto.
Inspirado en GPT-2 con adaptaciones específicas para series temporales financieras.

## Especificaciones del Modelo

### Tamaño objetivo: ~100M parámetros (GPT-2 small scale)

**Configuración de hiperparámetros:**
```
n_layers = 12              # Capas Transformer
d_model = 768              # Dimensión del modelo
n_heads = 12               # Cabezas de atención
d_ff = 3072                # Dimensión feedforward (4x d_model)
context_length = 2048      # Ventana de contexto (2048 velas = ~34 horas)
vocab_size = 1024          # Tamaño del vocabulario para tokenización
dropout = 0.1              # Dropout rate
```

**Cálculo de parámetros:**
- Embedding: vocab_size * d_model = 1024 * 768 = 786K
- Positional Encoding: context_length * d_model = 2048 * 768 = 1.5M
- Transformer Layers (x12):
  - Self-Attention: 4 * d_model^2 * n_layers = 4 * 768^2 * 12 ≈ 28.3M
  - FFN: 2 * d_model * d_ff * n_layers = 2 * 768 * 3072 * 12 ≈ 56.6M
  - LayerNorm + otros: ~2M
- Output Head: d_model * output_dim ≈ 3K

**Total estimado: ~87M parámetros** (cercano a objetivo de 100M)

## Arquitectura

### 1. Input Layer (Tokenización + Embedding)

**Problema:** Series temporales continuas vs tokens discretos
**Solución:** Discretización adaptativa multi-resolución

```
Input: OHLC de velas de 1 minuto
- Open, High, Low, Close (4 valores continuos)
- Normalización relativa (retornos porcentuales)

Tokenización:
1. Conversión a retornos logarítmicos relativos
2. Discretización usando k-means clustering (vocab_size bins)
3. Embedding separado para cada componente OHLC
4. Proyección final: concat(emb_O, emb_H, emb_L, emb_C) → d_model
```

### 2. Positional Encoding

**Temporal Positional Encoding:**
- Sinusoidal encoding estándar para posición en secuencia
- Encoding adicional para tiempo absoluto (hora del día, día de semana)
- Aprendible para capturar patrones estacionales

### 3. Transformer Blocks (x12)

**Cada bloque contiene:**

```python
x = x + MultiHeadSelfAttention(LayerNorm(x))
x = x + FeedForward(LayerNorm(x))
```

**Multi-Head Self-Attention:**
- Atención causal (autoregresiva)
- 12 cabezas paralelas
- Scaled dot-product attention
- Dropout en atención y proyección

**Feed-Forward Network:**
- Linear(d_model → d_ff=3072) + GELU
- Linear(d_ff → d_model)
- Dropout

### 4. Output Heads

**Dual-mode prediction:**

a) **Next-Token Head** (predicción autoregresiva):
```python
logits = Linear(d_model → vocab_size * 4)  # 4 para OHLC
output = softmax(logits)
→ Próxima vela tokenizada
```

b) **Multi-Step Head** (predicción directa):
```python
projection = Linear(d_model → d_model)
predictions = Linear(d_model → n_steps * 4)  # n_steps velas futuras
→ Valores OHLC continuos para próximas n_steps velas
```

## Walk-Forward Training Strategy

### Filosofía
Actualizar el modelo continuamente con nuevos datos, simulando trading en tiempo real.

### Proceso:

```
1. Initial Training Phase
   ├─ Train: primeros 80% de datos históricos
   └─ Validation: siguiente 10%

2. Walk-Forward Loop:
   Para cada ventana temporal W:
   ├─ Predecir siguiente periodo usando modelo actual
   ├─ Esperar datos reales del periodo
   ├─ Fine-tune modelo con nuevos datos
   │  ├─ Pequeño learning rate (1/10 del original)
   │  ├─ Pocas epochs (1-5)
   │  └─ Congelación opcional de capas tempranas
   └─ Actualizar métricas y avanzar ventana

3. Test Final:
   └─ Oro 20 años, evaluación en hold-out completo
```

### Ventanas:
- **Initial training**: Todo histórico Bitcoin hasta T-60 días
- **Walk-forward window**: 7 días (testing) → fine-tune con esos datos → avanzar
- **Test final**: Últimos 20 años oro (hold-out completo)

## Data Pipeline

### Bitcoin Data (Training)
```python
Source: Binance API (datos públicos)
Timeframe: 1 minuto
Period: Máximo histórico disponible (~2017 a presente)
Features: OHLC (4 features)
Total candles: ~3-4 millones de velas
```

### Gold Data (Testing)
```python
Source: APIs financieras (Alpha Vantage, Yahoo Finance, o similar)
Timeframe: 1 minuto (si disponible, sino interpolar de 5min)
Period: 20 años
Features: OHLC (4 features)
Total candles: ~5-6 millones de velas
```

### Preprocessing:
1. Limpieza: eliminar gaps, velas inválidas
2. Normalización: retornos logarítmicos relativos
3. Tokenización: k-means clustering sobre retornos
4. Secuenciación: ventanas de context_length

## Training Configuration

### Hardware: CESGA A100
```yaml
GPU: 1x A100 (40GB)
Mixed Precision: FP16 (Automatic Mixed Precision)
Gradient Accumulation: 4 steps (effective batch size 32)
Gradient Checkpointing: Enabled (reduce memoria)
```

### Hyperparameters:
```yaml
Initial Training:
  batch_size: 8 (por GPU, x4 grad_accum = 32 effective)
  learning_rate: 6e-4
  warmup_steps: 2000
  max_steps: 100000
  weight_decay: 0.1
  grad_clip: 1.0

Walk-Forward Fine-tuning:
  batch_size: 8
  learning_rate: 1e-5 (1/60 del original)
  epochs: 3
  freeze_layers: primeras 6 capas
```

### Loss Functions:
```python
# Next-token mode: Cross-entropy
loss_next_token = CrossEntropy(predictions, targets)

# Multi-step mode: MSE sobre retornos
loss_multi_step = MSE(predictions, targets)

# Combined loss
total_loss = α * loss_next_token + (1-α) * loss_multi_step
# α = 0.7 (prioridad a next-token)
```

### Optimización:
- Optimizer: AdamW (β1=0.9, β2=0.95, ε=1e-8)
- LR Scheduler: Cosine annealing con warmup
- Regularización: Weight decay, dropout, label smoothing

## Evaluation Metrics

### Trading-specific metrics:
```python
1. Directional Accuracy: % de predicciones correctas de dirección
2. Sharpe Ratio: retorno ajustado por riesgo
3. Maximum Drawdown: mayor caída desde pico
4. Win Rate: % de trades ganadores
5. Profit Factor: ganancia total / pérdida total
```

### Statistical metrics:
```python
1. RMSE: root mean squared error
2. MAE: mean absolute error
3. MAPE: mean absolute percentage error
4. R²: coeficiente de determinación
```

## Implementation Roadmap

### Fase 1: Core Architecture (2-3 días)
- Implementar Transformer blocks
- Tokenización y embeddings
- Training loop básico

### Fase 2: Data Pipeline (1-2 días)
- Bitcoin data fetching (Binance API)
- Gold data fetching
- Preprocessing y caching

### Fase 3: Walk-Forward System (2-3 días)
- Walk-forward training loop
- Checkpointing y recuperación
- Métricas y logging

### Fase 4: SLURM Integration (1 día)
- Scripts SLURM para A100
- Monitoreo y alertas
- Resultados y análisis

### Fase 5: Testing & Optimization (ongoing)
- Backtesting en oro
- Tuning de hiperparámetros
- Análisis de resultados

## Consideraciones Técnicas

### Memoria GPU:
```
Modelo: ~400MB (100M params × 4 bytes)
Activaciones (FP16, batch=8, seq=2048): ~8GB
Gradientes: ~400MB
Optimizer states (AdamW): ~800MB
Total estimado: ~10GB (cómodo en A100 40GB)
```

### Tiempo de entrenamiento estimado:
```
Single A100 (40GB):
- Initial training: 3-5 días
- Walk-forward iteration: 1-2 horas por ventana
- Total proyecto: 1-2 semanas
```

### Optimizaciones disponibles:
- Flash Attention 2 (2-4x speedup)
- Gradient checkpointing (reduce memoria 30-40%)
- Mixed precision training (2x speedup)
- Compiled model (torch.compile, 20-30% speedup)

## Referencias de Diseño

1. **GPT-2**: Architecture base
2. **TimeGPT**: Temporal embedding strategies
3. **Lag-Llama**: Time series tokenization
4. **Financial LLMs**: FinBERT, BloombergGPT adaptations
