# MarketGPT - Referencia Rápida

## Flujo de Trabajo Completo

```
FASE 1: PRE-ENTRENAMIENTO (3 Réplicas de LLM)
├── Case A: 600 tickers multi-mercado
├── Case B: 100 tickers curated
├── Case C: 20 tickers crypto
├── Train (70%) / Val (15%) / Test (15%)
└── Output: 3 modelos base (best_model.pt por case)

         ↓

FASE 2: FINE-TUNING POR MERCADO
├── Dentro de cada case → modelos por mercado
├── US Stocks, EU Stocks, Commodities, Crypto
├── Walk-Forward Analysis (evita look-forward bias)
├── Tickers ESPECÍFICOS por mercado para tuning
├── Ventanas: Train 2 años, Val 6 meses, Step 3 meses
└── Output: 4 modelos especializados × N folds (por case)

         ↓

FASE 3: SEÑALES + VALIDACIÓN CON BENCHMARKS
├── Generación Señales LONG/SHORT
│   ├── Condición 1: P(subida/bajada_1día) > 90%
│   ├── Condición 2: P(subida/bajada_horizonte) > 90%
│   └── Condición 3: Rango > 2σ Bollinger
├── Benchmarks con ÍNDICES (SPY, QQQ, GLD, BTC)
│   └── Diferentes de tickers de tuning
├── Métricas Entrenamiento: Tasa acierto, Precision LONG/SHORT
└── Métricas Benchmark: Win rate, Profit factor, Sharpe, Drawdown

         ↓

FASE 4: OPTIMIZACIÓN Y MONITOREO (Futuro)
├── Ajuste de parámetros de benchmark
└── Detección de deterioro temporal
```

---

## Señales de Compra

### Condiciones (TODAS deben cumplirse)

1. **Probabilidad Día**
   - `P(precio_sube_mañana) > 90%`

2. **Probabilidad Horizonte**
   - `P(precio_sube_en_horizonte) > 90%`
   - Horizontes: 1 semana, 2 semanas, 1 mes, 2 meses

3. **Rango Esperado**
   - `expected_return > 2 × σ_Bollinger`
   - Bollinger Bands calculadas sobre cada horizonte

### Tipos de Señales

| Tipo | Horizonte | Días | Uso |
|------|-----------|------|-----|
| Corto | 1 semana | 5 | Day/Swing trading |
| Medio-Corto | 2 semanas | 10 | Swing trading |
| Medio | 1 mes | 22 | Position trading |
| Medio-Largo | 2 meses | 44 | Investment |

---

## Walk-Forward Analysis

```python
train_window = 24 meses  # Ventana de entrenamiento
val_window = 6 meses     # Ventana de validación
step = 3 meses           # Avance temporal

# Evita look-forward bias
# Simula trading real
# Múltiples validaciones
```

---

## Mercados Especializados

1. **US Stocks**: S&P 500, NASDAQ (~300 assets)
2. **EU Stocks**: FTSE, DAX, CAC (~150 assets)
3. **Commodities**: Metals, Energy, Agriculture (~30 assets)
4. **Crypto**: BTC, ETH, altcoins (~70 assets)

---

## Anti Look-Forward Bias

✓ División temporal (no aleatoria)
✓ Test set solo UNA vez al final
✓ Walk-forward con ventanas no solapadas
✓ Bollinger solo con datos históricos
✓ Re-fit tokenizer solo en train
✓ Sin información futura en features

---

## Métricas Clave

- **Win Rate**: % señales correctas
- **Avg Return**: Retorno promedio por señal
- **Sharpe Ratio**: Return/Risk
- **Max Drawdown**: Pérdida máxima
- **Signal Frequency**: Señales por periodo

---

## Estado Actual

✅ **Completado**:
- Pre-entrenamiento (Cases A, B, C)
- Train/Val/Test splits
- Evaluación en test set
- Multi-GPU training (2×A100)
- Documentación arquitectura

❌ **Pendiente**:
- Scripts fine-tuning por mercado
- Pipeline walk-forward
- Motor de señales
- Cálculo Bollinger Bands
- Sistema backtesting
- Dashboard monitoreo

---

## Comandos Útiles

```bash
# Entrenar Case A
cd case_a_full_market/slurm_scripts
sbatch train_full_a100.sh

# Entrenar Case B
cd case_b_reduced/slurm_scripts
sbatch train_reduced_a100.sh

# Entrenar Case C
cd case_c_crypto/slurm_scripts
sbatch train_crypto_a100.sh

# Monitorear
squeue -u $(whoami)
tail -f case_*/logs/*.out
```

---

**Consultar documentación completa**: `SYSTEM_ARCHITECTURE.md`
