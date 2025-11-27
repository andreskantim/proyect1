# MarketGPT - Sistema de Trading con IA

Sistema completo de trading basado en transformers para predicción de mercados financieros y generación de señales de compra/venta de alta confianza.

---

## 📚 Documentación

| Documento | Descripción |
|-----------|-------------|
| **[GETTING_STARTED.md](GETTING_STARTED.md)** | Guía de instalación y primeros pasos |
| **[ARCHITECTURE.md](ARCHITECTURE.md)** | Arquitectura completa del sistema |
| **[TRAINING.md](TRAINING.md)** | Guía de entrenamiento multi-GPU |
| **[REFERENCE.md](REFERENCE.md)** | Referencia rápida de conceptos clave |
| **[PROJECT_STATUS.md](PROJECT_STATUS.md)** | Estado actual e implementación |
| **[CLAUDE_CONTEXT.md](CLAUDE_CONTEXT.md)** | Contexto para Claude (comando `/context`) |

---

## 🎯 Sistema de 3 Fases

### FASE 1: Pre-entrenamiento General
Entrenar modelo base multi-mercado con múltiples activos

- **Case A**: 600 assets (US, EU, EM, Commodities, Crypto)
- **Case B**: 100 assets (curated multi-market)
- **Case C**: 20 cryptos (prototipo rápido)

→ Output: `best_model.pt` (modelo general)

### FASE 2: Fine-Tuning por Mercado
Especializar modelo para cada mercado usando Walk-Forward Analysis

- US Stocks
- EU Stocks
- Commodities
- Crypto

→ Output: 4 modelos especializados

### FASE 3: Generación de Señales
Señales de compra/venta cuando:
- P(movimiento_1día) > 90%
- P(movimiento_horizonte) > 90%
- Rango > 2σ Bollinger

→ Output: Señales LONG/SHORT para 4 horizontes (1w, 2w, 1m, 2m)

---

## 📁 Estructura del Proyecto

```
stock_predictor/
├── 📖 README.md                       # Este archivo
├── 📖 ARCHITECTURE.md                 # Arquitectura completa
├── 📖 GETTING_STARTED.md              # Guía de inicio
├── 📖 TRAINING.md                     # Guía de entrenamiento
├── 📖 REFERENCE.md                    # Referencia rápida
├── 📖 PROJECT_STATUS.md               # Estado del proyecto
│
├── 📂 common/                         # Módulos compartidos
│   ├── market_gpt.py                  # Modelo transformer base
│   ├── market_gpt_multi.py            # Modelo multi-asset
│   ├── tokenizer.py                   # Tokenizer OHLC
│   ├── training_monitor.py            # Monitor de entrenamiento
│   └── distributed_utils.py           # Utilidades multi-GPU
│
├── 📂 case_a_full_market/             # 600 assets
│   ├── universal_loader.py
│   ├── train_full.py
│   ├── configs/
│   ├── slurm_scripts/
│   └── README.md
│
├── 📂 case_b_reduced/                 # 100 assets
│   ├── multi_market_loader.py
│   ├── train_reduced.py
│   ├── configs/
│   ├── slurm_scripts/
│   └── README.md
│
├── 📂 case_c_crypto/                  # 20 cryptos
│   ├── crypto_data_loader.py
│   ├── train_crypto.py
│   ├── configs/
│   ├── slurm_scripts/
│   └── README.md
│
└── 📂 checkpoints/                    # Modelos entrenados
    ├── case_a_full_market/
    ├── case_b_reduced/
    └── case_c_crypto/
```

---

## 🚀 Quick Start

### 1. Instalación

```bash
# Activar entorno
conda activate llm-training

# Instalar dependencias
cd stock_predictor
pip install -r requirements_gpu.txt
```

### 2. Entrenamiento (Fase 1)

#### Case C - Crypto Prototype (rápido, 1-2 días)
```bash
cd case_c_crypto/slurm_scripts
sbatch train_crypto_a100.sh
```

#### Case B - Reduced Market (medio, 3-5 días)
```bash
cd case_b_reduced/slurm_scripts
sbatch train_reduced_a100.sh
```

#### Case A - Full Market (completo, 7-10 días)
```bash
cd case_a_full_market/slurm_scripts
sbatch train_full_a100.sh
```

### 3. Monitorear

```bash
# Ver jobs activos
squeue -u $(whoami)

# Ver logs en tiempo real
tail -f case_c_crypto/logs/crypto_*.out

# Verificar GPUs
ssh <node_name>
nvidia-smi
```

---

## 🏗️ Arquitecturas de Modelos Disponibles

### 1. MarketGPT (Transformer Base)
- **Arquitectura**: GPT-style decoder-only transformer
- **Mejor para**: Series temporales financieras
- **Características**:
  - ~100M parámetros para patrones complejos
  - Tokenización adaptativa de datos OHLC
  - Entrenamiento walk-forward continuo
  - Predicción autoregresiva multi-step

```python
from common.market_gpt import MarketGPT

model = MarketGPT(
    vocab_size=4096,
    d_model=768,
    num_layers=12,
    num_heads=12,
    context_length=512
)
```

### 2. MarketGPTMultiAsset (Multi-Asset)
- **Arquitectura**: Transformer con embeddings de asset/category
- **Mejor para**: Entrenamiento con múltiples mercados
- **Ventajas**:
  - Aprende representaciones compartidas entre assets
  - Especialización por tipo de mercado
  - Transferencia de conocimiento cross-market

```python
from common.market_gpt_multi import MarketGPTMultiAsset

model = MarketGPTMultiAsset(
    vocab_size=4096,
    d_model=768,
    num_layers=12,
    num_heads=12,
    num_assets=600,
    num_categories=5
)
```

### 3. LSTM/GRU (Legacy)
- **Arquitectura**: Redes recurrentes tradicionales
- **Mejor para**: Prototipado rápido
- **Nota**: Modelos legacy disponibles en `legacy/`

---

## 📊 Características Técnicas

### Tokenización OHLC
El sistema usa un tokenizador adaptativo que convierte precios OHLC en tokens discretos:

- **Método**: Quantile-based binning
- **Canales**: 4 (Open, High, Low, Close)
- **Vocabulary**: 1024-4096 bins según case
- **Normalización**: Log returns relativos

### Indicadores Técnicos (Legacy models)
Los modelos legacy LSTM/GRU incluyen:

- **Moving Averages**: MA-7, MA-21, MA-50
- **Exponential Moving Averages**: EMA-12, EMA-26
- **MACD**: Moving Average Convergence Divergence
- **RSI**: Relative Strength Index
- **Bollinger Bands**: Upper, Middle, Lower
- **Volatility**: Desviación estándar móvil
- **ROC**: Rate of Change
- **Volume indicators**: Volume MA y Volume Ratio

### Multi-GPU Training
- **Hardware**: 2×A100-40GB por case
- **Método**: DataParallel
- **Batch size efectivo**: 64 (32 × 2 GPUs)
- **Sincronización**: Automática de gradientes

---

## 📈 Métricas de Evaluación

### Fase 1: Pre-entrenamiento
- **Loss**: CrossEntropyLoss para next-token prediction
- **Accuracy**: % predicciones correctas
- **Perplexity**: Exp(loss)

### Fase 2: Fine-tuning
- **Validation Loss**: En ventanas out-of-sample
- **Walk-Forward Performance**: Mejora por fold

### Fase 3: Señales de Trading
- **Win Rate**: % operaciones ganadoras
- **Profit Factor**: Ganancia total / Pérdida total
- **Sharpe Ratio**: Retorno ajustado por riesgo
- **Max Drawdown**: Pérdida máxima desde pico
- **Direction Accuracy**: Precisión en predecir dirección

---

## 💡 Conceptos Clave

### Walk-Forward Analysis
Método de validación que evita **look-forward bias**:
- Ventana de entrenamiento: 2 años
- Ventana de validación: 6 meses
- Avance: 3 meses
- Simula trading real en el tiempo

### Señales de Alta Confianza
Solo se genera señal si **TODAS** las condiciones se cumplen:
1. Probabilidad de movimiento mañana > 90%
2. Probabilidad de movimiento en horizonte > 90%
3. Rango esperado > 2 desviaciones estándar (Bollinger)

### Tipos de Señales
- **Corto plazo** (1 semana): Day/Swing trading
- **Medio-Corto** (2 semanas): Swing trading
- **Medio plazo** (1 mes): Position trading
- **Medio-Largo** (2 meses): Investment

---

## 🔧 Hardware Requirements

### Por Case:
- **GPUs**: 2×A100-40GB (un nodo)
- **CPUs**: 64
- **RAM**: 128GB
- **Storage**: 50-100GB

### Tiempos Estimados:
- **Case C**: 1-2 días
- **Case B**: 3-5 días
- **Case A**: 7-10 días

---

## 📝 Mejores Prácticas

### Preparación de Datos
- Usa al menos 2-3 años de datos históricos
- Para modelos de producción, considera usar 5-10 años
- División temporal (NO aleatoria) para preservar causalidad

### Entrenamiento
- Comienza con Case C para validar sistema
- Usa early stopping para evitar overfitting
- Monitorea train loss y validation loss
- Guarda múltiples checkpoints

### Evaluación
- Siempre realiza backtesting antes de usar en producción
- No confíes solo en las métricas de error (RMSE, MAE)
- La precisión de dirección es crítica para trading
- Prueba el modelo en diferentes condiciones de mercado

### Producción
- Actualiza los datos regularmente
- Re-entrena el modelo periódicamente (mensual/trimestral)
- Monitorea el rendimiento en tiempo real
- Ten un plan de fallback si el modelo falla

---

## ⚠️ Advertencias Importantes

**Este proyecto es solo para fines educativos y de investigación.**

- La predicción del mercado de valores es extremadamente difícil
- Los rendimientos pasados no garantizan resultados futuros
- No uses este modelo para tomar decisiones de inversión reales sin consultar a profesionales
- Los mercados son influenciados por muchos factores no capturados en datos históricos
- Siempre existe riesgo de pérdida de capital

---

## 🔍 Solución de Problemas

### Error: "No se pudieron descargar datos"
- Verifica tu conexión a internet
- Asegúrate de que el ticker es válido
- Algunos tickers requieren sufijos (ej: ".MX" para México)

### Error: "CUDA out of memory"
- Reduce el `batch_size` en config
- Reduce el `hidden_size` o `num_layers`
- Usa un modelo más pequeño (Case C en lugar de Case A)

### El modelo no converge
- Ajusta el `learning_rate` (prueba valores más pequeños como 1e-5)
- Aumenta las épocas de entrenamiento
- Verifica que los datos estén correctamente normalizados
- Prueba con un modelo diferente

### Job SLURM no arranca
- Ver cola: `squeue -p long`
- Ver detalles: `scontrol show job <job_id>`
- Ver particiones disponibles: `sinfo`
- Puede que haya mucha cola, esperar

---

## 📚 Próximas Mejoras

### Fase 2 (Pendiente)
- [ ] Scripts de fine-tuning por mercado
- [ ] Pipeline completo de walk-forward
- [ ] Ensemble de modelos por fold

### Fase 3 (Pendiente)
- [ ] Motor de generación de señales
- [ ] Cálculo de Bollinger Bands multi-horizonte
- [ ] Sistema de backtesting
- [ ] API de señales en tiempo real

### Fase 4 (Futuro)
- [ ] Dashboard web interactivo
- [ ] Análisis de sentimiento de noticias
- [ ] Optimización de umbrales dinámicos
- [ ] Sistema de detección de deterioro

---

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor:

1. Fork el proyecto
2. Crea una rama para tu feature
3. Commit tus cambios
4. Push a la rama
5. Abre un Pull Request

---

## 📄 Licencia

Este proyecto está bajo licencia MIT. Ver archivo LICENSE para más detalles.

---

## 📖 Referencias

- [PyTorch Documentation](https://pytorch.org/docs/)
- [Attention Is All You Need (Transformer paper)](https://arxiv.org/abs/1706.03762)
- [GPT-2 Architecture](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [Financial Time Series Prediction](https://arxiv.org/abs/2310.03589)

---

## 📞 Contacto

Para preguntas, sugerencias o problemas, por favor abre un issue en el repositorio.

---

**Versión**: 3.0 (MarketGPT Consolidado)
**Última actualización**: 2025-01-06
**Disclaimer**: Este software se proporciona "tal cual", sin garantías de ningún tipo. El uso de este software para trading real es bajo tu propio riesgo.
