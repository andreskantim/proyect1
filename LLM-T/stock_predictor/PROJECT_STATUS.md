# Project Status - MarketGPT

Estado actual de implementación, estructura del proyecto, y roadmap.

---

## 📊 Estado Actual

### Última actualización: 2025-01-06

---

## ✅ Completado

### Código Base

**Modelos**:
- [x] MarketGPT (transformer GPT-style base)
- [x] MarketGPTMultiAsset (multi-asset con embeddings)
- [x] OHLCTokenizer (quantile-based, 4 canales)
- [x] MultiTaskTransformer (experimental multi-task approach)

**Data Loaders**:
- [x] UniversalLoader (Case A: 600 assets)
- [x] MultiMarketLoader (Case B: 100 assets)
- [x] CryptoDataLoader (Case C: 20 cryptos)

**Training Infrastructure**:
- [x] Training scripts para 3 cases
- [x] Multi-GPU support (DataParallel, 2×A100)
- [x] Train/Val/Test splits temporales
- [x] Early stopping y checkpointing
- [x] Evaluación en test set
- [x] Training monitor con ETA
- [x] Distributed utils (multi-GPU wrappers)

**SLURM Scripts**:
- [x] train_crypto_a100.sh (Case C)
- [x] train_reduced_a100.sh (Case B)
- [x] train_full_a100.sh (Case A)
- [x] launch_parallel_training.sh (Cases A+B en paralelo)

### Documentación

- [x] README.md (documento principal consolidado)
- [x] GETTING_STARTED.md (instalación y primeros pasos)
- [x] ARCHITECTURE.md (arquitectura completa del sistema)
- [x] TRAINING.md (guía de entrenamiento multi-GPU)
- [x] REFERENCE.md (referencia rápida de conceptos)
- [x] PROJECT_STATUS.md (este archivo)
- [x] CLAUDE_CONTEXT.md (contexto para `/context`)
- [x] READMEs individuales por case (A, B, C)

### Bugs Corregidos

- [x] Case C: DatetimeArray.sort() error → fixed con `sorted()`
- [x] Case C: CUDA module error → removido
- [x] Cases A/B/C: Test set no se usaba → añadida evaluación final
- [x] Case B: Faltaba --num-gpus → añadido a script

---

## 🔄 En Desarrollo

Actualmente no hay desarrollo activo. Proyecto listo para entrenamiento de **Fase 1**.

---

## ❌ Pendiente (Roadmap)

### FASE 2: Fine-Tuning por Mercado

**Objetivo**: Especializar modelos pre-entrenados para mercados específicos

Pendiente:
- [ ] Scripts de fine-tuning por mercado (US, EU, Commodities, Crypto)
- [ ] Pipeline completo de walk-forward analysis
- [ ] Definir tickers específicos por mercado para tuning
- [ ] Sistema de gestión de múltiples folds
- [ ] Ensemble de modelos por fold
- [ ] Evaluación comparativa de folds

**Estimación**: 2-3 semanas de desarrollo

### FASE 3: Generación de Señales y Validación

**Objetivo**: Generar señales LONG/SHORT de alta confianza y validar con benchmarks

Pendiente:
- [ ] Motor de generación de señales (LONG/SHORT)
- [ ] Sistema de predicción de probabilidades por horizonte
- [ ] Cálculo de Bollinger Bands multi-horizonte (1w, 2w, 1m, 2m)
- [ ] Definir índices de benchmark por mercado (SPY, QQQ, GLD, etc.)
- [ ] Sistema de backtesting sobre benchmarks
- [ ] Métricas de entrenamiento (tasa acierto, precision LONG/SHORT)
- [ ] Métricas de benchmark (win rate, profit factor, sharpe, drawdown)
- [ ] Sistema de logging de señales generadas
- [ ] API de señales

**Estimación**: 3-4 semanas de desarrollo

### FASE 4: Optimización y Monitoreo (Futuro)

**Objetivo**: Ajustar sistema de benchmarking y detectar deterioro

Pendiente:
- [ ] Sistema de ajuste de parámetros de benchmark
- [ ] Detección de deterioro temporal
- [ ] Dashboard de métricas de deterioro
- [ ] Sistema de re-entrenamiento automático
- [ ] Alertas de deterioro
- [ ] Registro de eventos de deterioro

**Estimación**: 4-6 semanas de desarrollo

### Infraestructura General

- [ ] Dashboard de monitoreo en tiempo real
- [ ] Base de datos de señales históricas
- [ ] Sistema de alertas (email, Slack, etc.)
- [ ] API RESTful para consumo de señales
- [ ] Documentación de API
- [ ] Tests unitarios completos
- [ ] CI/CD pipeline

**Estimación**: 2-3 semanas de desarrollo

---

## 📁 Estructura del Proyecto

### Estructura General

```
stock_predictor/
├── 📄 README.md                       # Documento principal
├── 📄 GETTING_STARTED.md              # Instalación y setup
├── 📄 ARCHITECTURE.md                 # Arquitectura completa
├── 📄 TRAINING.md                     # Guía de entrenamiento
├── 📄 REFERENCE.md                    # Referencia rápida
├── 📄 PROJECT_STATUS.md               # Este archivo
├── 📄 CLAUDE_CONTEXT.md               # Contexto para Claude
│
├── 📂 common/                         # Módulos compartidos
│   ├── market_gpt.py                  # Transformer base
│   ├── market_gpt_multi.py            # Multi-asset model
│   ├── tokenizer.py                   # OHLC tokenizer
│   ├── multitask_tokenizer.py         # Multi-task tokenizer
│   ├── multitask_model.py             # Multi-task model
│   ├── training_monitor.py            # Monitor con ETA
│   ├── distributed_utils.py           # Multi-GPU utils
│   └── uncertainty_estimation.py      # MC Dropout
│
├── 📂 case_a_full_market/             # 600 assets
│   ├── universal_loader.py
│   ├── train_full.py
│   ├── configs/
│   │   └── full_market_config.json
│   ├── slurm_scripts/
│   │   └── train_full_a100.sh
│   └── README.md
│
├── 📂 case_b_reduced/                 # 100 assets
│   ├── multi_market_loader.py
│   ├── train_reduced.py
│   ├── configs/
│   │   └── reduced_config.json
│   ├── slurm_scripts/
│   │   └── train_reduced_a100.sh
│   └── README.md
│
├── 📂 case_c_crypto/                  # 20 cryptos
│   ├── crypto_data_loader.py
│   ├── train_crypto.py
│   ├── configs/
│   │   ├── crypto_prototype.json
│   │   └── crypto_multitask_daily.json
│   ├── slurm_scripts/
│   │   ├── train_crypto_a100.sh
│   │   └── test_multitask_t4.sh
│   ├── launch.sh
│   └── README.md
│
├── 📂 legacy/                         # Código legacy (LSTM/GRU)
│   ├── train_bitcoin.py
│   ├── architecture_design.md
│   └── multi_market_design.md
│
├── 📂 data/                           # Cache de datos descargados
│   ├── crypto_multi_cache/
│   ├── reduced_cache/
│   └── full_market_cache/
│
├── 📂 checkpoints/                    # Modelos entrenados
│   ├── case_a_full_market/
│   ├── case_b_reduced/
│   └── case_c_crypto/
│
├── 📂 logs/                           # Logs de SLURM
│   ├── crypto_*.out/err
│   ├── reduced_*.out/err
│   └── full_*.out/err
│
├── 📂 slurm_scripts/                  # Scripts auxiliares
│   └── launch_parallel_training.sh
│
├── requirements_gpu.txt               # Dependencias PyTorch GPU
└── verify_installation.py            # Script de verificación
```

### Comparación de los 3 Cases

| Feature | Case C (Crypto) | Case B (Reduced) | Case A (Full) |
|---------|-----------------|------------------|---------------|
| **Assets** | 20 cryptos | 100 multi-market | 600 multi-market |
| **Categories** | 1 (Crypto) | 4 (US, Crypto, Comm, EM) | 5 (US, EU, EM, Comm, Crypto) |
| **Timeframe** | Daily | Daily | Daily |
| **Historical** | ~5 años | ~10 años | ~20 años |
| **Total candles** | ~36,500 | ~250,000 | ~3,600,000 |
| **Model params** | ~25M | ~45M | ~85M |
| **Layers** | 6 | 8 | 12 |
| **Model dim** | 256 | 512 | 768 |
| **Context length** | 128 | 256 | 512 |
| **GPU time** | 1-2 días | 3-5 días | 7-10 días |
| **Purpose** | Fast prototype | Medium validation | Full production |
| **Status** | ✅ LISTO | ✅ LISTO | ✅ LISTO |

### Flujo de Trabajo Recomendado

```
1. Case C (Crypto Prototype)
   └─> Validar sistema básico
       └─> 1-2 días en 2×A100
           └─> Si funciona bien ✓

2. Case B (Reduced)
   └─> Escalar a multi-mercado
       └─> 3-5 días en 2×A100
           └─> Si generaliza bien ✓

3. Case A (Full Market)
   └─> Sistema completo producción
       └─> 7-10 días en 2×A100
           └─> Mejor modelo esperado
```

---

## 🎯 Ready to Launch

### Case C: Crypto Prototype ✅

**Estado**: Completamente implementado y listo para ejecutar

**Características**:
- 20 cryptocurrencies
- ~5 años de datos (2019-2024), daily candles
- ~36,500 candles totales
- ~25M parámetros
- Arquitectura: 6 layers, 256 dim, 8 heads

**Estimación**: ~1-2 días en 2×A100

**Cómo lanzar**:
```bash
cd case_c_crypto/slurm_scripts
sbatch train_crypto_a100.sh
```

O usando script interactivo:
```bash
cd case_c_crypto
bash launch.sh
```

**Resultados en**:
```
checkpoints/case_c_crypto/crypto_YYYYMMDD_HHMMSS/
├── best_model.pt
├── tokenizer.pkl
├── config.json
├── training_summary.json
└── test_results.json
```

### Case B: Reduced Market ✅

**Estado**: Completamente implementado y listo para ejecutar

**Características**:
- 100 assets multi-mercado
- ~10 años de datos
- ~250,000 candles totales
- ~45M parámetros
- Arquitectura: 8 layers, 512 dim, 8 heads

**Estimación**: ~3-5 días en 2×A100

**Cómo lanzar**:
```bash
cd case_b_reduced/slurm_scripts
sbatch train_reduced_a100.sh
```

### Case A: Full Market ✅

**Estado**: Completamente implementado y listo para ejecutar

**Características**:
- 600 assets multi-mercado completo
- ~20 años de datos
- ~3,600,000 candles totales
- ~85M parámetros
- Arquitectura: 12 layers, 768 dim, 12 heads

**Estimación**: ~7-10 días en 2×A100

**Cómo lanzar**:
```bash
cd case_a_full_market/slurm_scripts
sbatch train_full_a100.sh
```

---

## 🚧 Limitaciones Conocidas

### Fase 1 (Actual)

1. **Solo pre-entrenamiento**: Fine-tuning por mercado no implementado
2. **Sin señales**: Motor de generación de señales pendiente
3. **Sin walk-forward**: Pipeline completo pendiente
4. **Test set único**: Se evalúa solo una vez al final

### Diseño

1. **DataParallel**: Usa DataParallel (single-node). Para multi-node se requiere DistributedDataParallel
2. **Datos daily**: Solo soporta daily candles. Intraday requeriría ajustes
3. **Tokenización fija**: Vocabulary size fijo en tiempo de entrenamiento

---

## 📈 Métricas de Éxito

### Fase 1: Pre-entrenamiento

**Métricas objetivo**:
- Val loss < 3.0 (convergencia)
- Val accuracy > 40% (mejor que random)
- Test accuracy cercana a val accuracy (no overfitting)
- GPU utilization > 90%

### Fase 2: Fine-tuning (Futuro)

**Métricas objetivo**:
- Mejora sobre modelo base > 5%
- Consistencia en walk-forward folds
- Generalización a mercados específicos

### Fase 3: Señales (Futuro)

**Métricas objetivo**:
- Win rate > 55%
- Profit factor > 1.5
- Sharpe ratio > 1.0
- Max drawdown < 20%

---

## 🔄 Changelog

### 2025-01-06
- ✅ Consolidación de documentación (15 → 7 archivos)
- ✅ Creación de documentos unificados
- ✅ Eliminación de duplicaciones
- ✅ Estructura clara de navegación

### 2025-01-06 (Actualización previa)
- ✅ Añadida evaluación en test set para Cases A, B, C
- ✅ Corregido bug DatetimeArray.sort en Case C
- ✅ Corregido bug CUDA module en Case C
- ✅ Añadido --num-gpus a Case B
- ✅ Documentación completa del sistema (4 fases)
- ✅ Sistema de contexto con /context

### 2025-01-05
- ✅ Setup inicial completado
- ✅ Environment llm-training configurado
- ✅ Git repository inicializado

---

## 🎯 Próximos Hitos

### Corto Plazo (1-2 semanas)
1. ✅ Completar entrenamiento Case C
2. ⏳ Analizar resultados Case C
3. ⏳ Decidir si proceder con Case B o Case A

### Medio Plazo (1-2 meses)
1. ⏳ Implementar scripts de fine-tuning (Fase 2)
2. ⏳ Implementar walk-forward pipeline
3. ⏳ Testear fine-tuning en un mercado

### Largo Plazo (3-6 meses)
1. ⏳ Implementar motor de señales (Fase 3)
2. ⏳ Sistema de backtesting
3. ⏳ Validación con benchmarks
4. ⏳ Dashboard de monitoreo (Fase 4)

---

## 📞 Contacto y Soporte

Para preguntas sobre el estado del proyecto:
- Ver documentación actualizada
- Consultar `CLAUDE_CONTEXT.md` con `/context`
- Revisar logs de entrenamiento
- Abrir issue en repositorio

---

**Estado General**: ✅ **OPERATIVO - FASE 1 LISTA**

**Próximo Paso**: Entrenar Case C para validar sistema completo

---

**Última actualización**: 2025-01-06
**Versión**: 1.0 (Post-consolidación)
