# Training Guide - MarketGPT

Guía completa para entrenar modelos MarketGPT usando múltiples GPUs A100.

---

## 📋 Tabla de Contenidos

1. [Overview](#overview)
2. [Cases Summary](#cases-summary)
3. [Quick Start](#quick-start)
4. [Multi-GPU Training](#multi-gpu-training)
5. [Configuration Details](#configuration-details)
6. [Monitoring](#monitoring)
7. [Outputs](#outputs)
8. [Troubleshooting](#troubleshooting)
9. [Expected Performance](#expected-performance)
10. [After Training](#after-training)

---

## 🎯 Overview

MarketGPT soporta 3 configuraciones de entrenamiento (**Cases**), cada una con diferentes números de assets y tamaños de modelo. Cada case usa **2×A100 GPUs** en un solo nodo con DataParallel.

**Casos disponibles:**
- **Case A**: Full market (600 assets) - Sistema completo
- **Case B**: Reduced market (100 assets) - Baseline curado
- **Case C**: Crypto prototype (20 cryptos) - Prototipo rápido

---

## 📊 Cases Summary

| Case | Assets | GPUs | CPUs | RAM | Tiempo | Parámetros |
|------|--------|------|------|-----|--------|------------|
| **A** | 600 (all markets) | 2×A100 | 64 | 128GB | 7-10 días | ~85M |
| **B** | 100 (multi-market) | 2×A100 | 64 | 128GB | 3-5 días | ~45M |
| **C** | 20 (crypto only) | 2×A100 | 64 | 128GB | 1-2 días | ~25M |

### Case A: Full Market (600 assets)

**Assets incluidos:**
- **US Stocks**: 300 (S&P 500 top companies)
- **European Stocks**: 150 (major EU indices)
- **Emerging Markets**: 50 (ETFs + companies)
- **Commodities**: 30 (metals, energy, agriculture)
- **Crypto**: 70 (top cryptocurrencies)

**Model Configuration:**
- d_model=768, num_layers=12, num_heads=12
- context_length=512, vocab_size=4096
- Total parameters: ~85M

### Case B: Reduced Market (100 assets)

**Assets incluidos:**
- **US Stocks**: 50
- **Crypto**: 20
- **Commodities**: 15
- **Emerging Markets**: 15

**Model Configuration:**
- d_model=512, num_layers=8, num_heads=8
- context_length=256, vocab_size=2048
- Total parameters: ~45M

### Case C: Crypto Prototype (20 assets)

**Assets incluidos:**
- 20 major cryptocurrencies

**Model Configuration:**
- d_model=256, num_layers=6, num_heads=8
- context_length=128, vocab_size=1024
- Total parameters: ~25M

---

## 🚀 Quick Start

### Entrenar Individual Case

#### Case C (Recomendado para empezar)

```bash
cd case_c_crypto/slurm_scripts
sbatch train_crypto_a100.sh
```

Monitor:
```bash
squeue -u $(whoami)
tail -f ../logs/crypto_*.out
```

#### Case B

```bash
cd case_b_reduced/slurm_scripts
sbatch train_reduced_a100.sh
```

Monitor:
```bash
tail -f ../logs/reduced_*.out
```

#### Case A

```bash
cd case_a_full_market/slurm_scripts
sbatch train_full_a100.sh
```

Monitor:
```bash
tail -f ../logs/full_*.out
```

### Entrenar Múltiples Cases en Paralelo

Para entrenar Cases A y B simultáneamente (requiere 4 GPUs totales):

```bash
cd /mnt/netapp2/Home_FT2/home/ulc/cursos/curso396/LLM-T/stock_predictor
./launch_parallel_training.sh
```

Este script:
1. Verifica todos los scripts y recursos
2. Muestra estado actual del cluster
3. Pide confirmación
4. Envía ambos jobs SLURM
5. Crea script de monitoreo
6. Muestra Job IDs y comandos

---

## 🔧 Multi-GPU Training

### Architecture

Cada case usa **DataParallel** para entrenamiento multi-GPU:

- Modelo replicado en 2 GPUs
- Batch dividido automáticamente entre GPUs
- Effective batch size = `base_batch_size × num_gpus`
- Gradientes sincronizados después de cada backward pass

**Ejemplo:**
- Base batch size: 32
- Número de GPUs: 2
- **Effective batch size: 64**

Esto duplica el throughput manteniendo consistencia de gradientes.

### Implementation

El training script usa:

```python
from common.distributed_utils import MultiGPUWrapper

# Inicializar modelo
model = MarketGPT(config)

# Wrap para 2 GPUs
if num_gpus > 1:
    model = MultiGPUWrapper(model, num_gpus=2)
else:
    model = model.to('cuda')
```

### Benefits

- **2× throughput**: Batch dividido en 2 GPUs
- **Misma convergencia**: Gradientes sincronizados
- **Automático**: Sin cambios en training loop
- **Memory efficient**: Modelo replicado, no datos

---

## ⚙️ Configuration Details

### Case A Configuration

**File**: `case_a_full_market/configs/full_market_config.json`

```json
{
  "model": {
    "vocab_size": 4096,
    "context_length": 512,
    "d_model": 768,
    "num_layers": 12,
    "num_heads": 12,
    "d_ff": 3072,
    "dropout": 0.1,
    "asset_embed_dim": 64,
    "category_embed_dim": 32
  },
  "training": {
    "batch_size": 32,
    "learning_rate": 3e-4,
    "epochs": 100,
    "early_stopping_patience": 15,
    "gradient_clip": 1.0
  },
  "data": {
    "train_ratio": 0.70,
    "val_ratio": 0.15,
    "test_ratio": 0.15
  }
}
```

### Case B Configuration

**File**: `case_b_reduced/configs/reduced_config.json`

```json
{
  "model": {
    "vocab_size": 2048,
    "context_length": 256,
    "d_model": 512,
    "num_layers": 8,
    "num_heads": 8,
    "d_ff": 2048,
    "dropout": 0.1,
    "asset_embed_dim": 32,
    "category_embed_dim": 16
  },
  "training": {
    "batch_size": 32,
    "learning_rate": 5e-4,
    "epochs": 100,
    "early_stopping_patience": 10,
    "gradient_clip": 1.0
  }
}
```

### Case C Configuration

**File**: `case_c_crypto/configs/crypto_prototype.json`

```json
{
  "model": {
    "vocab_size": 1024,
    "context_length": 128,
    "d_model": 256,
    "num_layers": 6,
    "num_heads": 8,
    "d_ff": 1024,
    "dropout": 0.1
  },
  "training": {
    "batch_size": 32,
    "learning_rate": 3e-4,
    "epochs": 50,
    "early_stopping_patience": 10,
    "gradient_clip": 1.0
  }
}
```

### SLURM Configuration

Configuración estándar para cada case:

```bash
#SBATCH --nodes=1                    # Single node
#SBATCH --ntasks=1                   # Single task
#SBATCH --cpus-per-task=64           # 32 per GPU × 2
#SBATCH --gres=gpu:a100:2            # 2 A100 GPUs
#SBATCH --mem=128G                   # 64GB per GPU
#SBATCH --time=7-00:00:00            # 7 days max
#SBATCH --partition=long             # Long partition
```

### Environment Variables

```bash
export CUDA_VISIBLE_DEVICES=0,1      # Use both GPUs
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

---

## 📊 Monitoring

### Check Job Status

```bash
# All your jobs
squeue -u $(whoami)

# Specific job
scontrol show job <JOB_ID>

# Job details with format
squeue -j <JOB_ID> -o "%i %P %j %u %T %M %D %R"
```

### View Logs

```bash
# Real-time monitoring
tail -f case_*/logs/*_<JOB_ID>.out

# Check for errors
tail -f case_*/logs/*_<JOB_ID>.err

# Last N lines
tail -n 100 case_*/logs/*_<JOB_ID>.out
```

### GPU Utilization

```bash
# SSH to compute node
ssh <node_name>

# Watch GPU usage
nvidia-smi -l 1

# Detailed GPU info
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv
```

**Expected GPU utilization**: ~95%

### Training Progress

Los logs muestran:

```
Epoch 10/100
Train Loss: 3.245 | Train Acc: 42.3%
Val Loss: 3.512 | Val Acc: 39.8%
Time: 2h 15m | ETA: 19h 30m
```

Monitor especialmente:
- **Val Loss debe bajar**: Si sube, puede haber overfitting
- **Train/Val gap**: Gap grande = overfitting
- **ETA**: Tiempo estimado restante

---

## 📂 Outputs

### Output Structure

Cada case produce:

```
case_X/checkpoints/<experiment_name>/
├── best_model.pt              # Best model by val loss
├── checkpoint_epoch_N.pt      # Periodic checkpoints
├── config.json                # Full configuration
├── tokenizer.pkl              # Fitted tokenizer
├── asset_info.json            # Asset mappings
├── training_log.json          # Epoch-by-epoch metrics
├── data_info.json             # Dataset statistics
└── test_results.json          # Final test metrics
```

### Case A Outputs

```
case_a_full_market/
├── checkpoints/
│   └── full_market_YYYYMMDD_HHMMSS/
│       ├── best_model.pt
│       ├── checkpoint_epoch_N.pt
│       ├── config.json
│       ├── tokenizer.pkl
│       ├── asset_info.json
│       ├── training_log.json
│       ├── data_info.json
│       └── test_results.json
└── logs/
    ├── full_<JOB_ID>.out
    └── full_<JOB_ID>.err
```

### View Results

```bash
# Ver métricas finales
cat checkpoints/*/test_results.json | jq '.'

# Ver últimas 5 épocas
cat checkpoints/*/training_log.json | jq '.epochs[-5:]'

# Listar checkpoints
ls -lh checkpoints/*/checkpoint_*.pt
```

---

## 🔍 Troubleshooting

### Out of Memory (OOM)

**Síntoma**: `RuntimeError: CUDA out of memory`

**Soluciones**:

1. Reducir batch size en config:
```json
"training": {
  "batch_size": 16  // o 24
}
```

Con 2 GPUs, effective batch size será 32 o 48.

2. Reducir model size:
```json
"model": {
  "d_model": 512,  // en lugar de 768
  "num_layers": 8   // en lugar de 12
}
```

3. Reducir context length:
```json
"model": {
  "context_length": 256  // en lugar de 512
}
```

### Job Pending Forever

**Causas**:
1. No hay GPUs disponibles
2. Partición llena
3. Request de memoria/CPU muy alto

**Soluciones**:

```bash
# Ver GPUs disponibles
sinfo -p long -o "%P %D %N %G %C %m"

# Ver cola
squeue -p long

# Usar partición más corta para tests
#SBATCH --partition=medium  # en lugar de long
```

### Training Loss Not Decreasing

**Checks**:

1. **Learning rate muy alto/bajo**:
```json
"learning_rate": 1e-4  // probar valores diferentes
```

2. **Gradient clipping muy agresivo**:
```json
"gradient_clip": 5.0  // aumentar
```

3. **Data preprocessing issue**: Verificar tokenizer
```bash
python -c "from common.tokenizer import OHLCTokenizer; t = OHLCTokenizer(); print(t)"
```

4. **Model size**: Puede ser muy pequeño o muy grande

**Debug**:
```bash
# Ver training log
cat checkpoints/*/training_log.json | jq '.epochs[] | {epoch, train_loss, val_loss}'
```

### Import Errors

**Solución**:
```bash
# Verify conda environment
conda activate llm-training

# Reinstall dependencies
cd stock_predictor
pip install -r requirements_gpu.txt

# Verify imports
python -c "from common.market_gpt_multi import MarketGPTMultiAsset"
```

### Download de Datos Falla

**Solución**:
```bash
# Test manual
cd case_c_crypto
python crypto_data_loader.py

# Si falla, verificar internet
ping google.com

# Ver logs de descarga
ls -lh data/*/
```

---

## 📈 Expected Performance

### Case A (600 assets)

- **Training time**: 7-10 días en 2×A100
- **Val loss**: 2.5-3.5 (después de convergencia)
- **Val accuracy**: 40-50%
- **Iterations/sec**: ~5-10
- **GPU memory**: ~15-20GB por GPU

### Case B (100 assets)

- **Training time**: 3-5 días en 2×A100
- **Val loss**: 2.0-3.0 (después de convergencia)
- **Val accuracy**: 45-55%
- **Iterations/sec**: ~10-15
- **GPU memory**: ~10-15GB por GPU

### Case C (20 cryptos)

- **Training time**: 1-2 días en 2×A100
- **Val loss**: 1.5-2.5 (después de convergencia)
- **Val accuracy**: 50-60%
- **Iterations/sec**: ~15-25
- **GPU memory**: ~5-10GB por GPU

---

## 🎯 After Training

### 1. Evaluate Model

```bash
# Revisar test results
cat case_X/checkpoints/*/test_results.json

# Analizar training log
python -c "
import json
with open('case_X/checkpoints/.../training_log.json') as f:
    log = json.load(f)
    print(f'Best val loss: {log[\"best_val_loss\"]}')
    print(f'Best epoch: {log[\"best_epoch\"]}')
"
```

### 2. Compare Cases

Si entrenaste múltiples cases:

```bash
# Comparar test accuracy
echo "Case A:"
cat case_a_full_market/checkpoints/*/test_results.json | jq '.test_accuracy'
echo "Case B:"
cat case_b_reduced/checkpoints/*/test_results.json | jq '.test_accuracy'
echo "Case C:"
cat case_c_crypto/checkpoints/*/test_results.json | jq '.test_accuracy'
```

### 3. Next Steps

Después de pre-entrenamiento exitoso:

1. **Fase 2: Fine-Tuning** (Pendiente implementación)
   - Fine-tune por mercado (US, EU, Commodities, Crypto)
   - Walk-forward analysis
   - Ensemble de modelos

2. **Fase 3: Señales** (Pendiente implementación)
   - Generación de señales de trading
   - Backtesting
   - Validación con benchmarks

---

## 💡 Resource Optimization

### Para Reducir Tiempo de Entrenamiento

1. **Aumentar batch size**: 32 → 48 → 64 (si memoria permite)
2. **Reducir model size**: Menos layers/menor d_model
3. **Usar más GPUs**: 4 o 8 GPUs si disponibles

### Para Reducir Uso de Memoria

1. **Reducir batch size**: 32 → 16 → 8
2. **Reducir context length**: 512 → 256 → 128
3. **Enable gradient checkpointing**: `--gradient_checkpointing`

### Para Mejorar Calidad del Modelo

1. **Más datos**: Extender start_date para más historia
2. **Better preprocessing**: Añadir indicadores técnicos
3. **Hyperparameter tuning**: Grid search en lr, dropout
4. **Ensemble**: Entrenar múltiples modelos con diferentes seeds

---

## 📝 Workflow Recomendado

### Orden de Entrenamiento

Train cases secuencialmente para comparar performance:

1. **Empezar con Case C** (más rápido, 1-2 días)
   - Prototipo en crypto data
   - Test multi-GPU setup
   - Verificar que entrenamiento funciona

2. **Luego Case B** (medio, 3-5 días)
   - Multi-market baseline
   - Mejor cobertura de assets
   - Comparar con Case C

3. **Finalmente Case A** (más largo, 7-10 días)
   - Cobertura completa de mercado
   - Mejor performance esperada
   - Comparar con Case B

### ¿Por qué Secuencial?

- **Resource efficiency**: Uso completo de 2 GPUs por case
- **Fair comparison**: Cada case obtiene mismos recursos
- **Cluster friendly**: Single job por usuario
- **Debugging**: Foco en un case a la vez

**Alternativa:** Si tienes 4+ GPUs disponibles, puedes entrenar 2 cases en paralelo usando `launch_parallel_training.sh`.

---

## 🔗 Common Commands Cheat Sheet

```bash
# Submit job
sbatch case_X/slurm_scripts/train_X_a100.sh

# Check job status
squeue -u $(whoami)

# View job details
scontrol show job <JOB_ID>

# Cancel job
scancel <JOB_ID>

# View logs
tail -f case_X/logs/*_<JOB_ID>.out
tail -f case_X/logs/*_<JOB_ID>.err

# Check GPU usage (on compute node)
ssh <node_name>
nvidia-smi

# List checkpoints
ls -lh case_X/checkpoints/

# View training progress
cat case_X/checkpoints/*/training_log.json | jq '.epochs[-5:]'
```

---

## ❓ FAQ

**Q: ¿Puedo usar más de 2 GPUs por case?**
A: Sí, modifica `--gres=gpu:a100:N` en script SLURM y `--num-gpus N` en comando training.

**Q: ¿Puedo entrenar múltiples cases simultáneamente?**
A: Técnicamente sí, pero cada case ya usa 2 GPUs. Entrenar 2 cases = 4 GPUs, que puede no estar disponible.

**Q: ¿Por qué no usar todas las 8 GPUs de a100-65?**
A: Single-case no escala bien más allá de 2-4 GPUs. Mejor entrenar cases secuencialmente.

**Q: ¿Puedo resumir desde un checkpoint?**
A: Sí, modifica training script para cargar desde `checkpoint_epoch_N.pt`.

**Q: ¿Qué pasa si mi job se queda sin tiempo?**
A: Usa checkpoint para resumir. Aumenta time limit o usa partición con más tiempo.

**Q: ¿Cómo sé si multi-GPU está funcionando?**
A: Verifica logs: "Using GPUs: 2" y `nvidia-smi` muestra ambas GPUs activas.

---

## 📞 Support

Para problemas:
1. Verificar logs: `case_*/logs/*.err`
2. Verificar GPU availability: `sinfo -p long`
3. Test en single GPU primero: `--num-gpus 1`
4. Consultar READMEs de cases individuales

---

## 📖 Summary

- **Cada case usa 2 GPUs en un solo nodo**
- **Entrenar cases secuencialmente para mejores resultados**
- **Tiempos esperados**: C (1-2 días), B (3-5 días), A (7-10 días)
- **Monitorear con**: `squeue`, `tail -f logs/*.out`, `nvidia-smi`
- **Output**: Checkpoints, logs, tokenizers en `case_X/checkpoints/`

---

**Happy Training!** 🚀

---

**Última actualización**: 2025-01-06
