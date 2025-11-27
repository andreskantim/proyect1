# Getting Started - MarketGPT

Guía completa para instalar, configurar y comenzar a usar MarketGPT.

---

## 📋 Tabla de Contenidos

1. [Instalación](#instalación)
2. [Verificación](#verificación)
3. [Uso con Claude](#uso-con-claude)
4. [Primer Entrenamiento](#primer-entrenamiento)
5. [Monitoreo](#monitoreo)
6. [Troubleshooting](#troubleshooting)

---

## 🚀 Instalación

### 1. Activar Environment Conda

El proyecto usa el environment `llm-training` instalado en CESGA:

```bash
# Activar el entorno
conda activate llm-training

# Verificar activación
which python
# Debería mostrar: /mnt/netapp2/Store_uni/.../llm-training/bin/python
```

### 2. Navegar al Proyecto

```bash
# Ir al directorio del proyecto
cd /mnt/netapp2/Home_FT2/home/ulc/cursos/curso396/LLM-T/stock_predictor
```

### 3. Instalar Dependencias

```bash
# Asegurarse de que PyTorch esté instalado con CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Instalar otras dependencias
pip install -r requirements_gpu.txt
```

**Paquetes principales instalados:**
- Python 3.10+
- PyTorch 2.7+ con CUDA 11.8
- pandas, scikit-learn
- yfinance (para descargar datos de mercado)
- tqdm (barras de progreso)
- requests (para APIs)

---

## ✅ Verificación

### Script de Verificación Automática

```bash
# Ejecutar script de verificación
python verify_installation.py
```

**Salida esperada:**
```
✓ Python version OK: 3.10.x
✓ PyTorch installed: 2.7.x
✓ CUDA available: True
✓ GPU count: 1-2
✓ Required packages: OK
✓ All tests passed!
```

### Verificación Manual

```bash
# Verificar Python
python --version

# Verificar PyTorch y CUDA
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"

# Verificar GPUs disponibles
python -c "import torch; print('GPUs:', torch.cuda.device_count())"
```

---

## 🤖 Uso con Claude

MarketGPT está diseñado para trabajar con Claude Code. Esta sección explica cómo maximizar la colaboración.

### Comando `/context` (Recomendado)

**Al iniciar cada sesión con Claude**, ejecuta:

```
/context
```

O usa el atajo:

```
/ctx
```

**¿Qué hace?**
- Carga automáticamente `CLAUDE_CONTEXT.md`
- Proporciona a Claude contexto completo del proyecto
- Incluye arquitectura, estado actual, y pendientes
- Facilita que Claude entienda el proyecto sin repetir información

### Estructura de Documentación para Claude

```
stock_predictor/
├── README.md              ← Documento principal (overview)
├── GETTING_STARTED.md     ← Estás aquí (instalación y setup)
├── ARCHITECTURE.md        ← Detalles técnicos profundos
├── TRAINING.md            ← Guía de entrenamiento multi-GPU
├── REFERENCE.md           ← Referencia rápida de conceptos
├── PROJECT_STATUS.md      ← Estado actual del proyecto
└── CLAUDE_CONTEXT.md      ← Contexto para Claude (usado por /context)
```

### Cuándo Usar Cada Documento

| Pregunta | Documento a Consultar |
|----------|----------------------|
| "¿Cómo instalo el proyecto?" | `GETTING_STARTED.md` (este) |
| "¿Cómo funciona la arquitectura?" | `ARCHITECTURE.md` |
| "¿Cómo entreno un modelo?" | `TRAINING.md` |
| "¿Qué significa walk-forward?" | `REFERENCE.md` |
| "¿Qué está implementado?" | `PROJECT_STATUS.md` |
| "Contexto para nueva sesión" | `/context` (carga `CLAUDE_CONTEXT.md`) |

### Workflow Recomendado con Claude

#### Primera Vez en el Proyecto

1. **Lee este archivo** (GETTING_STARTED.md) ✓
2. **Ejecuta `/context`** para cargar contexto en Claude
3. **Pregunta a Claude**: "¿Cuál es el overview del proyecto?"
4. **Lee REFERENCE.md** para familiarizarte con conceptos clave
5. **Explora la estructura** de carpetas y archivos

#### Cada Nueva Sesión

1. **Ejecuta `/context`** inmediatamente
2. **Pregunta a Claude**: "¿En qué estábamos trabajando?"
3. **Consulta PROJECT_STATUS.md** para ver estado actual
4. **Continúa** desde donde lo dejaste

#### Antes de Implementar Algo Nuevo

1. **Consulta ARCHITECTURE.md** para entender la arquitectura
2. **Revisa código existente** en `common/` para ver patrones
3. **Planifica con Claude** usando el contexto cargado
4. **Implementa** siguiendo los estándares del proyecto

### Consejos para Aprovechar `/context`

✅ **Ejecútalo siempre al inicio**: Aunque Claude "deba recordar", es mejor cargarlo
✅ **Úsalo después de pausas largas**: Si llevas horas sin trabajar, recárgalo
✅ **Actualiza CLAUDE_CONTEXT.md**: Cuando completes algo importante, actualiza el archivo
✅ **Mantén sincronizado**: Si cambias arquitectura, actualiza primero CLAUDE_CONTEXT.md

### Comunicación Efectiva con Claude

Cuando Claude tiene el contexto cargado, puedes hacer preguntas como:

- "¿Qué falta por implementar?"
- "¿Cómo funciona el walk-forward analysis?"
- "¿Dónde están los scripts de entrenamiento?"
- "¿Cuál es el próximo paso en el roadmap?"

Claude responderá basándose en el contexto del proyecto.

---

## 🎯 Primer Entrenamiento

### Opción A: Test Rápido Local (Opcional)

Antes del entrenamiento completo, puedes hacer un test rápido:

```bash
# Test rápido con datos pequeños (30 min aprox)
python train_bitcoin.py \
    --config configs/quick_test.json \
    --output_dir checkpoints/quick_test \
    --log_dir logs/quick_test \
    --device cuda
```

Esto verifica que todo funciona antes de lanzar trabajos largos en SLURM.

### Opción B: Entrenamiento Completo en A100

#### Paso 1: Editar Scripts SLURM (Primera Vez)

```bash
# Editar para recibir notificaciones por email
nano case_c_crypto/slurm_scripts/train_crypto_a100.sh

# Cambiar esta línea:
#SBATCH --mail-user=your_email@domain.com
# Por tu email real
```

#### Paso 2: Lanzar Case C (Recomendado para empezar)

Case C es el más rápido (1-2 días) y perfecto para validar el sistema:

```bash
# Método 1: Script interactivo (recomendado)
cd case_c_crypto
bash launch.sh
# Selecciona opción 1: Submit SLURM job

# Método 2: Directo
cd case_c_crypto/slurm_scripts
sbatch train_crypto_a100.sh
```

**Guardar Job ID:**
```bash
# El comando sbatch devuelve algo como:
# Submitted batch job 2311562
JOB_ID=2311562
```

#### Paso 3: Otros Cases (Opcional)

Una vez validado Case C, puedes entrenar los demás:

**Case B** (100 assets, 3-5 días):
```bash
cd case_b_reduced/slurm_scripts
sbatch train_reduced_a100.sh
```

**Case A** (600 assets, 7-10 días):
```bash
cd case_a_full_market/slurm_scripts
sbatch train_full_a100.sh
```

---

## 📊 Monitoreo

### Verificar Estado del Job

```bash
# Ver tus jobs activos
squeue -u $(whoami)

# Ver estado específico
squeue -j $JOB_ID

# Ver detalles completos
scontrol show job $JOB_ID
```

### Ver Logs en Tiempo Real

```bash
# Ver output
tail -f case_c_crypto/logs/crypto_${JOB_ID}.out

# Ver errores
tail -f case_c_crypto/logs/crypto_${JOB_ID}.err
```

### Monitorear GPUs

```bash
# Identificar en qué nodo está corriendo
squeue -j $JOB_ID -o "%N"

# SSH al nodo (ejemplo: a100-01)
ssh a100-01

# Ver uso de GPU
nvidia-smi

# Monitoreo continuo
nvidia-smi -l 1  # actualiza cada segundo
```

**Uso esperado:**
- GPU Utilization: ~95%
- Memory Usage: 10-15GB (de 40GB)
- Temperature: <80°C

### Cancelar Job (si es necesario)

```bash
# Cancelar job específico
scancel $JOB_ID

# Cancelar todos tus jobs
scancel -u $(whoami)
```

---

## 📂 Resultados

Los resultados se guardarán en:

```
case_c_crypto/checkpoints/crypto_YYYYMMDD_HHMMSS/
├── best_model.pt              # Mejor modelo (por val_loss)
├── checkpoint_epoch_N.pt      # Checkpoints periódicos
├── config.json                # Configuración usada
├── tokenizer.pkl              # Tokenizador entrenado
├── asset_info.json            # Información de assets
├── training_log.json          # Log completo de entrenamiento
├── data_info.json             # Estadísticas de datos
└── test_results.json          # Resultados en test set
```

### Ver Resultados

```bash
# Ver métricas finales
cat case_c_crypto/checkpoints/*/test_results.json | jq '.'

# Ver progreso de entrenamiento
cat case_c_crypto/checkpoints/*/training_log.json | jq '.epochs[-5:]'

# Listar checkpoints
ls -lh case_c_crypto/checkpoints/*/checkpoint_*.pt
```

---

## 🔧 Troubleshooting

### Error: Environment no activado

```bash
# Activar environment
conda activate llm-training

# Si no funciona, inicializar conda
source $STORE/miniconda3/etc/profile.d/conda.sh
conda activate llm-training
```

### Error: "CUDA out of memory"

```bash
# Editar config y reducir batch_size
nano case_c_crypto/configs/crypto_prototype.json
# Cambiar batch_size: 32 → 16
```

### Error: Job no arranca

```bash
# Ver cola de la partición
squeue -p medium

# Ver GPUs disponibles
sinfo -p medium -o "%P %D %N %G %C %m"

# Si hay mucha cola, esperar o usar partición short para tests
```

### Error: Datos no descargan

```bash
# Test manual de descarga
python case_c_crypto/crypto_data_loader.py

# Si falla, verificar conexión internet
ping google.com
```

### Error: Import errors

```bash
# Verificar que estás en el directorio correcto
pwd
# Debería mostrar: .../LLM-T/stock_predictor

# Reinstalar dependencias
pip install -r requirements_gpu.txt
```

---

## 📚 Comandos Útiles SLURM

### Información de Jobs

```bash
# Ver todos tus jobs
squeue -u $(whoami)

# Ver detalles de un job
scontrol show job <job_id>

# Ver historial de jobs
sacct -u $(whoami)

# Ver jobs recientes con estado
sacct -u $(whoami) --format=JobID,JobName,State,Elapsed,TimeLimit
```

### Información del Cluster

```bash
# Ver particiones disponibles
sinfo

# Ver GPUs disponibles
sinfo -N -o "%N %G %t" | grep a100

# Ver nodos de una partición
sinfo -p medium -N
```

### Gestión de Jobs

```bash
# Cancelar job
scancel <job_id>

# Cancelar todos tus jobs
scancel -u $(whoami)

# Hold (pausar) un job pendiente
scontrol hold <job_id>

# Release (reanudar) un job en hold
scontrol release <job_id>
```

---

## ⏱️ Tiempos Esperados

| Case | Assets | GPUs | Tiempo Estimado |
|------|--------|------|----------------|
| **C** | 20 cryptos | 2×A100 | 1-2 días |
| **B** | 100 multi-market | 2×A100 | 3-5 días |
| **A** | 600 todos mercados | 2×A100 | 7-10 días |

**Nota:** Los tiempos son estimaciones. Pueden variar según:
- Carga del cluster
- Configuración de hiperparámetros
- Convergencia del modelo

---

## 🎓 Próximos Pasos

### 1. Durante el Entrenamiento
- Monitorea métricas en logs
- Verifica que val_loss disminuya
- Observa convergencia

### 2. Después del Entrenamiento
- Analiza resultados en test set
- Revisa training_log.json
- Compara diferentes cases (si entrenaste varios)

### 3. Fase 2: Fine-Tuning (Próximamente)
- Scripts de fine-tuning por mercado
- Walk-forward analysis
- Especialización de modelos

### 4. Fase 3: Señales (Futuro)
- Generación de señales de trading
- Backtesting
- Validación con benchmarks

---

## 📖 Más Información

- **Arquitectura completa**: Ver [ARCHITECTURE.md](ARCHITECTURE.md)
- **Guía de entrenamiento**: Ver [TRAINING.md](TRAINING.md)
- **Referencia rápida**: Ver [REFERENCE.md](REFERENCE.md)
- **Estado del proyecto**: Ver [PROJECT_STATUS.md](PROJECT_STATUS.md)

---

## 💡 Consejos Finales

✅ **Comienza con Case C**: Es rápido y valida que todo funciona
✅ **Monitorea activamente**: Revisa logs regularmente
✅ **Usa /context con Claude**: Facilita colaboración
✅ **Guarda Job IDs**: Anota los IDs para monitoreo
✅ **Sé paciente**: El entrenamiento puede tardar días

---

**¡Listo para empezar!** 🚀

Si tienes dudas, consulta la documentación o usa `/context` con Claude para obtener ayuda contextualizada.

---

**Última actualización**: 2025-01-06
