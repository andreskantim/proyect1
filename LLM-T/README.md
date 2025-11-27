# LLM-T - Machine Learning Training Project

Repositorio para proyectos de Machine Learning y entrenamiento de modelos usando PyTorch.

## 📁 Estructura del Proyecto

```
LLM-T/
├── stock_predictor/          # Sistema de predicción de precios de acciones
│   ├── data/                 # Datos descargados
│   ├── models/               # Modelos entrenados
│   ├── logs/                 # Logs y gráficas
│   ├── notebooks/            # Jupyter notebooks
│   └── ...                   # Scripts de entrenamiento y predicción
├── git-sync.sh              # Script de sincronización automática con GitHub
└── README.md                # Este archivo
```

## 🚀 Proyectos Incluidos

### Stock Predictor
Sistema completo de predicción de precios de acciones usando Redes Neuronales Recurrentes (LSTM, GRU) y Transformers.

**Características:**
- Múltiples arquitecturas: LSTM, GRU, Transformer
- Indicadores técnicos: MA, EMA, MACD, RSI, Bollinger Bands
- Entrenamiento con early stopping y backtesting
- Visualizaciones y métricas de evaluación

📖 [Ver documentación completa](stock_predictor/README.md)

## 🛠️ Configuración del Entorno

### Environment Conda: llm-training

El proyecto usa un environment conda dedicado instalado en `$STORE`:

```bash
# Activar environment con el alias
llmt

# O manualmente
conda activate llm-training
```

**Ubicación:** `/mnt/netapp2/Store_uni/home/ulc/cursos/curso396/miniconda3/envs/llm-training`

**Paquetes principales:**
- Python 3.11.14
- PyTorch 2.9.0 (CUDA 12.8)
- NumPy, Pandas, Scikit-learn
- Matplotlib, Seaborn
- Jupyter, IPython

## 📦 Gestión de Git y GitHub

### Configuración Inicial Completada

✅ Repositorio inicializado en local
✅ Remote configurado: https://github.com/andreskantim/LLM-T.git
✅ Primer commit realizado
✅ .gitignore configurado (excluye datos, modelos, logs grandes)

### Sincronización Automática con GitHub

#### Método 1: Script git-sync.sh (Recomendado)

```bash
# Sincronizar cambios con mensaje personalizado
./git-sync.sh "Descripción de los cambios realizados"

# Sincronizar con mensaje automático (timestamp)
./git-sync.sh
```

El script hace automáticamente:
1. Muestra los cambios pendientes
2. Añade todos los archivos modificados
3. Crea un commit con tu mensaje
4. Hace push a GitHub

#### Método 2: Comandos Git Tradicionales

```bash
# Ver estado
git status

# Añadir archivos
git add .

# Commit
git commit -m "Tu mensaje aquí"

# Push
git push origin master
```

### Configuración para el Primer Push

**IMPORTANTE:** Antes de hacer el primer push, debes crear el repositorio en GitHub:

1. Ve a https://github.com/new
2. Nombre del repositorio: `LLM-T`
3. Descripción: "Sistema de predicción de precios de acciones usando PyTorch"
4. Público o Privado (elige según prefieras)
5. **NO** inicialices con README, .gitignore o licencia
6. Haz clic en "Create repository"

Luego haz el primer push:

```bash
git push -u origin master
```

Te pedirá autenticación. Puedes:
- Usar un token de acceso personal (recomendado)
- Configurar SSH keys
- Usar GitHub CLI

Para configurar credenciales persistentes:
```bash
git config --global credential.helper store
```

## 🔧 Configuración de Claude Code

### Permisos Permanentes Configurados

Se ha añadido un alias para evitar confirmaciones de permisos en cada acción:

```bash
# El alias ya está configurado en ~/.bashrc
alias claude="claude --dangerously-skip-permissions"
```

Para aplicarlo en la sesión actual:
```bash
source ~/.bashrc
```

## 📝 Flujo de Trabajo Recomendado

### 1. Activar el Environment

```bash
llmt
```

### 2. Trabajar en tu Proyecto

```bash
cd stock_predictor
python train.py
```

### 3. Sincronizar Cambios

Después de completar una tarea o conjunto de cambios:

```bash
cd ~/LLM-T
./git-sync.sh "Descripción de lo que hiciste"
```

## 🎯 Comandos Útiles

### Environment Management

```bash
# Activar environment
llmt

# Desactivar
conda deactivate

# Listar environments
conda env list

# Instalar paquetes
pip install nombre_paquete
conda install nombre_paquete
```

### Git Operations

```bash
# Ver historial
git log --oneline -10

# Ver diferencias
git diff

# Ver ramas
git branch -a

# Crear nueva rama
git checkout -b nombre-rama

# Cambiar de rama
git checkout nombre-rama

# Mergear cambios
git merge nombre-rama
```

### Verificaciones Rápidas

```bash
# Espacio usado
du -sh $STORE/miniconda3
du -sh ~/LLM-T

# Verificar Python y librerías
python --version
python -c "import torch; print('PyTorch:', torch.__version__)"

# Git status
git status --short
```

## 📊 Variables de Entorno Importantes

```bash
# Directorio de almacenamiento (500GB disponibles)
$STORE = /mnt/netapp2/Store_uni/home/ulc/cursos/curso396

# Home (limitado a 10GB)
$HOME = /home/ulc/cursos/curso396
```

**Nota:** Miniconda y environments están en `$STORE` para no ocupar espacio en `$HOME`.

## 🔐 Seguridad y Mejores Prácticas

### ⚠️ Tokens de GitHub

**NUNCA** compartas tus tokens de GitHub en:
- Commits
- Código fuente
- Mensajes de chat
- Screenshots

Si accidentalmente expusiste un token:
1. Ve a https://github.com/settings/tokens
2. Revoca el token inmediatamente
3. Crea uno nuevo

### 📝 .gitignore

El repositorio está configurado para NO subir:
- Archivos de configuración de Claude (`.claude/`)
- Datos grandes (`stock_predictor/data/*.csv`)
- Modelos entrenados (`stock_predictor/models/*.pth`)
- Logs y gráficas temporales
- Cache de Python (`__pycache__/`)
- Environments virtuales

## 🆘 Solución de Problemas

### Error: "failed to push"

```bash
# Asegúrate de haber creado el repo en GitHub primero
# Luego:
git push -u origin master
```

### Error: "authentication failed"

```bash
# Configura credenciales
git config --global credential.helper store

# O usa un token en la URL
git remote set-url origin https://TU_TOKEN@github.com/andreskantim/LLM-T.git
```

### El alias llmt no funciona

```bash
# Recarga bashrc
source ~/.bashrc

# O abre una nueva terminal
```

### Problemas con conda

```bash
# Reinicializar conda
source $STORE/miniconda3/etc/profile.d/conda.sh

# Verificar configuración
conda config --show envs_dirs
```

## 📚 Recursos Adicionales

- [PyTorch Documentation](https://pytorch.org/docs/)
- [Git Documentation](https://git-scm.com/doc)
- [GitHub Guides](https://guides.github.com/)
- [Conda Documentation](https://docs.conda.io/)

## 🤝 Contribuciones

Este es un proyecto personal de aprendizaje y entrenamiento de modelos ML.

## 📄 Licencia

MIT License

---

**Última actualización:** 2025-11-05
**Mantenido por:** andreskantim
**Generado con:** Claude Code
