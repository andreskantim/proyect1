# Dask - Computación Paralela y Distribuida

Este directorio contiene las utilidades y documentación para usar Dask en los subproyectos de proyect1.

## ⚠️ IMPORTANTE: Modo HPC

**En entornos HPC (Slurm, cgroups), SIEMPRE usa `processes=False`:**

```python
client = Client(
    n_workers=1,
    threads_per_worker=51,  # Múltiples threads
    processes=False,        # ⭐ CRUCIAL en HPC
)
```

Ver [HPC_MODE.md](HPC_MODE.md) para más detalles.

## Contenido

```
dask/
├── dask_utils.py       # Utilidades de cliente Dask
├── cleanup_dask.sh     # Script para limpiar procesos colgados
├── test_dask_config.py # Test de configuración
├── HPC_MODE.md         # ⭐ Configuración para HPC (NUEVO)
├── DASK_INFO.md        # Información técnica completa
├── PROGRESO.md         # Sistema de monitoreo de progreso
└── README.md           # Este archivo
```

## Archivos

### `dask_utils.py`
Utilidades para crear y gestionar clientes Dask. Incluye:
- `get_dask_client()` - Crea cliente con configuración automática
- `close_dask_client()` - Cierra cliente limpiamente

**Uso:**
```python
from dask_utils import get_dask_client, close_dask_client

client = get_dask_client(n_workers=16, memory_limit='8GB')
# ... usar cliente ...
close_dask_client(client)
```

### `cleanup_dask.sh`
Script para limpiar procesos de Dask que quedaron colgados.

**Uso:**
```bash
cd /path/to/proyect1/dask
./cleanup_dask.sh
```

Elimina:
- Procesos `dask-worker` colgados
- Procesos `dask-scheduler` colgados
- Archivos temporales
- Semáforos leaked

### `test_dask_config.py`
Script de prueba para verificar que Dask funciona correctamente.

**Uso:**
```bash
cd /path/to/proyect1/dask
python test_dask_config.py
```

Ejecuta 50 tareas simples y muestra:
- Configuración de recursos
- Progreso en tiempo real
- Distribución de trabajo por worker
- Verificación de resultados

## Documentación

### [DASK_INFO.md](DASK_INFO.md)
Documentación técnica completa sobre la configuración de Dask:
- Uso de recursos (80% CPU, 70% RAM)
- Configuración sin dashboard
- Monitoreo en terminal
- Ejemplos de salida

### [PROGRESO.md](PROGRESO.md)
Explicación del sistema de monitoreo de progreso:
- Actualizaciones cada 5 segundos
- Barra visual de progreso
- Velocidad y ETA
- Estadísticas por worker
- Solución de problemas

## Configuración Recomendada

### Recursos del Sistema

**Para sistemas grandes (64+ CPUs, 200+ GB RAM):**
```python
# En tu script
import os
os.environ['DASK_N_WORKERS'] = '16'  # Conservador
# o
os.environ['DASK_N_WORKERS'] = '32'  # Más rápido si el sistema lo soporta
```

**Para sistemas medianos (16-32 CPUs, 64-128 GB RAM):**
```python
os.environ['DASK_N_WORKERS'] = '8'
```

**Para sistemas pequeños (<16 CPUs, <64 GB RAM):**
```python
os.environ['DASK_N_WORKERS'] = '4'
```

### Memoria por Worker

**Recomendaciones:**
- Mínimo: 2 GB por worker
- Óptimo: 4-8 GB por worker
- Máximo: 16 GB por worker

Si tienes problemas de memoria:
```python
client = Client(
    n_workers=8,
    memory_limit='4GB'  # Reduce esto
)
```

## Uso en Subproyectos

### Desde mcpt/

```python
import sys
from pathlib import Path

# Agregar proyect1 al path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Ya NO importar dask_utils (usar configuración directa)
from dask.distributed import Client
import psutil

# Configurar recursos
total_cpus = psutil.cpu_count(logical=True)
n_workers = min(int(total_cpus * 0.8), 16)
memory_per_worker = '8GB'

# Crear cliente
client = Client(
    n_workers=n_workers,
    threads_per_worker=1,
    memory_limit=memory_per_worker,
    dashboard_address=None,
    silence_logs='error',
    processes=True
)

# ... usar cliente ...

# Cerrar
client.close()
```

## Solución de Problemas

### Error: "Killed"
```bash
# 1. Limpiar procesos
cd /path/to/proyect1/dask
./cleanup_dask.sh

# 2. Reducir workers
DASK_N_WORKERS=8 python tu_script.py
```

### Error: "resource_tracker: leaked semaphore objects"
```bash
# Limpiar y reintentar
./cleanup_dask.sh
```

### Error: Timeout al iniciar cluster
```python
# Aumentar timeout en tu script
client = Client(
    ...,
    timeout='180s'  # 3 minutos
)
```

### Script se cuelga al iniciar
```bash
# Ejecutar test primero
cd /path/to/proyect1/dask
python test_dask_config.py

# Si funciona, el problema está en el script, no en Dask
```

## Monitoreo

### Ver workers activos
```bash
watch -n 2 "ps aux | grep dask-worker | wc -l"
```

### Ver uso de CPU
```bash
htop
# Presiona 't' para vista de árbol
# Busca procesos 'dask-worker'
```

### Ver uso de memoria
```bash
watch -n 2 "free -h"
```

## Mejores Prácticas

1. **Siempre limpiar antes de ejecutar**
   ```bash
   ./cleanup_dask.sh
   ```

2. **Empezar conservador**
   ```bash
   DASK_N_WORKERS=8 python script.py
   ```

3. **Aumentar gradualmente**
   ```bash
   DASK_N_WORKERS=16 python script.py
   DASK_N_WORKERS=24 python script.py
   ```

4. **Monitorear progreso**
   - El progreso se muestra cada 5 segundos
   - Incluye velocidad y ETA

5. **Cerrar limpiamente**
   ```python
   client.close()
   ```

## Ejemplos de Uso

### Procesamiento de Permutaciones (mcpt)
Ver: `../mcpt/insample_donchian_mcpt.py`

### Test Simple
Ver: `test_dask_config.py`

### Walk-forward Analysis
Ver: `../mcpt/walkforward_donchian_mcpt.py`

## Referencias

- [Documentación oficial de Dask](https://docs.dask.org/)
- [Dask Distributed](https://distributed.dask.org/)
- [Best Practices](https://docs.dask.org/en/stable/best-practices.html)

## Contacto y Soporte

Para problemas específicos:
1. Revisar `DASK_INFO.md` y `PROGRESO.md`
2. Ejecutar `test_dask_config.py`
3. Verificar logs de ejecución
4. Usar `cleanup_dask.sh` si hay procesos colgados
