# Dask Dashboard - Visualización Web

## Inicio Rápido

### Método 1: Automático (Recomendado)
```bash
# Ejecutar cualquier script - el dashboard se inicia automáticamente
cd pruebas
python reduccion_dim.py --data ../data/raw/bitcoin_hourly.csv
```
**Dashboard:** http://localhost:8787

### Método 2: Servidor Persistente
```bash
# Terminal 1: Iniciar cluster Dask
cd scripts
./start_dask_dashboard.sh 8787 16  # puerto workers

# Terminal 2: Ejecutar scripts
cd ../pruebas
python reduccion_dim.py --data ../data/raw/bitcoin_hourly.csv
```

## Dashboard UI

Abrir en navegador: **http://localhost:8787**

### Pestañas Principales:

1. **Status** - Overview del cluster
   - Workers activos
   - CPU/memoria total
   - Tareas en cola

2. **Workers** - Detalle de cada worker
   - CPU por worker
   - Memoria por worker
   - Tareas asignadas

3. **Task Stream** - Flujo de tareas en tiempo real
   - Visualización temporal
   - Colores por tipo de tarea

4. **Progress** - Progreso de computaciones
   - % completado
   - Tiempo estimado

5. **Graph** - Grafo de dependencias
   - Estructura de tareas
   - Relaciones entre operaciones

## Monitoreo Durante Ejecución

```bash
# Ejecutar con dashboard
python reduccion_dim.py --data ../data/raw/bitcoin_hourly.csv --n-workers 16

# Ver en navegador:
# http://localhost:8787
```

**Visualizarás:**
- ⚡ Tareas ejecutándose en paralelo
- 📊 Uso de CPU por core
- 💾 Consumo de memoria
- ⏱️ Tiempo por tarea

## Scripts Compatibles

Todos los scripts del proyecto usan Dask:
- ✅ `pruebas/reduccion_dim.py`
- ✅ `validation/full_validation_pipeline.py`
- ✅ `validation/permutation_tests/permutation_test.py`
- ✅ `validation/monte_carlo/monte_carlo_test.py`

## Configuración Avanzada

### Puerto Personalizado
```python
from src.dask_utils import get_dask_client
client = get_dask_client(dashboard_address=':9999')
# Dashboard: http://localhost:9999
```

### Workers Específicos
```bash
python script.py --n-workers 32  # 32 workers
```

### Acceso Remoto
```bash
# En servidor
./start_dask_dashboard.sh 8787

# Desde local (SSH tunnel)
ssh -L 8787:localhost:8787 user@server

# Abrir: http://localhost:8787
```

## Troubleshooting

**Puerto ocupado:**
```bash
# Usar otro puerto
./start_dask_dashboard.sh 9999
```

**Dashboard no carga:**
```bash
# Verificar cluster
python -c "from dask.distributed import Client; c = Client(); print(c.dashboard_link)"
```

**Cerrar cluster:**
```python
client.close()  # En script
# O Ctrl+C en terminal
```
