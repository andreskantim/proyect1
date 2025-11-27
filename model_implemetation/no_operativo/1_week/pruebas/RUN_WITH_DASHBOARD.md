# Ejecutar con Dashboard Dask

## Uso Rápido

```bash
cd pruebas
python reduccion_dim.py
```

**El script automáticamente:**
1. ✅ Inicia cluster Dask
2. ✅ Abre navegador con dashboard (http://localhost:8787)
3. ✅ Ejecuta análisis en paralelo
4. ✅ Muestra progreso en tiempo real

## Qué Ver en el Dashboard

### Pestaña "Status"
- Workers activos
- CPU total del cluster
- Memoria total disponible
- Tareas en cola

### Pestaña "Task Stream" ⭐ (Más Útil)
- **Flujo de tareas en tiempo real**
- Barras de colores por worker
- Duración de cada tarea
- Qué worker ejecuta qué

### Pestaña "Workers"
- CPU por worker (gráfico en tiempo real)
- Memoria por worker
- Tareas asignadas a cada worker

### Pestaña "Progress"
- Barra de progreso global
- % completado
- Tiempo estimado restante

### Pestaña "Graph"
- Grafo de dependencias entre tareas
- Visualización de pipeline

## Interpretación Visual

**Task Stream - Códigos de Color:**
- 🟦 Azul: Transferencia de datos
- 🟩 Verde: Computación activa
- 🟨 Amarillo: Serialización
- 🟥 Rojo: Error
- ⬜ Blanco: Idle (esperando)

**Workers:**
- Cada línea horizontal = un worker
- Ancho de barra = duración de tarea
- Más denso = más trabajo en paralelo

## Ejemplo de Salida

```
Iniciando Dask cluster...
✓ Created Dask cluster: 16 workers, 2 threads/worker
✓ Dashboard: http://localhost:8787/status

======================================================================
Dashboard Dask: http://localhost:8787/status
======================================================================

Abriendo dashboard en navegador...

✓ Dashboard abierto
✓ Puedes monitorear la ejecución en tiempo real

Presiona Ctrl+C para detener

Cargando datos desde: ../data/raw/bitcoin_hourly.csv
Total horas disponibles: 96543

Parámetros:
  Semana: 168 horas
  Ventana para samples: 48 horas
  Features por sample: 240
  Varianza PCA: 0.95

Preparando 574 semanas para análisis...

🚀 Ejecutando 574 análisis en paralelo con Dask...
👀 Observa el dashboard en tu navegador para ver el progreso

[########################################] | 100% Completed | 2m 15s
```

## Monitoreo en Tiempo Real

1. **Abrir dashboard**: Automático al ejecutar script
2. **Task Stream**: Verás barras moviéndose en tiempo real
3. **Workers**: Cada worker mostrará su carga de CPU
4. **Progress**: Barra de progreso global

## Troubleshooting

**Dashboard no se abre:**
```bash
# Abrir manualmente
http://localhost:8787
```

**Puerto ocupado:**
```python
# Editar src/dask_utils.py
client = get_dask_client(dashboard_address=':9999')
```

**Ver solo en terminal (sin navegador):**
```python
# Comentar en reduccion_dim.py:
# webbrowser.open(dashboard_url)
```

## Beneficios del Dashboard

- ✅ Ver qué workers están trabajando
- ✅ Detectar cuellos de botella
- ✅ Identificar workers lentos
- ✅ Optimizar particionamiento
- ✅ Debug visual de problemas
- ✅ Estimar tiempo restante

## Capturas de Pantalla Esperadas

**Task Stream:**
```
Worker 0  |████████░░░░████░░░░░░██████░░░░░|
Worker 1  |░░░░████████░░░░████░░░░░░████░░░|
Worker 2  |██░░░░░░████░░░░░░░░████░░░░░░██|
...       |....................................|
          0s     10s     20s     30s     40s
```

**Workers CPU:**
```
Worker 0: ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░ 75%
Worker 1: ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░ 95%
Worker 2: ▓▓▓▓▓▓▓▓░░░░░░░░░░░ 40%
```

## Configuración Avanzada

**Cambiar número de workers:**
```python
# En reduccion_dim.py, línea ~59
client = get_dask_client(n_workers=32)  # 32 workers
```

**Más threads por worker:**
```python
client = get_dask_client(n_workers=8, threads_per_worker=4)
```

**Cambiar tamaño de particiones:**
```python
# En reduccion_dim.py, línea donde creas el bag
bag = db.from_sequence(tasks, partition_size=10)  # 10 tareas/partición
```
