# Información de Exchanges para Descarga de Datos

## Comparación de Exchanges Disponibles

| Exchange | Símbolo | Datos Desde | Años Disponibles | Volumen | Recomendación |
|----------|---------|-------------|------------------|---------|---------------|
| **Kraken** ⭐ | BTC/USD | Sep 2013 | ~11+ años | Alto | **MEJOR para historiales largos** |
| Binance | BTC/USDT | Ago 2017 | ~7 años | Muy Alto | Bueno, pero menos historia |
| Bitstamp | BTC/USD | Ago 2011 | ~13+ años | Medio | Más antiguo, pero menos líquido |
| Coinbase Pro | BTC/USD | Dic 2014 | ~10 años | Alto | Alternativa sólida |

## ⭐ Kraken (Recomendado - Por Defecto)

### Ventajas:
- ✅ **Datos desde 2013** - Más de 11 años de historia
- ✅ Alta liquidez y volumen
- ✅ Datos de calidad y confiables
- ✅ API estable y bien documentada
- ✅ Símbolo BTC/USD (par fiat, más estable)

### Desventajas:
- ⚠️ Rate limits más estrictos que Binance
- ⚠️ Descarga inicial puede tardar más (más datos)

### Uso:
```bash
# Por defecto (recomendado)
python download_bitcoin_data.py --output ../data/raw/bitcoin_hourly.csv

# Con fechas específicas
python download_bitcoin_data.py \
    --start-date 2015-01-01 \
    --end-date 2024-12-31 \
    --output ../data/raw/bitcoin_hourly.csv
```

## Binance

### Ventajas:
- ✅ Mayor volumen de trading del mundo
- ✅ API muy rápida
- ✅ Rate limits generosos

### Desventajas:
- ❌ Solo datos desde 2017 (~7 años)
- ⚠️ Símbolo BTC/USDT (stablecoin, no fiat)

### Uso:
```bash
python download_bitcoin_data.py \
    --exchange binance \
    --symbol BTC/USDT \
    --output ../data/raw/bitcoin_binance.csv
```

## Bitstamp

### Ventajas:
- ✅ **Datos más antiguos** (desde 2011)
- ✅ Exchange muy establecido
- ✅ BTC/USD (par fiat)

### Desventajas:
- ❌ Menor liquidez que Kraken/Binance
- ⚠️ API puede ser más lenta
- ⚠️ Gaps en datos históricos en algunos períodos

### Uso:
```bash
python download_bitcoin_data.py \
    --exchange bitstamp \
    --symbol BTC/USD \
    --output ../data/raw/bitcoin_bitstamp.csv
```

## Coinbase Pro

### Ventajas:
- ✅ Datos desde 2014
- ✅ Alta liquidez en mercados US
- ✅ BTC/USD (par fiat)

### Desventajas:
- ⚠️ Rate limits estrictos
- ⚠️ Menos volumen que Binance

### Uso:
```bash
python download_bitcoin_data.py \
    --exchange coinbasepro \
    --symbol BTC/USD \
    --output ../data/raw/bitcoin_coinbase.csv
```

## Recomendaciones por Caso de Uso

### Para Investigación Académica (Máximo historial)
**Opción 1**: Kraken ⭐
```bash
python download_bitcoin_data.py --exchange kraken --symbol BTC/USD
```

**Opción 2**: Bitstamp (si necesitas datos pre-2013)
```bash
python download_bitcoin_data.py --exchange bitstamp --symbol BTC/USD --start-date 2011-01-01
```

### Para Trading en Vivo (Liquidez actual)
**Binance** - Mayor volumen mundial
```bash
python download_bitcoin_data.py --exchange binance --symbol BTC/USDT
```

### Para Backtesting de Largo Plazo
**Kraken** ⭐ - Balance perfecto entre historia y calidad
```bash
python download_bitcoin_data.py --exchange kraken --symbol BTC/USD
```

### Para Validación Cruzada
Descargar de **múltiples exchanges** y comparar:
```bash
# Kraken (principal)
python download_bitcoin_data.py --exchange kraken --output ../data/raw/btc_kraken.csv

# Binance (comparación)
python download_bitcoin_data.py --exchange binance --symbol BTC/USDT --output ../data/raw/btc_binance.csv

# Comparar precios
python compare_exchanges.py --file1 btc_kraken.csv --file2 btc_binance.csv
```

## Consideraciones Importantes

### 1. BTC/USD vs BTC/USDT
- **BTC/USD**: Par fiat real, usado en Kraken, Bitstamp, Coinbase
- **BTC/USDT**: Par con stablecoin Tether, usado en Binance
- Para análisis serio, **BTC/USD es preferible** (menos riesgo de Tether)

### 2. Calidad de Datos
Todos los exchanges pueden tener:
- ⚠️ **Gaps**: Períodos sin datos (mantenimiento, crashes)
- ⚠️ **Flash crashes**: Caídas/subidas extremas momentáneas
- ⚠️ **Baja liquidez temprana**: Pre-2015 puede tener spreads grandes

El script incluye validación automática para detectar estos problemas.

### 3. Tiempo de Descarga

Estimaciones para descarga completa (todos los datos disponibles):

| Exchange | Años | Horas | Candles | Tiempo Estimado |
|----------|------|-------|---------|-----------------|
| Kraken | 11+ | ~96,000 | ~96,000 | 15-30 min |
| Binance | 7 | ~61,000 | ~61,000 | 10-20 min |
| Bitstamp | 13+ | ~114,000 | ~114,000 | 20-40 min |

### 4. Rate Limits

| Exchange | Requests/min | Candles/request | Total candles/min |
|----------|--------------|-----------------|-------------------|
| Kraken | ~15 | 720 | ~10,800 |
| Binance | ~60 | 1000 | ~60,000 |
| Bitstamp | ~8 | 1000 | ~8,000 |

El script maneja rate limits automáticamente con `enableRateLimit=True`.

## Troubleshooting

### Error: "Exchange doesn't support 1h timeframe"
Solución: Verificar timeframes disponibles
```python
import ccxt
exchange = ccxt.kraken()
print(exchange.timeframes)
```

### Error: "Rate limit exceeded"
Solución: El script ya maneja esto automáticamente. Si persiste, reduce `--n-jobs`.

### Error: "Symbol not found"
Solución: Verificar símbolo correcto para el exchange
- Kraken: `BTC/USD`, `BTC/EUR`
- Binance: `BTC/USDT`, `BTC/BUSD`
- Bitstamp: `BTC/USD`, `BTC/EUR`

### Datos con muchos gaps
Solución:
1. Probar otro exchange
2. Usar rango de fechas más reciente
3. Interpolar gaps (con precaución)

## Validación de Datos

Después de descargar, siempre revisar:

```bash
# Ver estadísticas
python -c "
import pandas as pd
df = pd.read_csv('../data/raw/bitcoin_hourly.csv')
print(f'Rows: {len(df)}')
print(f'Date range: {df.timestamp.min()} to {df.timestamp.max()}')
print(f'Missing values: {df.isnull().sum().sum()}')
print(f'Price range: ${df.close.min():.2f} to ${df.close.max():.2f}')
"

# Buscar gaps
python -c "
import pandas as pd
df = pd.read_csv('../data/raw/bitcoin_hourly.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp')
time_diffs = df['timestamp'].diff()
gaps = time_diffs[time_diffs > pd.Timedelta(hours=1.5)]
print(f'Number of gaps: {len(gaps)}')
if len(gaps) > 0:
    print('Gaps found at:')
    print(gaps)
"
```

## Resumen Ejecutivo

**Para este proyecto (predicción Bitcoin 1 semana):**

### ✅ Recomendación Principal: **Kraken**
- 11+ años de datos (2013-2024)
- Suficiente para train (año 1), validation (año 2), confidence (año 3)
- Alta calidad y liquidez
- BTC/USD (par fiat)

**Comando:**
```bash
cd scripts
python download_bitcoin_data.py --output ../data/raw/bitcoin_hourly.csv
```

**Resultado esperado:**
- ~96,000 velas horarias
- ~11 GB de datos
- 15-30 minutos de descarga
- Archivo CSV listo para usar
