# Market GPT Multi-Market Design
## Sistema de Predicción Universal de Mercados Financieros

---

## 1. ARQUITECTURA MULTI-MERCADO

### 1.1 Categorías de Assets

```
┌─────────────────────────────────────────────────────────────┐
│                    MARKET GPT UNIVERSE                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────┐  ┌──────────────────┐                │
│  │   US STOCKS      │  │  EUROPE STOCKS   │                │
│  │   300 tickets    │  │   150 tickets    │                │
│  │   30 años        │  │   20 años        │                │
│  │   S&P 500 top    │  │   STOXX 600      │                │
│  └──────────────────┘  └──────────────────┘                │
│                                                              │
│  ┌──────────────────┐  ┌──────────────────┐                │
│  │ EMERGING MKTS    │  │   COMMODITIES    │                │
│  │   100 tickets    │  │    30 tickets    │                │
│  │   15 años        │  │    30 años       │                │
│  │   India/Brasil/  │  │   Gold/Oil/      │                │
│  │   China          │  │   Silver/Copper  │                │
│  └──────────────────┘  └──────────────────┘                │
│                                                              │
│  ┌──────────────────┐  ┌──────────────────┐                │
│  │   CRYPTO         │  │   FOREX (opt)    │                │
│  │   20 tickets     │  │   15 pairs       │                │
│  │   5-8 años       │  │   20 años        │                │
│  │   Top 20 mcap    │  │   Majors + Exot  │                │
│  └──────────────────┘  └──────────────────┘                │
│                                                              │
│  TOTAL: 600-615 assets × ~20 años × 252 días = 3.6M candles│
└─────────────────────────────────────────────────────────────┘
```

### 1.2 Timeframe: DIARIO (1 day candles)

**Decisión: Velas diarias**

**Ventajas:**
- ✅ 20-30 años de histórico disponible para la mayoría de assets
- ✅ Menos ruido, patrones más claros
- ✅ Menos storage (~3GB vs 300GB para 1-min)
- ✅ Entrenamiento más rápido
- ✅ Suficientes datos para 100M parámetros

**Desventajas:**
- ❌ No captura patrones intraday
- ❌ Solo una predicción por día (vs 1440 en 1-min)

**Balance:**
Para un modelo general multi-mercado, diario es óptimo. Si después se necesita intraday, se puede fine-tune específico.

---

## 2. ASSET SELECTION DETALLADA

### 2.1 US Stocks (300 tickets)

**Selección: S&P 500 Top 300 por Market Cap**

**Categorías internas:**
- Technology (60): AAPL, MSFT, GOOGL, NVDA, META, TSLA, etc.
- Financials (50): JPM, BAC, WFC, GS, MS, etc.
- Healthcare (40): UNH, JNJ, LLY, PFE, ABBV, etc.
- Consumer (40): AMZN, WMT, HD, NKE, COST, etc.
- Industrials (30): BA, CAT, GE, UPS, etc.
- Energy (30): XOM, CVX, COP, etc.
- Others (50): REITs, Utilities, Materials

**Data source:** Yahoo Finance (yfinance)
**Histórico:** 30 años (1994-2024)
**Trading days:** ~252/año

### 2.2 Europe Stocks (150 tickets)

**Selección: STOXX Europe 600 Top 150**

**Por país:**
- UK (40): Shell, HSBC, BP, GSK, Unilever, etc.
- Germany (30): SAP, Siemens, Allianz, BMW, Deutsche Bank, etc.
- France (30): LVMH, L'Oréal, TotalEnergies, Sanofi, etc.
- Switzerland (20): Nestlé, Novartis, Roche, UBS, etc.
- Others (30): Netherlands, Italy, Spain

**Data source:** Yahoo Finance
**Histórico:** 20 años
**Trading days:** ~250/año (European markets)

### 2.3 Emerging Markets (100 tickets)

**Por región:**

**India (30):**
- RELIANCE.NS, TCS.NS, INFY.NS, HDFCBANK.NS, etc.
- Nifty 50 principales

**China (30):**
- Alibaba (BABA), Tencent (0700.HK), ICBC, etc.
- Top ADRs + Hong Kong listings

**Brazil (20):**
- VALE, PETR4.SA, ITUB, BBDC4.SA, etc.
- Bovespa principales

**Others (20):**
- México, Corea del Sur, Taiwan, Sudáfrica

**Data source:** Yahoo Finance
**Histórico:** 15 años (más limitado)

### 2.4 Commodities (30 tickets)

**Metales Preciosos (8):**
- GC=F (Gold futures)
- SI=F (Silver)
- PL=F (Platinum)
- PA=F (Palladium)
- GLD, SLV (ETFs)

**Energía (8):**
- CL=F (Crude Oil WTI)
- BZ=F (Brent Oil)
- NG=F (Natural Gas)
- USO, UNG (ETFs)

**Metales Industriales (6):**
- HG=F (Copper)
- ALI=F (Aluminum)

**Agrícolas (8):**
- ZC=F (Corn)
- ZW=F (Wheat)
- ZS=F (Soybeans)
- KC=F (Coffee)
- SB=F (Sugar)

**Data source:** Yahoo Finance futures
**Histórico:** 30+ años

### 2.5 Cryptocurrencies (20 tickets)

**Top 20 por Market Cap:**
1. BTC-USD (Bitcoin)
2. ETH-USD (Ethereum)
3. BNB-USD (Binance Coin)
4. XRP-USD (Ripple)
5. ADA-USD (Cardano)
6. DOGE-USD (Dogecoin)
7. SOL-USD (Solana)
8. TRX-USD (Tron)
9. DOT-USD (Polkadot)
10. MATIC-USD (Polygon)
... (top 20)

**Data source:** Yahoo Finance (crypto pairs)
**Histórico:** 3-8 años (limitado por asset)
**Trading days:** 365/año (24/7 markets)

### 2.6 Forex (Opcional - 15 pairs)

**Majors (7):**
- EUR/USD, GBP/USD, USD/JPY, USD/CHF
- AUD/USD, NZD/USD, USD/CAD

**Crosses (5):**
- EUR/GBP, EUR/JPY, GBP/JPY, EUR/CHF, AUD/JPY

**Exotics (3):**
- USD/MXN, USD/BRL, USD/ZAR

**Data source:** Yahoo Finance o Alpha Vantage
**Histórico:** 20 años
**Trading days:** ~260/año (business days, 24h market)

**Nota sobre Forex:** El usuario dijo "menos el FOREX" inicialmente, pero preguntó si se puede incluir. Mi recomendación: **Opcional - añadir después del pre-training principal**

---

## 3. PROBLEMA DE CORRELACIONES

### 3.1 ¿Es preocupante la correlación?

**Respuesta: NO, es BENEFICIOSA si se maneja bien**

**Tipos de correlación:**

1. **Intra-categoría (alta correlación):**
   - S&P 500 stocks: correlación ~0.6-0.8 entre sí
   - Todas suben/bajan con el mercado

2. **Inter-categoría (correlación media):**
   - US stocks ↔ Europe stocks: ~0.5-0.7
   - Crypto ↔ Tech stocks: ~0.3-0.5

3. **Negativa (diversificación):**
   - Stocks ↔ Gold: ~-0.2
   - USD ↔ Commodities: variable

### 3.2 Por qué NO es problema:

**A. Asset Embeddings**
```python
# El modelo SABE qué asset está viendo
asset_embedding = nn.Embedding(num_assets, d_model)

# Aprende:
# - Patrones comunes (momentum, reversiones)
# - Características únicas de cada asset
# - Relaciones entre assets (correlaciones)
```

**B. Jerarquía de Embeddings**
```python
# MEJOR: Embeddings jerárquicos
category_embedding = nn.Embedding(num_categories, d_model)  # US, EU, Crypto, etc.
asset_embedding = nn.Embedding(num_assets, d_model)

# El modelo aprende:
# - Nivel 1: Patrones generales de mercado
# - Nivel 2: Patrones por categoría (Crypto más volátil)
# - Nivel 3: Patrones específicos de asset (BTC vs ETH)
```

**C. Las correlaciones son INFORMACIÓN útil**
- Si S&P 500 cae → probablemente Nasdaq también
- Si Gold sube → probablemente hay risk-off en stocks
- El modelo puede APRENDER y USAR estas relaciones

### 3.3 Mitigación de problemas:

**Estrategia 1: Data Augmentation**
- NO dar siempre los mismos assets en el mismo orden
- Shuffle por batch
- Sample random assets por época

**Estrategia 2: Balanceo por categoría**
```python
# En cada batch, incluir:
- 40% US stocks (más representados)
- 20% Europe stocks
- 15% Emerging
- 10% Commodities
- 10% Crypto
- 5% Forex (si se incluye)
```

**Estrategia 3: Regularización**
- Dropout en asset embeddings
- Weight decay
- Mixup entre assets de misma categoría

---

## 4. ENTRENAMIENTO WALK-FORWARD POR CATEGORÍA

### 4.1 Pipeline de Entrenamiento

```
PHASE 0: Data Preparation
├─ Download 600 assets × 20 años daily data
├─ Clean, validate, handle missing data
├─ Fit tokenizer en todos los datos
└─ Create category/asset ID mappings

PHASE 1: Pre-training Multi-Market (GENERAL)
├─ Train: 90% de datos de TODOS los mercados
├─ Val: 5% held-out de todos los mercados
├─ Test: 5% held-out temporal (últimos N días)
├─ Objetivo: Aprender patrones UNIVERSALES
├─ Epochs: 20-30
├─ Time: ~5-7 días en A100
└─ Output: model_pretrained.pt

PHASE 2: Fine-tuning por Categoría (Walk-Forward)
Para cada categoría C en [US, EU, EM, Commodities, Crypto]:
│
├─ Load: model_pretrained.pt
│
├─ Walk-Forward en datos de categoría C:
│  │
│  ├─ Window 1 (6 meses):
│  │  ├─ Test: evaluar modelo actual
│  │  ├─ Fine-tune: 3 epochs solo en assets de C
│  │  └─ Metrics: guardar resultados
│  │
│  ├─ Window 2 (siguiente 6 meses):
│  │  ├─ Test: evaluar modelo actualizado
│  │  ├─ Fine-tune: 3 epochs
│  │  └─ Metrics: guardar
│  │
│  └─ ... (hasta agotar datos de test)
│
├─ Out-of-sample test:
│  └─ Últimos 2 años NUNCA vistos
│     └─ Evaluar modelo final de categoría C
│
└─ Output: model_finetuned_{category}.pt


PHASE 3: Test de Generalización
├─ Cargar model_pretrained.pt (sin fine-tune)
├─ Evaluar en cada categoría
└─ Comparar: pretrained vs fine-tuned por categoría
```

### 4.2 Cross-Validation en Series Temporales

**Pregunta: ¿Usar CV?**

**Respuesta: SÍ, pero Time Series Split (NO K-Fold)**

**❌ K-Fold tradicional:**
```
Fold 1: Train [1,2,4,5] Test [3]
Fold 2: Train [1,3,4,5] Test [2]
→ DATA LEAKAGE! Entrenas en datos futuros
```

**✅ Time Series Split:**
```
Fold 1: Train [1,2] → Test [3]
Fold 2: Train [1,2,3] → Test [4]
Fold 3: Train [1,2,3,4] → Test [5]
→ CORRECTO: Solo entrenas en pasado
```

**✅ Walk-Forward (mejor para trading):**
```
Nuestra implementación actual ES walk-forward
= Time Series CV + retraining continuo
```

**Implementación:**
```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)

for train_idx, test_idx in tscv.split(data):
    train_data = data[train_idx]
    test_data = data[test_idx]

    # Train/fine-tune
    model.fit(train_data)

    # Evaluate
    metrics = model.evaluate(test_data)
```

---

## 5. UNCERTAINTY ESTIMATION & SEÑALES DE TRADING

### 5.1 Objetivo

**"Emitir señales solo cuando esté SEGURO"**

Necesitamos medir la **confianza** del modelo en sus predicciones.

### 5.2 Métodos de Uncertainty Estimation

**A. Monte Carlo Dropout**
```python
class MarketGPT(nn.Module):
    def __init__(self, config):
        ...
        self.mc_dropout = nn.Dropout(0.1)

    def forward_with_uncertainty(self, x, n_samples=20):
        """
        Forward pass múltiple con dropout activo.
        Mayor varianza = mayor incertidumbre.
        """
        self.train()  # Mantener dropout activo

        predictions = []
        for _ in range(n_samples):
            pred = self.forward(x)
            predictions.append(pred)

        predictions = torch.stack(predictions)

        # Mean prediction
        mean_pred = predictions.mean(dim=0)

        # Uncertainty (std dev)
        uncertainty = predictions.std(dim=0)

        return mean_pred, uncertainty
```

**B. Ensemble de Modelos**
```python
# Entrenar 5 modelos con diferentes seeds
models = [
    MarketGPT(config, seed=i)
    for i in range(5)
]

# Predicción ensemble
predictions = [model(x) for model in models]
mean_pred = torch.stack(predictions).mean(dim=0)
uncertainty = torch.stack(predictions).std(dim=0)
```

**C. Temperatura en Softmax**
```python
def predict_with_temperature(logits, temperature=1.0):
    """
    Temperatura baja → más "confiado" (picos agudos)
    Temperatura alta → menos confiado (distribución suave)
    """
    scaled_logits = logits / temperature
    probs = F.softmax(scaled_logits, dim=-1)

    # Entropy como medida de incertidumbre
    entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)

    return probs, entropy
```

**D. Predictive Variance (para multi-step head)**
```python
# En lugar de predicción puntual, predecir distribución
class MultiStepHead(nn.Module):
    def __init__(self, d_model, n_steps, n_features):
        super().__init__()
        # Predecir media Y varianza
        self.mean_head = nn.Linear(d_model, n_steps * n_features)
        self.logvar_head = nn.Linear(d_model, n_steps * n_features)

    def forward(self, x):
        mean = self.mean_head(x)
        logvar = self.logvar_head(x)
        std = torch.exp(0.5 * logvar)

        return mean, std
```

### 5.3 Trading Signal Generation

```python
class TradingSignalGenerator:
    """
    Genera señales BUY/SELL/HOLD basadas en predicciones + uncertainty.
    """

    def __init__(
        self,
        confidence_threshold=0.7,  # Mínima confianza para operar
        min_return_threshold=0.02,  # Mínimo retorno esperado (2%)
    ):
        self.conf_threshold = confidence_threshold
        self.return_threshold = min_return_threshold

    def generate_signal(
        self,
        predicted_price,
        current_price,
        uncertainty
    ):
        """
        Args:
            predicted_price: Precio predicho para mañana
            current_price: Precio actual
            uncertainty: Std dev de la predicción

        Returns:
            signal: 'BUY', 'SELL', 'HOLD'
            confidence: 0-1
            expected_return: retorno esperado
        """
        # Calcular retorno esperado
        expected_return = (predicted_price - current_price) / current_price

        # Calcular confianza (inversa de uncertainty)
        # Normalizar uncertainty a [0, 1]
        confidence = 1.0 / (1.0 + uncertainty)

        # Decisión
        if confidence < self.conf_threshold:
            return 'HOLD', confidence, expected_return

        if abs(expected_return) < self.return_threshold:
            return 'HOLD', confidence, expected_return

        if expected_return > self.return_threshold:
            return 'BUY', confidence, expected_return
        else:
            return 'SELL', confidence, expected_return
```

**Uso:**
```python
# Durante inference
mean_pred, uncertainty = model.forward_with_uncertainty(x, n_samples=20)

signal_gen = TradingSignalGenerator(
    confidence_threshold=0.75,  # Solo operar si >75% confianza
    min_return_threshold=0.015  # Solo si esperamos >1.5% retorno
)

signal, conf, exp_ret = signal_gen.generate_signal(
    predicted_price=mean_pred,
    current_price=current_price,
    uncertainty=uncertainty
)

print(f"Signal: {signal}")
print(f"Confidence: {conf:.2%}")
print(f"Expected return: {exp_ret:.2%}")
```

### 5.4 Backtesting con Señales

```python
class SignalBacktester:
    """
    Evalúa performance de señales en histórico.
    """

    def backtest(self, signals, prices, initial_capital=100000):
        """
        Simula trading siguiendo señales.
        """
        capital = initial_capital
        position = 0  # 0 = no position, 1 = long, -1 = short
        trades = []

        for i, signal in enumerate(signals):
            current_price = prices[i]

            if signal == 'BUY' and position == 0:
                # Abrir posición long
                shares = capital / current_price
                position = 1
                entry_price = current_price
                trades.append({
                    'type': 'BUY',
                    'price': current_price,
                    'shares': shares,
                    'date': i
                })

            elif signal == 'SELL' and position == 1:
                # Cerrar posición
                exit_price = current_price
                pnl = (exit_price - entry_price) * shares
                capital += pnl
                position = 0
                trades.append({
                    'type': 'SELL',
                    'price': current_price,
                    'pnl': pnl,
                    'return': pnl / (entry_price * shares),
                    'date': i
                })

        # Métricas
        total_return = (capital - initial_capital) / initial_capital
        num_trades = len([t for t in trades if t['type'] == 'SELL'])

        if num_trades > 0:
            winning_trades = len([t for t in trades if t.get('pnl', 0) > 0])
            win_rate = winning_trades / num_trades
        else:
            win_rate = 0

        return {
            'final_capital': capital,
            'total_return': total_return,
            'num_trades': num_trades,
            'win_rate': win_rate,
            'trades': trades
        }
```

---

## 6. MONITOREO DE ENTRENAMIENTO EN TIEMPO REAL

### 6.1 Sistema de Progress Tracking

**Implementación con tqdm + logging + dashboard**

```python
from tqdm import tqdm
import time
from datetime import datetime, timedelta

class TrainingMonitor:
    """
    Monitorea progreso de entrenamiento con estimación de tiempo.
    """

    def __init__(self, total_steps, log_file='training_progress.log'):
        self.total_steps = total_steps
        self.current_step = 0
        self.start_time = time.time()
        self.log_file = log_file

        # Métricas
        self.loss_history = []
        self.step_times = []

        # Progress bar
        self.pbar = tqdm(total=total_steps, desc="Training")

    def update(self, loss, step=1):
        """Actualizar progreso."""
        self.current_step += step
        self.loss_history.append(loss)

        # Timing
        current_time = time.time()
        elapsed = current_time - self.start_time
        self.step_times.append(elapsed / self.current_step)

        # Estimación tiempo restante
        avg_step_time = sum(self.step_times[-100:]) / min(len(self.step_times), 100)
        remaining_steps = self.total_steps - self.current_step
        eta_seconds = avg_step_time * remaining_steps
        eta = timedelta(seconds=int(eta_seconds))

        # Progress %
        progress_pct = self.current_step / self.total_steps * 100

        # Update progress bar
        self.pbar.update(step)
        self.pbar.set_postfix({
            'loss': f'{loss:.4f}',
            'ETA': str(eta),
            'progress': f'{progress_pct:.1f}%'
        })

        # Log to file
        with open(self.log_file, 'a') as f:
            f.write(f"{datetime.now()},{self.current_step},{loss},{eta}\n")

    def get_stats(self):
        """Obtener estadísticas actuales."""
        elapsed = time.time() - self.start_time
        progress_pct = self.current_step / self.total_steps * 100

        avg_step_time = sum(self.step_times[-100:]) / min(len(self.step_times), 100)
        remaining_steps = self.total_steps - self.current_step
        eta_seconds = avg_step_time * remaining_steps

        return {
            'current_step': self.current_step,
            'total_steps': self.total_steps,
            'progress_pct': progress_pct,
            'elapsed_time': str(timedelta(seconds=int(elapsed))),
            'eta': str(timedelta(seconds=int(eta_seconds))),
            'avg_loss': sum(self.loss_history[-100:]) / min(len(self.loss_history), 100),
            'current_loss': self.loss_history[-1] if self.loss_history else 0
        }
```

### 6.2 Dashboard Web (Opcional)

**Con TensorBoard:**

```python
from torch.utils.tensorboard import SummaryWriter

class TensorBoardMonitor:
    def __init__(self, log_dir='runs/market_gpt'):
        self.writer = SummaryWriter(log_dir)

    def log_metrics(self, metrics, step):
        """Log métricas a TensorBoard."""
        for key, value in metrics.items():
            self.writer.add_scalar(key, value, step)

    def log_histogram(self, name, values, step):
        """Log histograma (ej: gradientes)."""
        self.writer.add_histogram(name, values, step)

    def log_text(self, name, text, step):
        """Log texto (ej: predicciones sample)."""
        self.writer.add_text(name, text, step)
```

**Uso:**
```bash
# En terminal separada
tensorboard --logdir=runs/market_gpt --port=6006

# Abrir en navegador
http://localhost:6006
```

### 6.3 Notificaciones Push

```python
import requests

class SlackNotifier:
    """Enviar notificaciones a Slack."""

    def __init__(self, webhook_url):
        self.webhook_url = webhook_url

    def send_notification(self, message):
        data = {'text': message}
        requests.post(self.webhook_url, json=data)

# Uso
notifier = SlackNotifier(webhook_url='YOUR_SLACK_WEBHOOK')

# Cada epoch
notifier.send_notification(
    f"Epoch {epoch} complete. Loss: {loss:.4f}. ETA: {eta}"
)

# Al terminar
notifier.send_notification(
    "Training complete! Check results in checkpoints/"
)
```

---

## 7. RESUMEN DE CAMBIOS NECESARIOS

Para implementar este diseño multi-mercado completo, necesito modificar:

### Nuevos archivos:
1. `universal_data_loader.py` - Descarga 600 assets diarios
2. `asset_universe.py` - Definición de assets por categoría
3. `hierarchical_embeddings.py` - Asset + Category embeddings
4. `uncertainty_estimation.py` - MC Dropout, ensembles
5. `trading_signals.py` - Generación de señales con confianza
6. `training_monitor.py` - Progress tracking en tiempo real
7. `category_finetuner.py` - Fine-tuning walk-forward por categoría

### Archivos a modificar:
1. `market_gpt.py` - Añadir hierarchical embeddings
2. `walk_forward_trainer.py` - Integrar monitoring + category fine-tuning
3. `train_bitcoin.py` → `train_universal.py` - Entrenamiento multi-market
4. Configs - Nueva config para 600 assets

### Estimación de tiempo:
- Implementación: 1-2 días
- Testing: 1 día
- Descarga de datos: 4-6 horas
- Entrenamiento pre-training: 7-10 días en A100
- Fine-tuning por categoría: 2-3 días c/u × 5 = 10-15 días

**TOTAL: ~3-4 semanas para proyecto completo**

---

¿Procedo con la implementación del sistema multi-mercado completo?
