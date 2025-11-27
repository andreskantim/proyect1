"""
Script de predicción operativa para el modelo de predicción de acciones.
Permite hacer predicciones sobre datos nuevos y en tiempo real.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Optional, List, Tuple
import os

from data_loader import StockDataLoader
from preprocessing import StockPreprocessor
from model import get_model


class StockPredictor:
    """Clase para realizar predicciones con el modelo entrenado."""

    def __init__(
        self,
        model_path: str,
        scaler_path: str,
        device: Optional[str] = None
    ):
        """
        Inicializa el predictor.

        Args:
            model_path: Ruta al modelo guardado
            scaler_path: Ruta al scaler guardado
            device: Dispositivo para inferencia
        """
        # Configurar dispositivo
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Cargar configuración del modelo
        checkpoint = torch.load(model_path, map_location=self.device)
        self.config = checkpoint['config']

        # Inicializar preprocesador
        self.preprocessor = StockPreprocessor(
            sequence_length=self.config['sequence_length']
        )
        self.preprocessor.load_scaler(scaler_path)

        # Cargar modelo
        self.model = None
        self._load_model(checkpoint)

        print(f"Predictor cargado correctamente")
        print(f"Modelo: {self.config['model_type'].upper()}")
        print(f"Dispositivo: {self.device}")

    def _load_model(self, checkpoint: dict):
        """Carga el modelo desde un checkpoint."""
        # Obtener el tamaño de entrada del scaler
        input_size = self.preprocessor.scaler.n_features_in_

        # Crear modelo
        if self.config['model_type'] == "transformer":
            self.model = get_model(
                model_type=self.config['model_type'],
                input_size=input_size,
                d_model=self.config['hidden_size'],
                nhead=8,
                num_layers=self.config['num_layers'],
                dropout=self.config['dropout']
            )
        else:
            self.model = get_model(
                model_type=self.config['model_type'],
                input_size=input_size,
                hidden_size=self.config['hidden_size'],
                num_layers=self.config['num_layers'],
                dropout=self.config['dropout']
            )

        # Cargar pesos
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()

    def predict_next_day(self, ticker: str, period: str = "1y") -> Tuple[float, pd.DataFrame]:
        """
        Predice el precio de cierre del próximo día.

        Args:
            ticker: Símbolo de la acción
            period: Periodo de datos históricos para la predicción

        Returns:
            Tupla (precio_predicho, datos_utilizados)
        """
        # Cargar datos recientes
        loader = StockDataLoader()
        data = loader.download_stock_data(ticker, period=period)

        # Preprocesar
        processed_data = self.preprocessor.prepare_features(data)

        # Verificar que tenemos suficientes datos
        if len(processed_data) < self.config['sequence_length']:
            raise ValueError(
                f"Se necesitan al menos {self.config['sequence_length']} días de datos. "
                f"Solo hay {len(processed_data)} disponibles."
            )

        # Normalizar
        normalized_data = self.preprocessor.normalize_data(processed_data, fit=False)

        # Tomar la última secuencia
        last_sequence = normalized_data[-self.config['sequence_length']:]
        last_sequence = torch.FloatTensor(last_sequence).unsqueeze(0).to(self.device)

        # Predecir
        with torch.no_grad():
            prediction = self.model(last_sequence)
            prediction = prediction.cpu().numpy()

        # Desnormalizar
        predicted_price = self.preprocessor.inverse_transform_predictions(prediction)[0]

        # Obtener el último precio real
        last_real_price = data['Close'].iloc[-1]
        change_percent = ((predicted_price - last_real_price) / last_real_price) * 100

        print(f"\n{'='*60}")
        print(f"Predicción para {ticker}")
        print(f"{'='*60}")
        print(f"Último precio real: ${last_real_price:.2f}")
        print(f"Precio predicho para mañana: ${predicted_price:.2f}")
        print(f"Cambio esperado: {change_percent:+.2f}%")
        print(f"{'='*60}")

        return predicted_price, data

    def predict_multiple_days(
        self,
        ticker: str,
        days_ahead: int = 7,
        period: str = "1y"
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Predice múltiples días en el futuro.

        Args:
            ticker: Símbolo de la acción
            days_ahead: Número de días a predecir
            period: Periodo de datos históricos

        Returns:
            Tupla (array_de_predicciones, datos_utilizados)
        """
        # Cargar datos
        loader = StockDataLoader()
        data = loader.download_stock_data(ticker, period=period)

        # Preprocesar
        processed_data = self.preprocessor.prepare_features(data)
        normalized_data = self.preprocessor.normalize_data(processed_data, fit=False)

        predictions = []
        current_sequence = normalized_data[-self.config['sequence_length']:].copy()

        print(f"\nPrediciendo {days_ahead} días para {ticker}...")

        for day in range(days_ahead):
            # Preparar secuencia
            sequence = torch.FloatTensor(current_sequence).unsqueeze(0).to(self.device)

            # Predecir
            with torch.no_grad():
                prediction = self.model(sequence)
                prediction = prediction.cpu().numpy()

            predictions.append(prediction[0, 0])

            # Actualizar la secuencia para la próxima predicción
            # Crear una nueva fila con la predicción
            new_row = current_sequence[-1].copy()
            new_row[3] = prediction[0, 0]  # Actualizar Close price (índice 3)

            # Agregar la nueva fila y eliminar la primera
            current_sequence = np.vstack([current_sequence[1:], new_row])

        # Desnormalizar predicciones
        predictions = np.array(predictions)
        predicted_prices = self.preprocessor.inverse_transform_predictions(predictions)

        return predicted_prices, data

    def backtest(
        self,
        ticker: str,
        test_days: int = 30,
        period: str = "2y"
    ) -> dict:
        """
        Realiza backtesting del modelo.

        Args:
            ticker: Símbolo de la acción
            test_days: Número de días para probar
            period: Periodo de datos históricos

        Returns:
            Diccionario con métricas de rendimiento
        """
        print(f"\nRealizando backtesting para {ticker} ({test_days} días)...")

        # Cargar datos
        loader = StockDataLoader()
        data = loader.download_stock_data(ticker, period=period)

        # Preprocesar
        processed_data = self.preprocessor.prepare_features(data)
        normalized_data = self.preprocessor.normalize_data(processed_data, fit=False)

        # Tomar los últimos test_days para prueba
        test_start = len(normalized_data) - test_days
        predictions = []
        actuals = []

        for i in range(test_start, len(normalized_data)):
            # Obtener secuencia
            if i < self.config['sequence_length']:
                continue

            sequence = normalized_data[i - self.config['sequence_length']:i]
            sequence = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)

            # Predecir
            with torch.no_grad():
                prediction = self.model(sequence)
                prediction = prediction.cpu().numpy()

            predictions.append(prediction[0, 0])
            actuals.append(normalized_data[i, 3])  # Close price

        # Desnormalizar
        predictions = np.array(predictions)
        actuals = np.array(actuals)

        predicted_prices = self.preprocessor.inverse_transform_predictions(predictions)
        actual_prices = self.preprocessor.inverse_transform_predictions(actuals)

        # Calcular métricas
        mse = np.mean((predicted_prices - actual_prices) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predicted_prices - actual_prices))
        mape = np.mean(np.abs((actual_prices - predicted_prices) / actual_prices)) * 100

        # Calcular dirección correcta
        actual_direction = np.sign(np.diff(actual_prices))
        predicted_direction = np.sign(predicted_prices[1:] - actual_prices[:-1])
        direction_accuracy = np.mean(actual_direction == predicted_direction) * 100

        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'direction_accuracy': direction_accuracy,
            'predictions': predicted_prices,
            'actuals': actual_prices
        }

        print(f"\n{'='*60}")
        print(f"Resultados del Backtesting")
        print(f"{'='*60}")
        print(f"RMSE: ${rmse:.2f}")
        print(f"MAE: ${mae:.2f}")
        print(f"MAPE: {mape:.2f}%")
        print(f"Precisión de dirección: {direction_accuracy:.2f}%")
        print(f"{'='*60}")

        return metrics

    def plot_prediction(
        self,
        ticker: str,
        days_ahead: int = 7,
        period: str = "3mo",
        save_path: Optional[str] = None
    ):
        """
        Visualiza las predicciones.

        Args:
            ticker: Símbolo de la acción
            days_ahead: Días a predecir
            period: Periodo de datos históricos a mostrar
            save_path: Ruta para guardar la figura
        """
        # Hacer predicciones
        predictions, data = self.predict_multiple_days(ticker, days_ahead, period="1y")

        # Tomar solo los últimos datos según el periodo
        if period == "1mo":
            display_days = 30
        elif period == "3mo":
            display_days = 90
        elif period == "6mo":
            display_days = 180
        else:
            display_days = len(data)

        data_to_plot = data.tail(display_days)

        # Crear fechas para las predicciones
        last_date = pd.to_datetime(data['Date'].iloc[-1])
        future_dates = [last_date + timedelta(days=i+1) for i in range(days_ahead)]

        # Plotear
        plt.figure(figsize=(15, 7))

        # Datos históricos
        plt.plot(data_to_plot['Date'], data_to_plot['Close'],
                label='Precio Histórico', linewidth=2, color='blue')

        # Predicciones
        plt.plot(future_dates, predictions,
                label='Predicciones', linewidth=2, color='red', marker='o')

        # Conectar último precio real con primera predicción
        plt.plot([data_to_plot['Date'].iloc[-1], future_dates[0]],
                [data_to_plot['Close'].iloc[-1], predictions[0]],
                linewidth=2, color='red', linestyle='--', alpha=0.5)

        plt.xlabel('Fecha', fontsize=12)
        plt.ylabel('Precio ($)', fontsize=12)
        plt.title(f'Predicción de precio para {ticker}', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Gráfica guardada en {save_path}")

        plt.show()

    def plot_backtest(self, ticker: str, test_days: int = 30, save_path: Optional[str] = None):
        """
        Visualiza los resultados del backtesting.

        Args:
            ticker: Símbolo de la acción
            test_days: Días de prueba
            save_path: Ruta para guardar la figura
        """
        metrics = self.backtest(ticker, test_days, period="2y")

        plt.figure(figsize=(15, 10))

        # Plot 1: Predicciones vs Reales
        plt.subplot(2, 1, 1)
        dates = range(len(metrics['actuals']))
        plt.plot(dates, metrics['actuals'], label='Precio Real', linewidth=2, color='blue')
        plt.plot(dates, metrics['predictions'], label='Precio Predicho', linewidth=2,
                color='red', alpha=0.7)
        plt.xlabel('Días')
        plt.ylabel('Precio ($)')
        plt.title(f'Backtesting para {ticker} - Predicciones vs Precios Reales')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot 2: Error
        plt.subplot(2, 1, 2)
        errors = metrics['predictions'] - metrics['actuals']
        plt.bar(dates, errors, color=['red' if e < 0 else 'green' for e in errors], alpha=0.6)
        plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        plt.xlabel('Días')
        plt.ylabel('Error ($)')
        plt.title('Error de Predicción (Predicho - Real)')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Gráfica guardada en {save_path}")

        plt.show()


def main():
    """Función principal para hacer predicciones."""
    # Configuración
    TICKER = "AAPL"
    MODEL_PATH = "models/best_model.pth"
    SCALER_PATH = "models/scaler.pkl"

    # Verificar que existan los archivos
    if not os.path.exists(MODEL_PATH):
        print(f"Error: No se encontró el modelo en {MODEL_PATH}")
        print("Primero debes entrenar el modelo ejecutando train.py")
        return

    # Crear predictor
    predictor = StockPredictor(
        model_path=MODEL_PATH,
        scaler_path=SCALER_PATH
    )

    # Predicción para el próximo día
    predictor.predict_next_day(TICKER)

    # Predicción para múltiples días
    print("\n" + "="*60)
    predictor.plot_prediction(TICKER, days_ahead=7, period="3mo",
                             save_path="logs/prediction.png")

    # Backtesting
    print("\n" + "="*60)
    predictor.plot_backtest(TICKER, test_days=30, save_path="logs/backtest.png")


if __name__ == "__main__":
    main()
