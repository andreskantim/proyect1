"""
Módulo para preprocesar datos del mercado de valores.
Incluye normalización, creación de características técnicas y preparación de secuencias.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, Optional
import pickle


class StockPreprocessor:
    """Clase para preprocesar datos de acciones."""

    def __init__(self, sequence_length: int = 60):
        """
        Inicializa el preprocesador.

        Args:
            sequence_length: Número de días históricos para usar como entrada
        """
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.feature_columns = None

    def add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Añade indicadores técnicos a los datos.

        Args:
            data: DataFrame con datos del mercado

        Returns:
            DataFrame con indicadores técnicos añadidos
        """
        df = data.copy()

        # Moving Averages
        df['MA_7'] = df['Close'].rolling(window=7).mean()
        df['MA_21'] = df['Close'].rolling(window=21).mean()
        df['MA_50'] = df['Close'].rolling(window=50).mean()

        # Exponential Moving Averages
        df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()

        # MACD (Moving Average Convergence Divergence)
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

        # RSI (Relative Strength Index)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)

        # Volatility
        df['Volatility'] = df['Close'].rolling(window=10).std()

        # Price Rate of Change
        df['ROC'] = df['Close'].pct_change(periods=10) * 100

        # Volume indicators
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']

        # Price change
        df['Price_Change'] = df['Close'].pct_change()
        df['Price_Change_1d'] = df['Close'].pct_change(1)
        df['Price_Change_5d'] = df['Close'].pct_change(5)

        return df

    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepara las características para el modelo.

        Args:
            data: DataFrame con datos del mercado

        Returns:
            DataFrame con características preparadas
        """
        df = data.copy()

        # Añadir indicadores técnicos
        df = self.add_technical_indicators(df)

        # Eliminar filas con valores NaN
        df = df.dropna()

        # Seleccionar las características a utilizar
        self.feature_columns = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'MA_7', 'MA_21', 'MA_50',
            'EMA_12', 'EMA_26', 'MACD', 'Signal_Line',
            'RSI', 'BB_Middle', 'BB_Upper', 'BB_Lower',
            'Volatility', 'ROC', 'Volume_Ratio',
            'Price_Change_1d', 'Price_Change_5d'
        ]

        return df[['Date'] + self.feature_columns]

    def normalize_data(self, data: pd.DataFrame, fit: bool = True) -> np.ndarray:
        """
        Normaliza los datos usando MinMaxScaler.

        Args:
            data: DataFrame con características
            fit: Si es True, ajusta el scaler. Si es False, usa el scaler existente

        Returns:
            Array numpy con datos normalizados
        """
        # Excluir la columna Date
        feature_data = data[self.feature_columns].values

        if fit:
            normalized_data = self.scaler.fit_transform(feature_data)
        else:
            normalized_data = self.scaler.transform(feature_data)

        return normalized_data

    def create_sequences(
        self,
        data: np.ndarray,
        target_column_idx: int = 3  # Close price
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Crea secuencias para el modelo LSTM.

        Args:
            data: Array numpy con datos normalizados
            target_column_idx: Índice de la columna objetivo (default: 3 para Close)

        Returns:
            Tupla (X, y) donde X son las secuencias de entrada e y los valores objetivo
        """
        X, y = [], []

        for i in range(self.sequence_length, len(data)):
            # Secuencia de entrada: sequence_length días de todas las características
            X.append(data[i - self.sequence_length:i])

            # Valor objetivo: precio de cierre del día siguiente
            y.append(data[i, target_column_idx])

        return np.array(X), np.array(y)

    def prepare_data_for_training(
        self,
        data: pd.DataFrame,
        train_ratio: float = 0.8
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Prepara los datos completamente para entrenamiento.

        Args:
            data: DataFrame con datos crudos
            train_ratio: Proporción de datos para entrenamiento

        Returns:
            Tupla (X_train, y_train, X_test, y_test, processed_data)
        """
        # Preparar características
        processed_data = self.prepare_features(data)

        # Normalizar datos
        normalized_data = self.normalize_data(processed_data, fit=True)

        # Crear secuencias
        X, y = self.create_sequences(normalized_data)

        # Dividir en train/test
        train_size = int(len(X) * train_ratio)

        X_train = X[:train_size]
        y_train = y[:train_size]
        X_test = X[train_size:]
        y_test = y[train_size:]

        print(f"Datos de entrenamiento: {X_train.shape}")
        print(f"Datos de prueba: {X_test.shape}")

        return X_train, y_train, X_test, y_test, processed_data

    def inverse_transform_predictions(self, predictions: np.ndarray) -> np.ndarray:
        """
        Desnormaliza las predicciones.

        Args:
            predictions: Array con predicciones normalizadas

        Returns:
            Array con predicciones en escala original
        """
        # Crear un array dummy con el mismo número de características
        dummy = np.zeros((len(predictions), len(self.feature_columns)))

        # Colocar las predicciones en la columna de Close (índice 3)
        dummy[:, 3] = predictions.flatten()

        # Invertir la transformación
        inverse = self.scaler.inverse_transform(dummy)

        # Retornar solo la columna de Close
        return inverse[:, 3]

    def save_scaler(self, filepath: str):
        """
        Guarda el scaler entrenado.

        Args:
            filepath: Ruta donde guardar el scaler
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"Scaler guardado en {filepath}")

    def load_scaler(self, filepath: str):
        """
        Carga un scaler previamente entrenado.

        Args:
            filepath: Ruta del scaler guardado
        """
        with open(filepath, 'rb') as f:
            self.scaler = pickle.load(f)
        print(f"Scaler cargado desde {filepath}")


if __name__ == "__main__":
    # Ejemplo de uso
    from data_loader import StockDataLoader

    # Cargar datos
    loader = StockDataLoader()
    data = loader.download_stock_data("AAPL", period="2y")

    # Preprocesar
    preprocessor = StockPreprocessor(sequence_length=60)
    X_train, y_train, X_test, y_test, processed = preprocessor.prepare_data_for_training(data)

    print("\nForma de los datos procesados:")
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")
