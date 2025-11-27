"""
Módulo para obtener datos del mercado de valores.
Utiliza yfinance para descargar datos históricos.
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List
import os


class StockDataLoader:
    """Clase para cargar datos del mercado de valores."""

    def __init__(self, data_dir: str = "data"):
        """
        Inicializa el cargador de datos.

        Args:
            data_dir: Directorio donde se guardarán los datos descargados
        """
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

    def download_stock_data(
        self,
        ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        period: str = "5y"
    ) -> pd.DataFrame:
        """
        Descarga datos históricos de una acción.

        Args:
            ticker: Símbolo de la acción (ej: 'AAPL', 'GOOGL')
            start_date: Fecha de inicio en formato 'YYYY-MM-DD'
            end_date: Fecha de fin en formato 'YYYY-MM-DD'
            period: Periodo de tiempo si no se especifican fechas
                   (ej: '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')

        Returns:
            DataFrame con los datos históricos
        """
        print(f"Descargando datos para {ticker}...")

        if start_date and end_date:
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        else:
            data = yf.download(ticker, period=period, progress=False)

        if data.empty:
            raise ValueError(f"No se pudieron descargar datos para {ticker}")

        # Resetear el índice para tener la fecha como columna
        data.reset_index(inplace=True)

        print(f"Descargados {len(data)} registros para {ticker}")
        return data

    def download_multiple_stocks(
        self,
        tickers: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        period: str = "5y"
    ) -> dict:
        """
        Descarga datos para múltiples acciones.

        Args:
            tickers: Lista de símbolos de acciones
            start_date: Fecha de inicio
            end_date: Fecha de fin
            period: Periodo de tiempo

        Returns:
            Diccionario con ticker como clave y DataFrame como valor
        """
        stock_data = {}

        for ticker in tickers:
            try:
                data = self.download_stock_data(ticker, start_date, end_date, period)
                stock_data[ticker] = data
            except Exception as e:
                print(f"Error descargando {ticker}: {str(e)}")

        return stock_data

    def save_data(self, data: pd.DataFrame, filename: str):
        """
        Guarda los datos en un archivo CSV.

        Args:
            data: DataFrame con los datos
            filename: Nombre del archivo (sin extensión)
        """
        filepath = os.path.join(self.data_dir, f"{filename}.csv")
        data.to_csv(filepath, index=False)
        print(f"Datos guardados en {filepath}")

    def load_data(self, filename: str) -> pd.DataFrame:
        """
        Carga datos desde un archivo CSV.

        Args:
            filename: Nombre del archivo (sin extensión)

        Returns:
            DataFrame con los datos
        """
        filepath = os.path.join(self.data_dir, f"{filename}.csv")

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Archivo no encontrado: {filepath}")

        data = pd.read_csv(filepath)

        # Convertir la columna Date a datetime si existe
        if 'Date' in data.columns:
            data['Date'] = pd.to_datetime(data['Date'])

        print(f"Datos cargados desde {filepath}")
        return data

    def get_stock_info(self, ticker: str) -> dict:
        """
        Obtiene información de la acción.

        Args:
            ticker: Símbolo de la acción

        Returns:
            Diccionario con información de la acción
        """
        stock = yf.Ticker(ticker)
        return stock.info


if __name__ == "__main__":
    # Ejemplo de uso
    loader = StockDataLoader()

    # Descargar datos de Apple
    data = loader.download_stock_data("AAPL", period="2y")
    print("\nPrimeras filas:")
    print(data.head())
    print("\nÚltimas filas:")
    print(data.tail())

    # Guardar los datos
    loader.save_data(data, "AAPL_2y")
