"""
Stock Predictor - Sistema de predicción de precios de acciones usando PyTorch.

Este paquete proporciona herramientas para descargar datos del mercado,
preprocesarlos, entrenar modelos de redes neuronales y hacer predicciones.
"""

__version__ = "1.0.0"
__author__ = "Stock Predictor Team"

from .data_loader import StockDataLoader
from .preprocessing import StockPreprocessor
from .model import StockLSTM, StockGRU, StockTransformer, get_model
from .train import StockTrainer
from .predict import StockPredictor
from .config import Config, get_quick_test_config, get_production_config, get_transformer_config

__all__ = [
    'StockDataLoader',
    'StockPreprocessor',
    'StockLSTM',
    'StockGRU',
    'StockTransformer',
    'get_model',
    'StockTrainer',
    'StockPredictor',
    'Config',
    'get_quick_test_config',
    'get_production_config',
    'get_transformer_config',
]
