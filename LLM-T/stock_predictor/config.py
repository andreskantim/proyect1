"""
Archivo de configuración para el proyecto de predicción de acciones.
Centraliza todos los parámetros configurables.
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class DataConfig:
    """Configuración para la carga de datos."""
    ticker: str = "AAPL"  # Símbolo de la acción
    period: str = "5y"  # Periodo de datos históricos
    start_date: Optional[str] = None  # Fecha de inicio (formato: 'YYYY-MM-DD')
    end_date: Optional[str] = None  # Fecha de fin (formato: 'YYYY-MM-DD')
    train_ratio: float = 0.8  # Proporción de datos para entrenamiento


@dataclass
class PreprocessingConfig:
    """Configuración para el preprocesamiento."""
    sequence_length: int = 60  # Número de días históricos para la entrada


@dataclass
class ModelConfig:
    """Configuración del modelo."""
    model_type: str = "lstm"  # Tipo de modelo: 'lstm', 'gru', 'transformer'
    hidden_size: int = 128  # Tamaño de la capa oculta
    num_layers: int = 2  # Número de capas
    dropout: float = 0.2  # Tasa de dropout
    output_size: int = 1  # Tamaño de salida

    # Configuración específica para Transformer
    d_model: int = 128  # Dimensión del modelo transformer
    nhead: int = 8  # Número de cabezas de atención
    dim_feedforward: int = 512  # Dimensión feedforward


@dataclass
class TrainingConfig:
    """Configuración del entrenamiento."""
    batch_size: int = 32  # Tamaño del batch
    learning_rate: float = 0.001  # Tasa de aprendizaje
    epochs: int = 100  # Número máximo de épocas
    patience: int = 15  # Paciencia para early stopping
    device: Optional[str] = None  # Dispositivo ('cuda', 'cpu', o None para auto)


@dataclass
class PathConfig:
    """Configuración de rutas."""
    data_dir: str = "data"  # Directorio de datos
    models_dir: str = "models"  # Directorio de modelos
    logs_dir: str = "logs"  # Directorio de logs
    notebooks_dir: str = "notebooks"  # Directorio de notebooks

    # Archivos específicos
    best_model_path: str = os.path.join(models_dir, "best_model.pth")
    scaler_path: str = os.path.join(models_dir, "scaler.pkl")
    training_history_path: str = os.path.join(models_dir, "training_history.json")


@dataclass
class Config:
    """Configuración principal que agrupa todas las configuraciones."""
    data: DataConfig = DataConfig()
    preprocessing: PreprocessingConfig = PreprocessingConfig()
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    paths: PathConfig = PathConfig()

    def __post_init__(self):
        """Crea los directorios necesarios si no existen."""
        os.makedirs(self.paths.data_dir, exist_ok=True)
        os.makedirs(self.paths.models_dir, exist_ok=True)
        os.makedirs(self.paths.logs_dir, exist_ok=True)
        os.makedirs(self.paths.notebooks_dir, exist_ok=True)

    def print_config(self):
        """Imprime la configuración actual."""
        print("="*60)
        print("CONFIGURACIÓN DEL PROYECTO")
        print("="*60)

        print("\n[DATOS]")
        print(f"  Ticker: {self.data.ticker}")
        print(f"  Periodo: {self.data.period}")
        print(f"  Train ratio: {self.data.train_ratio}")

        print("\n[PREPROCESAMIENTO]")
        print(f"  Sequence length: {self.preprocessing.sequence_length}")

        print("\n[MODELO]")
        print(f"  Tipo: {self.model.model_type}")
        print(f"  Hidden size: {self.model.hidden_size}")
        print(f"  Num layers: {self.model.num_layers}")
        print(f"  Dropout: {self.model.dropout}")

        print("\n[ENTRENAMIENTO]")
        print(f"  Batch size: {self.training.batch_size}")
        print(f"  Learning rate: {self.training.learning_rate}")
        print(f"  Epochs: {self.training.epochs}")
        print(f"  Patience: {self.training.patience}")

        print("\n[RUTAS]")
        print(f"  Data dir: {self.paths.data_dir}")
        print(f"  Models dir: {self.paths.models_dir}")
        print(f"  Logs dir: {self.paths.logs_dir}")

        print("="*60)


# Configuraciones predefinidas para diferentes casos de uso

def get_quick_test_config() -> Config:
    """Configuración para pruebas rápidas."""
    config = Config()
    config.data.period = "1y"
    config.training.epochs = 20
    config.training.patience = 5
    config.training.batch_size = 64
    return config


def get_production_config() -> Config:
    """Configuración para producción con mejor rendimiento."""
    config = Config()
    config.data.period = "10y"
    config.preprocessing.sequence_length = 90
    config.model.hidden_size = 256
    config.model.num_layers = 3
    config.model.dropout = 0.3
    config.training.epochs = 200
    config.training.patience = 20
    config.training.learning_rate = 0.0005
    return config


def get_transformer_config() -> Config:
    """Configuración optimizada para modelo Transformer."""
    config = Config()
    config.model.model_type = "transformer"
    config.model.d_model = 128
    config.model.nhead = 8
    config.model.num_layers = 3
    config.model.dim_feedforward = 512
    config.training.learning_rate = 0.0001
    return config


# Configuración por defecto
default_config = Config()


if __name__ == "__main__":
    # Mostrar configuración por defecto
    config = Config()
    config.print_config()

    print("\n\n" + "="*60)
    print("CONFIGURACIÓN PARA PRUEBAS RÁPIDAS")
    print("="*60)
    quick_config = get_quick_test_config()
    quick_config.print_config()

    print("\n\n" + "="*60)
    print("CONFIGURACIÓN PARA PRODUCCIÓN")
    print("="*60)
    prod_config = get_production_config()
    prod_config.print_config()
