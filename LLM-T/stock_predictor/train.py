"""
Script de entrenamiento del modelo de predicción de acciones.
Incluye validación, early stopping, y guardado de checkpoints.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime
from typing import Optional, Dict, List

from data_loader import StockDataLoader
from preprocessing import StockPreprocessor
from model import get_model


class StockTrainer:
    """Clase para entrenar el modelo de predicción de acciones."""

    def __init__(
        self,
        model_type: str = "lstm",
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        sequence_length: int = 60,
        device: Optional[str] = None
    ):
        """
        Inicializa el entrenador.

        Args:
            model_type: Tipo de modelo ('lstm', 'gru', 'transformer')
            hidden_size: Tamaño de la capa oculta
            num_layers: Número de capas
            dropout: Tasa de dropout
            learning_rate: Tasa de aprendizaje
            batch_size: Tamaño del batch
            sequence_length: Longitud de la secuencia de entrada
            device: Dispositivo ('cuda' o 'cpu')
        """
        self.model_type = model_type
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.sequence_length = sequence_length

        # Configurar dispositivo
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        print(f"Usando dispositivo: {self.device}")

        # Inicializar componentes
        self.preprocessor = StockPreprocessor(sequence_length=sequence_length)
        self.model = None
        self.optimizer = None
        self.criterion = nn.MSELoss()

        # Métricas de entrenamiento
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')

    def prepare_data(self, ticker: str, period: str = "5y", train_ratio: float = 0.8):
        """
        Prepara los datos para entrenamiento.

        Args:
            ticker: Símbolo de la acción
            period: Periodo de datos históricos
            train_ratio: Proporción de datos para entrenamiento

        Returns:
            Tupla de DataLoaders (train_loader, val_loader)
        """
        print(f"\n{'='*60}")
        print(f"Preparando datos para {ticker}")
        print(f"{'='*60}")

        # Cargar datos
        loader = StockDataLoader()
        data = loader.download_stock_data(ticker, period=period)

        # Preprocesar datos
        X_train, y_train, X_val, y_val, _ = self.preprocessor.prepare_data_for_training(
            data, train_ratio=train_ratio
        )

        # Convertir a tensores de PyTorch
        X_train = torch.FloatTensor(X_train)
        y_train = torch.FloatTensor(y_train).unsqueeze(1)
        X_val = torch.FloatTensor(X_val)
        y_val = torch.FloatTensor(y_val).unsqueeze(1)

        # Crear datasets
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)

        # Crear dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )

        print(f"\nDatasets creados:")
        print(f"  Train: {len(train_dataset)} muestras")
        print(f"  Validation: {len(val_dataset)} muestras")
        print(f"  Batch size: {self.batch_size}")

        # Inicializar el modelo con el tamaño de entrada correcto
        input_size = X_train.shape[2]
        self._initialize_model(input_size)

        return train_loader, val_loader

    def _initialize_model(self, input_size: int):
        """Inicializa el modelo y el optimizador."""
        # Crear modelo
        if self.model_type == "transformer":
            self.model = get_model(
                model_type=self.model_type,
                input_size=input_size,
                d_model=self.hidden_size,
                nhead=8,
                num_layers=self.num_layers,
                dropout=self.dropout
            )
        else:
            self.model = get_model(
                model_type=self.model_type,
                input_size=input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                dropout=self.dropout
            )

        self.model = self.model.to(self.device)

        # Inicializar optimizador
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Imprimir información del modelo
        num_params = sum(p.numel() for p in self.model.parameters())
        print(f"\nModelo {self.model_type.upper()} inicializado")
        print(f"  Parámetros totales: {num_params:,}")
        print(f"  Input size: {input_size}")
        print(f"  Hidden size: {self.hidden_size}")
        print(f"  Num layers: {self.num_layers}")

    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        Entrena el modelo por una época.

        Args:
            train_loader: DataLoader de entrenamiento

        Returns:
            Pérdida promedio de la época
        """
        self.model.train()
        total_loss = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        return avg_loss

    def validate(self, val_loader: DataLoader) -> float:
        """
        Valida el modelo.

        Args:
            val_loader: DataLoader de validación

        Returns:
            Pérdida promedio de validación
        """
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                total_loss += loss.item()

        avg_loss = total_loss / len(val_loader)
        return avg_loss

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 100,
        patience: int = 10,
        save_dir: str = "models"
    ) -> Dict:
        """
        Entrena el modelo con early stopping.

        Args:
            train_loader: DataLoader de entrenamiento
            val_loader: DataLoader de validación
            epochs: Número máximo de épocas
            patience: Paciencia para early stopping
            save_dir: Directorio para guardar el modelo

        Returns:
            Diccionario con historial de entrenamiento
        """
        print(f"\n{'='*60}")
        print(f"Iniciando entrenamiento")
        print(f"{'='*60}")
        print(f"Épocas: {epochs}")
        print(f"Learning rate: {self.learning_rate}")
        print(f"Early stopping patience: {patience}")

        os.makedirs(save_dir, exist_ok=True)

        patience_counter = 0
        start_time = datetime.now()

        for epoch in range(epochs):
            # Entrenar
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)

            # Validar
            val_loss = self.validate(val_loader)
            self.val_losses.append(val_loss)

            # Calcular RMSE
            train_rmse = np.sqrt(train_loss)
            val_rmse = np.sqrt(val_loss)

            # Imprimir progreso
            print(f"Época [{epoch+1}/{epochs}] - "
                  f"Train Loss: {train_loss:.6f} (RMSE: {train_rmse:.6f}) - "
                  f"Val Loss: {val_loss:.6f} (RMSE: {val_rmse:.6f})")

            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                patience_counter = 0

                # Guardar mejor modelo
                self.save_model(os.path.join(save_dir, "best_model.pth"))
                print(f"  ✓ Mejor modelo guardado (val_loss: {val_loss:.6f})")
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"\nEarly stopping activado en época {epoch+1}")
                break

        end_time = datetime.now()
        training_time = (end_time - start_time).total_seconds()

        print(f"\n{'='*60}")
        print(f"Entrenamiento completado")
        print(f"{'='*60}")
        print(f"Tiempo total: {training_time:.2f} segundos")
        print(f"Mejor val_loss: {self.best_val_loss:.6f}")
        print(f"Mejor val_RMSE: {np.sqrt(self.best_val_loss):.6f}")

        # Guardar historial
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'epochs_trained': len(self.train_losses),
            'training_time': training_time,
            'config': {
                'model_type': self.model_type,
                'hidden_size': self.hidden_size,
                'num_layers': self.num_layers,
                'dropout': self.dropout,
                'learning_rate': self.learning_rate,
                'batch_size': self.batch_size,
                'sequence_length': self.sequence_length
            }
        }

        # Guardar historial
        with open(os.path.join(save_dir, "training_history.json"), 'w') as f:
            json.dump(history, f, indent=4)

        return history

    def save_model(self, filepath: str):
        """Guarda el modelo."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': {
                'model_type': self.model_type,
                'hidden_size': self.hidden_size,
                'num_layers': self.num_layers,
                'dropout': self.dropout,
                'sequence_length': self.sequence_length
            }
        }, filepath)

    def plot_training_history(self, save_path: Optional[str] = None):
        """
        Visualiza el historial de entrenamiento.

        Args:
            save_path: Ruta para guardar la figura
        """
        plt.figure(figsize=(12, 5))

        # Plot de pérdidas
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss', linewidth=2)
        plt.plot(self.val_losses, label='Validation Loss', linewidth=2)
        plt.xlabel('Época')
        plt.ylabel('MSE Loss')
        plt.title('Pérdida durante el entrenamiento')
        plt.legend()
        plt.grid(True)

        # Plot de RMSE
        plt.subplot(1, 2, 2)
        train_rmse = [np.sqrt(loss) for loss in self.train_losses]
        val_rmse = [np.sqrt(loss) for loss in self.val_losses]
        plt.plot(train_rmse, label='Train RMSE', linewidth=2)
        plt.plot(val_rmse, label='Validation RMSE', linewidth=2)
        plt.xlabel('Época')
        plt.ylabel('RMSE')
        plt.title('RMSE durante el entrenamiento')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Gráfica guardada en {save_path}")

        plt.show()


def main():
    """Función principal para entrenar el modelo."""
    # Configuración
    TICKER = "AAPL"  # Cambia esto por la acción que quieras
    MODEL_TYPE = "lstm"  # 'lstm', 'gru', o 'transformer'
    EPOCHS = 100
    PATIENCE = 15

    # Crear entrenador
    trainer = StockTrainer(
        model_type=MODEL_TYPE,
        hidden_size=128,
        num_layers=2,
        dropout=0.2,
        learning_rate=0.001,
        batch_size=32,
        sequence_length=60
    )

    # Preparar datos
    train_loader, val_loader = trainer.prepare_data(
        ticker=TICKER,
        period="5y",
        train_ratio=0.8
    )

    # Entrenar
    history = trainer.train(
        train_loader,
        val_loader,
        epochs=EPOCHS,
        patience=PATIENCE,
        save_dir="models"
    )

    # Guardar preprocesador
    trainer.preprocessor.save_scaler("models/scaler.pkl")

    # Visualizar resultados
    trainer.plot_training_history(save_path="logs/training_history.png")


if __name__ == "__main__":
    main()
