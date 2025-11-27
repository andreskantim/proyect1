"""
Módulo que define la arquitectura de la Red Neuronal para predicción de acciones.
Utiliza LSTM (Long Short-Term Memory) para capturar patrones temporales.
"""

import torch
import torch.nn as nn


class StockLSTM(nn.Module):
    """
    Red Neuronal LSTM para predicción de precios de acciones.

    Arquitectura:
    - Múltiples capas LSTM apiladas
    - Dropout para regularización
    - Capas totalmente conectadas para la salida
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_size: int = 1
    ):
        """
        Inicializa el modelo LSTM.

        Args:
            input_size: Número de características de entrada
            hidden_size: Tamaño de la capa oculta LSTM
            num_layers: Número de capas LSTM apiladas
            dropout: Tasa de dropout para regularización
            output_size: Tamaño de la salida (default: 1 para un valor de predicción)
        """
        super(StockLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Capas LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Dropout adicional
        self.dropout = nn.Dropout(dropout)

        # Capas totalmente conectadas
        self.fc1 = nn.Linear(hidden_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, output_size)

    def forward(self, x):
        """
        Forward pass del modelo.

        Args:
            x: Tensor de entrada con forma (batch_size, sequence_length, input_size)

        Returns:
            Tensor de salida con forma (batch_size, output_size)
        """
        # Inicializar estados ocultos
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # LSTM forward pass
        # out: (batch_size, sequence_length, hidden_size)
        out, _ = self.lstm(x, (h0, c0))

        # Tomar solo la última salida de la secuencia
        out = out[:, -1, :]

        # Dropout
        out = self.dropout(out)

        # Capas fully connected
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)

        return out


class StockGRU(nn.Module):
    """
    Red Neuronal GRU (Gated Recurrent Unit) para predicción de precios.

    Alternativa más ligera al LSTM, puede ser más rápida de entrenar.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_size: int = 1
    ):
        """
        Inicializa el modelo GRU.

        Args:
            input_size: Número de características de entrada
            hidden_size: Tamaño de la capa oculta GRU
            num_layers: Número de capas GRU apiladas
            dropout: Tasa de dropout para regularización
            output_size: Tamaño de la salida
        """
        super(StockGRU, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Capas GRU
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Dropout adicional
        self.dropout = nn.Dropout(dropout)

        # Capas totalmente conectadas
        self.fc1 = nn.Linear(hidden_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, output_size)

    def forward(self, x):
        """
        Forward pass del modelo.

        Args:
            x: Tensor de entrada con forma (batch_size, sequence_length, input_size)

        Returns:
            Tensor de salida con forma (batch_size, output_size)
        """
        # Inicializar estado oculto
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # GRU forward pass
        out, _ = self.gru(x, h0)

        # Tomar solo la última salida de la secuencia
        out = out[:, -1, :]

        # Dropout
        out = self.dropout(out)

        # Capas fully connected
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)

        return out


class StockTransformer(nn.Module):
    """
    Modelo Transformer para predicción de precios de acciones.

    Utiliza mecanismos de atención para capturar dependencias a largo plazo.
    """

    def __init__(
        self,
        input_size: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.2,
        output_size: int = 1
    ):
        """
        Inicializa el modelo Transformer.

        Args:
            input_size: Número de características de entrada
            d_model: Dimensión del modelo
            nhead: Número de cabezas de atención
            num_layers: Número de capas del encoder
            dim_feedforward: Dimensión de la capa feedforward
            dropout: Tasa de dropout
            output_size: Tamaño de la salida
        """
        super(StockTransformer, self).__init__()

        self.d_model = d_model

        # Capa de proyección de entrada
        self.input_projection = nn.Linear(input_size, d_model)

        # Positional encoding (embedding de posición)
        self.pos_encoder = nn.Parameter(torch.randn(1, 500, d_model))

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Capas de salida
        self.fc1 = nn.Linear(d_model, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, output_size)

    def forward(self, x):
        """
        Forward pass del modelo.

        Args:
            x: Tensor de entrada con forma (batch_size, sequence_length, input_size)

        Returns:
            Tensor de salida con forma (batch_size, output_size)
        """
        # Proyectar entrada a d_model dimensiones
        x = self.input_projection(x)

        # Añadir positional encoding
        seq_len = x.size(1)
        x = x + self.pos_encoder[:, :seq_len, :]

        # Transformer encoder
        x = self.transformer_encoder(x)

        # Tomar la última salida de la secuencia
        x = x[:, -1, :]

        # Capas de salida
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x


def get_model(model_type: str = "lstm", **kwargs):
    """
    Función factory para crear modelos.

    Args:
        model_type: Tipo de modelo ('lstm', 'gru', o 'transformer')
        **kwargs: Argumentos adicionales para el modelo

    Returns:
        Instancia del modelo seleccionado
    """
    model_type = model_type.lower()

    if model_type == "lstm":
        return StockLSTM(**kwargs)
    elif model_type == "gru":
        return StockGRU(**kwargs)
    elif model_type == "transformer":
        return StockTransformer(**kwargs)
    else:
        raise ValueError(f"Tipo de modelo no soportado: {model_type}")


if __name__ == "__main__":
    # Ejemplo de uso
    batch_size = 32
    sequence_length = 60
    input_size = 21  # Número de características

    # Crear datos de ejemplo
    x = torch.randn(batch_size, sequence_length, input_size)

    # Probar modelo LSTM
    print("=== Modelo LSTM ===")
    lstm_model = StockLSTM(input_size=input_size, hidden_size=128, num_layers=2)
    lstm_output = lstm_model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {lstm_output.shape}")
    print(f"Número de parámetros: {sum(p.numel() for p in lstm_model.parameters())}")

    # Probar modelo GRU
    print("\n=== Modelo GRU ===")
    gru_model = StockGRU(input_size=input_size, hidden_size=128, num_layers=2)
    gru_output = gru_model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {gru_output.shape}")
    print(f"Número de parámetros: {sum(p.numel() for p in gru_model.parameters())}")

    # Probar modelo Transformer
    print("\n=== Modelo Transformer ===")
    transformer_model = StockTransformer(input_size=input_size, d_model=128, nhead=8, num_layers=3)
    transformer_output = transformer_model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {transformer_output.shape}")
    print(f"Número de parámetros: {sum(p.numel() for p in transformer_model.parameters())}")
