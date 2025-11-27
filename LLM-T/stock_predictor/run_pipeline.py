#!/usr/bin/env python3
"""
Script principal para ejecutar el pipeline completo de predicción de acciones.
Incluye entrenamiento y predicción en un solo comando.
"""

import argparse
import os
import sys

from train import StockTrainer
from predict import StockPredictor


def train_model(args):
    """Entrena un modelo con los parámetros especificados."""
    print("\n" + "="*60)
    print("INICIANDO ENTRENAMIENTO")
    print("="*60)

    # Crear entrenador
    trainer = StockTrainer(
        model_type=args.model_type,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length
    )

    # Preparar datos
    train_loader, val_loader = trainer.prepare_data(
        ticker=args.ticker,
        period=args.period,
        train_ratio=args.train_ratio
    )

    # Entrenar
    history = trainer.train(
        train_loader,
        val_loader,
        epochs=args.epochs,
        patience=args.patience,
        save_dir=args.save_dir
    )

    # Guardar preprocesador
    scaler_path = os.path.join(args.save_dir, "scaler.pkl")
    trainer.preprocessor.save_scaler(scaler_path)

    # Visualizar resultados
    if not args.no_plot:
        plot_path = os.path.join("logs", "training_history.png")
        trainer.plot_training_history(save_path=plot_path)

    print("\n" + "="*60)
    print("ENTRENAMIENTO COMPLETADO")
    print("="*60)
    print(f"Modelo guardado en: {os.path.join(args.save_dir, 'best_model.pth')}")
    print(f"Scaler guardado en: {scaler_path}")

    return True


def predict(args):
    """Realiza predicciones con un modelo entrenado."""
    print("\n" + "="*60)
    print("INICIANDO PREDICCIÓN")
    print("="*60)

    # Verificar que existan los archivos necesarios
    model_path = os.path.join(args.model_dir, "best_model.pth")
    scaler_path = os.path.join(args.model_dir, "scaler.pkl")

    if not os.path.exists(model_path):
        print(f"Error: No se encontró el modelo en {model_path}")
        print("Primero debes entrenar el modelo usando --mode train")
        return False

    if not os.path.exists(scaler_path):
        print(f"Error: No se encontró el scaler en {scaler_path}")
        return False

    # Crear predictor
    predictor = StockPredictor(
        model_path=model_path,
        scaler_path=scaler_path
    )

    # Predicción para el próximo día
    if args.next_day:
        predictor.predict_next_day(args.ticker)

    # Predicción para múltiples días
    if args.days_ahead > 0:
        print("\n" + "="*60)
        if not args.no_plot:
            plot_path = os.path.join("logs", f"{args.ticker}_prediction.png")
            predictor.plot_prediction(
                ticker=args.ticker,
                days_ahead=args.days_ahead,
                period=args.display_period,
                save_path=plot_path
            )
        else:
            predictions, _ = predictor.predict_multiple_days(
                ticker=args.ticker,
                days_ahead=args.days_ahead
            )
            print(f"\nPredicciones para los próximos {args.days_ahead} días:")
            for i, price in enumerate(predictions, 1):
                print(f"  Día {i}: ${price:.2f}")

    # Backtesting
    if args.backtest_days > 0:
        print("\n" + "="*60)
        if not args.no_plot:
            plot_path = os.path.join("logs", f"{args.ticker}_backtest.png")
            predictor.plot_backtest(
                ticker=args.ticker,
                test_days=args.backtest_days,
                save_path=plot_path
            )
        else:
            metrics = predictor.backtest(
                ticker=args.ticker,
                test_days=args.backtest_days
            )

    print("\n" + "="*60)
    print("PREDICCIÓN COMPLETADA")
    print("="*60)

    return True


def main():
    """Función principal."""
    parser = argparse.ArgumentParser(
        description="Stock Predictor - Sistema de predicción de precios de acciones",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:

  # Entrenar un modelo LSTM para Apple
  python run_pipeline.py --mode train --ticker AAPL --model-type lstm --epochs 100

  # Hacer predicciones con el modelo entrenado
  python run_pipeline.py --mode predict --ticker AAPL --next-day --days-ahead 7 --backtest-days 30

  # Entrenar y predecir en un solo comando
  python run_pipeline.py --mode both --ticker TSLA --epochs 50

  # Entrenar con configuración personalizada
  python run_pipeline.py --mode train --ticker GOOGL --hidden-size 256 --num-layers 3 --dropout 0.3
        """
    )

    # Argumentos generales
    parser.add_argument(
        "--mode",
        type=str,
        default="both",
        choices=["train", "predict", "both"],
        help="Modo de ejecución: train (entrenar), predict (predecir), both (ambos)"
    )
    parser.add_argument(
        "--ticker",
        type=str,
        default="AAPL",
        help="Símbolo de la acción (default: AAPL)"
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="No generar gráficas"
    )

    # Argumentos de entrenamiento
    train_group = parser.add_argument_group("Argumentos de entrenamiento")
    train_group.add_argument(
        "--model-type",
        type=str,
        default="lstm",
        choices=["lstm", "gru", "transformer"],
        help="Tipo de modelo (default: lstm)"
    )
    train_group.add_argument(
        "--period",
        type=str,
        default="5y",
        help="Periodo de datos históricos (default: 5y)"
    )
    train_group.add_argument(
        "--hidden-size",
        type=int,
        default=128,
        help="Tamaño de la capa oculta (default: 128)"
    )
    train_group.add_argument(
        "--num-layers",
        type=int,
        default=2,
        help="Número de capas (default: 2)"
    )
    train_group.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="Tasa de dropout (default: 0.2)"
    )
    train_group.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Tasa de aprendizaje (default: 0.001)"
    )
    train_group.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Tamaño del batch (default: 32)"
    )
    train_group.add_argument(
        "--sequence-length",
        type=int,
        default=60,
        help="Longitud de la secuencia (default: 60)"
    )
    train_group.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Número máximo de épocas (default: 100)"
    )
    train_group.add_argument(
        "--patience",
        type=int,
        default=15,
        help="Paciencia para early stopping (default: 15)"
    )
    train_group.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Proporción de datos para entrenamiento (default: 0.8)"
    )
    train_group.add_argument(
        "--save-dir",
        type=str,
        default="models",
        help="Directorio para guardar el modelo (default: models)"
    )

    # Argumentos de predicción
    pred_group = parser.add_argument_group("Argumentos de predicción")
    pred_group.add_argument(
        "--model-dir",
        type=str,
        default="models",
        help="Directorio del modelo entrenado (default: models)"
    )
    pred_group.add_argument(
        "--next-day",
        action="store_true",
        help="Predecir el precio del próximo día"
    )
    pred_group.add_argument(
        "--days-ahead",
        type=int,
        default=7,
        help="Número de días a predecir (default: 7)"
    )
    pred_group.add_argument(
        "--backtest-days",
        type=int,
        default=30,
        help="Número de días para backtesting (default: 30)"
    )
    pred_group.add_argument(
        "--display-period",
        type=str,
        default="3mo",
        help="Periodo a mostrar en gráficas (default: 3mo)"
    )

    args = parser.parse_args()

    # Crear directorios necesarios
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # Ejecutar según el modo
    success = True

    if args.mode in ["train", "both"]:
        success = train_model(args)
        if not success:
            sys.exit(1)

    if args.mode in ["predict", "both"]:
        # Si estamos en modo both, usar los parámetros de entrenamiento
        if args.mode == "both":
            args.model_dir = args.save_dir
            args.next_day = True

        success = predict(args)
        if not success:
            sys.exit(1)

    print("\n" + "="*60)
    print("PIPELINE COMPLETADO EXITOSAMENTE")
    print("="*60)


if __name__ == "__main__":
    main()
