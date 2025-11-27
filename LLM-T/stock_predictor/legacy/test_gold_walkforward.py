"""
Walk-forward testing script for Market GPT on Gold data.
Tests model trained on Bitcoin on Gold market.
"""

import argparse
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime

from market_gpt import MarketGPT, MarketGPTConfig
from tokenizer import OHLCTokenizer
from gold_data_loader import GoldDataLoader
from walk_forward_trainer import WalkForwardTrainer


def load_config(config_path: str) -> dict:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def main(args):
    print("="*80)
    print("MARKET GPT - GOLD WALK-FORWARD TESTING")
    print("="*80)
    print(f"Start time: {datetime.now()}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output dir: {args.output_dir}")
    print("="*80)

    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load configuration
    config = load_config(args.config)

    # ========================================
    # Step 1: Load Gold Data
    # ========================================
    print("\n" + "="*80)
    print("STEP 1: LOADING GOLD DATA")
    print("="*80)

    data_loader = GoldDataLoader(cache_dir="data/gold_cache")

    print("Downloading 20 years of gold data...")
    df = data_loader.download_gold_20_years(
        interval='1h',  # Highest resolution for 20 years
        use_cache=True
    )

    # Interpolate to 1-minute if requested
    if args.interpolate_to_1min:
        print("\nInterpolating to 1-minute resolution...")
        df = data_loader.interpolate_to_1min(df, method='linear')

    # Validate and clean
    is_valid, message = data_loader.validate_data(df)
    print(f"\nData validation: {message}")

    if not is_valid:
        df = data_loader.clean_data(df)

    # Extract OHLC
    ohlc_data = data_loader.get_ohlc_array(df)
    print(f"\nGold OHLC data shape: {ohlc_data.shape}")
    print(f"Price range: [{ohlc_data.min():.2f}, {ohlc_data.max():.2f}]")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    # ========================================
    # Step 2: Load Model and Tokenizer
    # ========================================
    print("\n" + "="*80)
    print("STEP 2: LOADING MODEL & TOKENIZER")
    print("="*80)

    # Load tokenizer
    checkpoint_dir = Path(args.checkpoint).parent
    tokenizer_path = checkpoint_dir / "tokenizer.pkl"

    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")

    tokenizer = OHLCTokenizer.load(str(tokenizer_path))
    print(f"Tokenizer loaded from {tokenizer_path}")

    # Load model configuration
    model_config = MarketGPTConfig(
        n_layers=config['model']['n_layers'],
        d_model=config['model']['d_model'],
        n_heads=config['model']['n_heads'],
        d_ff=config['model']['d_ff'],
        context_length=config['model']['context_length'],
        vocab_size=config['tokenizer']['vocab_size'],
        n_features=4,
        n_steps_pred=config['model'].get('n_steps_pred', 10),
        dropout=config['model']['dropout']
    )

    model = MarketGPT(model_config)
    print(f"\nModel created with {model.count_parameters()['total']:,} parameters")

    # Load checkpoint
    print(f"\nLoading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("  Model state loaded")
    else:
        model.load_state_dict(checkpoint)
        print("  Model state loaded (legacy format)")

    # ========================================
    # Step 3: Walk-Forward Testing
    # ========================================
    print("\n" + "="*80)
    print("STEP 3: WALK-FORWARD TESTING")
    print("="*80)

    trainer = WalkForwardTrainer(
        model=model,
        tokenizer=tokenizer,
        config=config,
        device=args.device
    )

    # Run walk-forward
    walkforward_config = config['training'].get('walkforward', {})

    results = trainer.walk_forward(
        test_data=ohlc_data,
        window_size=args.window_size or walkforward_config.get('window_size', 10080),
        fine_tune_epochs=args.fine_tune_epochs or walkforward_config.get('fine_tune_epochs', 3),
        batch_size=walkforward_config.get('batch_size', 8),
        learning_rate=walkforward_config.get('learning_rate', 1e-5),
        freeze_layers=walkforward_config.get('freeze_layers', 6),
        save_dir=args.output_dir
    )

    # ========================================
    # Step 4: Analyze Results
    # ========================================
    print("\n" + "="*80)
    print("STEP 4: RESULTS ANALYSIS")
    print("="*80)

    # Compute summary statistics
    improvements = [r['improvement'] for r in results]
    avg_improvement = np.mean(improvements)
    std_improvement = np.std(improvements)

    final_losses_before = [r['losses_before']['total'] for r in results]
    final_losses_after = [r['losses_after']['total'] for r in results]

    print(f"\nWalk-forward summary:")
    print(f"  Total windows: {len(results)}")
    print(f"  Average improvement: {avg_improvement:.6f} ± {std_improvement:.6f}")
    print(f"  Average loss before: {np.mean(final_losses_before):.6f}")
    print(f"  Average loss after: {np.mean(final_losses_after):.6f}")
    print(f"  Improvement rate: {sum(1 for i in improvements if i > 0) / len(improvements) * 100:.1f}%")

    # Save detailed results
    summary = {
        'config': config,
        'data_info': {
            'total_candles': len(ohlc_data),
            'num_windows': len(results),
            'window_size': args.window_size or walkforward_config.get('window_size', 10080),
        },
        'results_summary': {
            'avg_improvement': float(avg_improvement),
            'std_improvement': float(std_improvement),
            'avg_loss_before': float(np.mean(final_losses_before)),
            'avg_loss_after': float(np.mean(final_losses_after)),
            'improvement_rate': float(sum(1 for i in improvements if i > 0) / len(improvements)),
        },
        'window_results': results
    }

    with open(output_path / "walkforward_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {output_path}")

    print("\n" + "="*80)
    print("WALK-FORWARD TESTING COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Market GPT on Gold data")

    # Required arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration JSON file')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save results')

    # Walk-forward options
    parser.add_argument('--window_size', type=int, default=None,
                       help='Window size in candles (default: from config)')
    parser.add_argument('--fine_tune_epochs', type=int, default=None,
                       help='Epochs for fine-tuning (default: from config)')
    parser.add_argument('--interpolate_to_1min', action='store_true',
                       help='Interpolate gold data to 1-minute resolution')

    # Device options
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use')

    args = parser.parse_args()

    # Check CUDA
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("WARNING: CUDA requested but not available. Falling back to CPU.")
        args.device = 'cpu'

    main(args)
