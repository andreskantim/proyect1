"""
Verification script to check if Market GPT installation is correct.
Tests all components before running full training.
"""

import sys
import os
from pathlib import Path

print("="*80)
print("Market GPT Installation Verification")
print("="*80)

errors = []
warnings = []

# Test 1: Python version
print("\n1. Checking Python version...")
if sys.version_info < (3, 10):
    errors.append(f"Python 3.10+ required, found {sys.version}")
else:
    print(f"   ✓ Python {sys.version_info.major}.{sys.version_info.minor}")

# Test 2: Required packages
print("\n2. Checking required packages...")
required_packages = [
    'torch',
    'numpy',
    'pandas',
    'sklearn',
    'requests',
    'yfinance',
    'tqdm'
]

for pkg in required_packages:
    try:
        __import__(pkg)
        print(f"   ✓ {pkg}")
    except ImportError:
        errors.append(f"Package {pkg} not found. Run: pip install -r requirements_gpu.txt")

# Test 3: CUDA availability
print("\n3. Checking CUDA...")
try:
    import torch
    if torch.cuda.is_available():
        print(f"   ✓ CUDA available")
        print(f"   ✓ GPU: {torch.cuda.get_device_name(0)}")
        print(f"   ✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        warnings.append("CUDA not available. Training will use CPU (very slow).")
        print("   ⚠ CUDA not available (CPU mode)")
except Exception as e:
    errors.append(f"Error checking CUDA: {e}")

# Test 4: Project structure
print("\n4. Checking project structure...")
required_files = [
    'market_gpt.py',
    'tokenizer.py',
    'crypto_data_loader.py',
    'gold_data_loader.py',
    'walk_forward_trainer.py',
    'train_bitcoin.py',
    'test_gold_walkforward.py',
    'configs/bitcoin_gpt_small.json',
    'configs/quick_test.json',
    'slurm_scripts/train_bitcoin_a100.sh',
    'slurm_scripts/test_gold_a100.sh'
]

for file in required_files:
    filepath = Path(file)
    if filepath.exists():
        print(f"   ✓ {file}")
    else:
        errors.append(f"Missing file: {file}")

# Test 5: Import core modules
print("\n5. Testing core modules...")
try:
    from market_gpt import MarketGPT, MarketGPTConfig
    print("   ✓ market_gpt")
except Exception as e:
    errors.append(f"Error importing market_gpt: {e}")

try:
    from tokenizer import OHLCTokenizer, TokenizerConfig
    print("   ✓ tokenizer")
except Exception as e:
    errors.append(f"Error importing tokenizer: {e}")

try:
    from crypto_data_loader import CryptoDataLoader
    print("   ✓ crypto_data_loader")
except Exception as e:
    errors.append(f"Error importing crypto_data_loader: {e}")

try:
    from gold_data_loader import GoldDataLoader
    print("   ✓ gold_data_loader")
except Exception as e:
    errors.append(f"Error importing gold_data_loader: {e}")

try:
    from walk_forward_trainer import WalkForwardTrainer
    print("   ✓ walk_forward_trainer")
except Exception as e:
    errors.append(f"Error importing walk_forward_trainer: {e}")

# Test 6: Quick functionality test
print("\n6. Running quick functionality tests...")

try:
    import torch
    import numpy as np
    from market_gpt import MarketGPT, MarketGPTConfig

    # Create small model
    config = MarketGPTConfig(
        n_layers=2,
        d_model=128,
        n_heads=4,
        d_ff=512,
        context_length=64,
        vocab_size=256
    )
    model = MarketGPT(config)

    # Test forward pass
    batch_size = 2
    seq_len = 32
    dummy_input = torch.randint(0, 256, (batch_size, seq_len, 4))

    with torch.no_grad():
        next_logits, multi_preds = model(dummy_input, mode="both")

    assert next_logits.shape == (batch_size, seq_len, 256 * 4)
    assert multi_preds.shape == (batch_size, seq_len, config.n_steps_pred * 4)

    print("   ✓ Model forward pass")

except Exception as e:
    errors.append(f"Model test failed: {e}")

try:
    from tokenizer import OHLCTokenizer, TokenizerConfig
    import numpy as np

    # Create synthetic data
    ohlc = np.random.randn(1000, 4).cumsum(axis=0) + 100

    # Create and fit tokenizer
    tok_config = TokenizerConfig(vocab_size=128)
    tokenizer = OHLCTokenizer(tok_config)
    tokenizer.fit(ohlc)

    # Test encode/decode
    tokens = tokenizer.encode(ohlc[:100])
    reconstructed = tokenizer.decode(tokens, ohlc[:100])

    assert tokens.shape == (100, 4)
    assert reconstructed.shape == (100, 4)

    print("   ✓ Tokenizer encode/decode")

except Exception as e:
    errors.append(f"Tokenizer test failed: {e}")

# Test 7: Data loader connectivity
print("\n7. Testing data loader connectivity...")
try:
    from crypto_data_loader import CryptoDataLoader
    import requests

    # Test Binance API connectivity
    response = requests.get("https://api.binance.com/api/v3/ping", timeout=5)
    if response.status_code == 200:
        print("   ✓ Binance API accessible")
    else:
        warnings.append("Binance API not accessible. Data download may fail.")

except Exception as e:
    warnings.append(f"Could not test Binance API: {e}")

try:
    import yfinance as yf
    # Quick test
    ticker = yf.Ticker("GC=F")
    print("   ✓ Yahoo Finance (yfinance) accessible")
except Exception as e:
    warnings.append(f"Yahoo Finance test failed: {e}")

# Summary
print("\n" + "="*80)
print("VERIFICATION SUMMARY")
print("="*80)

if errors:
    print(f"\n❌ {len(errors)} ERROR(S) FOUND:")
    for i, error in enumerate(errors, 1):
        print(f"  {i}. {error}")

if warnings:
    print(f"\n⚠ {len(warnings)} WARNING(S):")
    for i, warning in enumerate(warnings, 1):
        print(f"  {i}. {warning}")

if not errors and not warnings:
    print("\n✓ All tests passed! Installation is correct.")
    print("\nNext steps:")
    print("  1. Quick test: python train_bitcoin.py --config configs/quick_test.json ...")
    print("  2. Full training: bash slurm_scripts/submit_jobs.sh")
elif not errors:
    print("\n✓ Installation is functional (with warnings).")
    print("  You can proceed but review warnings above.")
else:
    print("\n❌ Installation has errors. Please fix them before proceeding.")
    sys.exit(1)

print("\n" + "="*80)
