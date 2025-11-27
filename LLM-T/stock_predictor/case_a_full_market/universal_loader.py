"""
Universal Market Data Loader for Case A: Full Market (600 assets)

Downloads and processes data from all major markets:
- US Stocks: 300 assets (SP500 top companies)
- European Stocks: 150 assets (major EU indices)
- Emerging Markets: 50 assets (ETFs + major companies)
- Commodities: 30 assets (metals, energy, agriculture)
- Crypto: 70 assets (top cryptocurrencies)

Total: 600 assets across 5 categories
Historical: ~20 years of daily data where available
"""

import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import time
from tqdm import tqdm
import json


class UniversalMarketLoader:
    """
    Loads 600 assets from all major markets for Case A.
    """

    # Category 0: US Stocks (300 assets - top SP500)
    US_STOCKS = [
        # Mega caps (50)
        'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B', 'UNH',
        'JNJ', 'V', 'WMT', 'JPM', 'MA', 'PG', 'XOM', 'HD', 'CVX', 'MRK',
        'ABBV', 'KO', 'PEP', 'COST', 'AVGO', 'TMO', 'MCD', 'CSCO', 'ACN', 'DHR',
        'ABT', 'NKE', 'TXN', 'NEE', 'DIS', 'VZ', 'CMCSA', 'ADBE', 'PM', 'NFLX',
        'CRM', 'INTC', 'AMD', 'QCOM', 'HON', 'UNP', 'LOW', 'IBM', 'BA', 'CAT',

        # Large caps (100)
        'GE', 'RTX', 'UPS', 'SBUX', 'AMGN', 'SPGI', 'BLK', 'AXP', 'DE', 'MMM',
        'ISRG', 'GS', 'TJX', 'BKNG', 'GILD', 'MDLZ', 'LMT', 'MO', 'AMT', 'CVS',
        'PLD', 'SYK', 'CI', 'C', 'ZTS', 'CB', 'NOW', 'BDX', 'REGN', 'TGT',
        'ADI', 'VRTX', 'MU', 'AMAT', 'LRCX', 'KLAC', 'NXPI', 'MRVL', 'MCHP', 'FTNT',
        'PANW', 'SNPS', 'CDNS', 'TEAM', 'WDAY', 'DXCM', 'IDXX', 'ILMN', 'BIIB', 'MRNA',
        'PYPL', 'SQ', 'COIN', 'SHOP', 'UBER', 'LYFT', 'ABNB', 'DASH', 'RIVN', 'LCID',
        'F', 'GM', 'TSLA', 'NIO', 'LI', 'XPEV', 'RIVN', 'LCID', 'GOEV', 'FSR',
        'AAL', 'DAL', 'UAL', 'LUV', 'ALK', 'JBLU', 'SAVE', 'HA', 'SKYW', 'MESA',
        'CCL', 'RCL', 'NCLH', 'MAR', 'HLT', 'IHG', 'H', 'WH', 'EXPE', 'BKNG',
        'DIS', 'CMCSA', 'NFLX', 'T', 'TMUS', 'VZ', 'DISH', 'PARA', 'WBD', 'DISCA',
        'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'PXD', 'MPC', 'PSX', 'VLO', 'HES',

        # Mid-large caps (150)
        'KMI', 'WMB', 'OKE', 'LNG', 'TRGP', 'DTR', 'WES', 'EPD', 'ET', 'PAA',
        'FCX', 'NEM', 'GOLD', 'AA', 'STLD', 'NUE', 'CLF', 'X', 'MT', 'VALE',
        'CAT', 'DE', 'PCAR', 'CMI', 'ITW', 'EMR', 'ETN', 'PH', 'ROK', 'DOV',
        'JCI', 'CARR', 'OTIS', 'WM', 'RSG', 'WCN', 'GFL', 'CWST', 'MEG', 'SRCL',
        'NEE', 'DUK', 'SO', 'D', 'EXC', 'AEP', 'SRE', 'PCG', 'XEL', 'ED',
        'EIX', 'WEC', 'ES', 'FE', 'AEE', 'CMS', 'DTE', 'PEG', 'PPL', 'CNP',
        'AWK', 'WTRG', 'SJW', 'MSEX', 'CWT', 'YORW', 'ARTNA', 'CDZI', 'GWRS', 'WTTR',
        'PG', 'KO', 'PEP', 'CL', 'KMB', 'CHD', 'CLX', 'CPB', 'GIS', 'K',
        'HSY', 'MDLZ', 'MNST', 'KDP', 'TSN', 'HRL', 'CAG', 'SJM', 'MKC', 'LW',
        'COST', 'WMT', 'TGT', 'DG', 'DLTR', 'BIG', 'FIVE', 'OLLI', 'ROST', 'TJX',
        'HD', 'LOW', 'WSM', 'BBWI', 'M', 'KSS', 'JWN', 'BBBY', 'DDS', 'BURL',
        'AMZN', 'EBAY', 'ETSY', 'W', 'CVNA', 'PRTS', 'KMX', 'AN', 'LAD', 'ABG',
        'AAPL', 'DELL', 'HPQ', 'HPE', 'NTAP', 'STX', 'WDC', 'PSTG', 'SMCI', 'SCKT',
        'GOOGL', 'META', 'SNAP', 'PINS', 'TWTR', 'MTCH', 'BMBL', 'IAC', 'YELP', 'GRUB',
        'NFLX', 'DIS', 'PARA', 'WBD', 'ROKU', 'FUBO', 'SONO', 'SIRI', 'SPOT', 'TME'
    ]

    # Category 1: European Stocks (150 assets)
    EUROPEAN_STOCKS = [
        # Germany (30)
        'SAP', 'SIE.DE', 'ALV.DE', 'DTE.DE', 'VOW3.DE', 'BAS.DE', 'BAYN.DE', 'MBG.DE', 'BMW.DE', 'DAI.DE',
        'ADS.DE', 'MUV2.DE', 'DB1.DE', 'DBK.DE', 'HEN3.DE', 'FME.DE', 'IFX.DE', 'SHL.DE', 'RWE.DE', 'EOAN.DE',
        'LIN.DE', 'HEI.DE', 'CON.DE', 'FRE.DE', 'BEI.DE', 'PAH3.DE', 'PUM.DE', 'ZAL.DE', 'DTG.DE', '1COV.DE',

        # France (30)
        'MC.PA', 'OR.PA', 'SAN.PA', 'TTE.PA', 'AIR.PA', 'BNP.PA', 'SU.PA', 'SAF.PA', 'CS.PA', 'ORA.PA',
        'RMS.PA', 'DG.PA', 'EL.PA', 'CAP.PA', 'ACA.PA', 'BN.PA', 'EN.PA', 'SGO.PA', 'VIE.PA', 'KER.PA',
        'PUB.PA', 'VIV.PA', 'STM.PA', 'RI.PA', 'DSY.PA', 'ATO.PA', 'ERF.PA', 'WLN.PA', 'GLE.PA', 'FP.PA',

        # UK (30)
        'SHEL.L', 'BP.L', 'HSBA.L', 'AZN.L', 'ULVR.L', 'DGE.L', 'GSK.L', 'RIO.L', 'BATS.L', 'NG.L',
        'REL.L', 'LSEG.L', 'VOD.L', 'PRU.L', 'AAL.L', 'GLEN.L', 'IMB.L', 'CPG.L', 'BHP.L', 'CRH.L',
        'BARC.L', 'LLOY.L', 'NWG.L', 'STAN.L', 'III.L', 'AVV.L', 'ANTO.L', 'RR.L', 'BA.L', 'TSCO.L',

        # Spain/Italy/Netherlands (30)
        'ITX.MC', 'IBE.MC', 'SAN.MC', 'BBVA.MC', 'TEF.MC', 'REP.MC', 'CABK.MC', 'FER.MC', 'ACS.MC', 'ENG.MC',
        'ISP.MI', 'ENI.MI', 'ENEL.MI', 'UCG.MI', 'TIT.MI', 'G.MI', 'STM.MI', 'CPR.MI', 'RACE.MI', 'ATL.MI',
        'ASML.AS', 'PHIA.AS', 'INGA.AS', 'HEIA.AS', 'AD.AS', 'UNA.AS', 'AKZA.AS', 'ABN.AS', 'WKL.AS', 'KPN.AS',

        # Nordics (30)
        'NOVO-B.CO', 'DSV.CO', 'NESTE.HE', 'NOKIA.HE', 'SAMPO.HE', 'NOK.OL', 'EQNR.OL', 'DNB.OL', 'TEL.OL', 'ORK.OL',
        'ERIC-B.ST', 'VOLV-B.ST', 'AZA.ST', 'HM-B.ST', 'ATCO-A.ST', 'ABB.ST', 'SAND.ST', 'SEB-A.ST', 'SWED-A.ST', 'SKF-B.ST',
        'NESN.SW', 'ROG.SW', 'NOVN.SW', 'UBSG.SW', 'CSGN.SW', 'ABB.SW', 'SREN.SW', 'ZURN.SW', 'SOON.SW', 'LONN.SW'
    ]

    # Category 2: Emerging Markets (50 assets)
    EMERGING_MARKETS = [
        # ETFs (15)
        'EEM', 'VWO', 'IEMG', 'EWZ', 'FXI', 'INDA', 'EWY', 'EWT', 'EWW', 'EZA',
        'THD', 'EIDO', 'EPHE', 'TUR', 'RSX',

        # China/HK (15)
        'BABA', 'TCEHY', 'JD', 'PDD', 'BIDU', 'NIO', 'LI', 'XPEV', 'TME', 'BILI',
        '0700.HK', '0941.HK', '9988.HK', '3690.HK', '1810.HK',

        # India (10)
        'INFY', 'WIT', 'HDB', 'IBN', 'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'BHARTIARTL.NS', 'ITC.NS', 'SBIN.NS',

        # Brazil/LatAm (10)
        'VALE', 'PBR', 'ITUB', 'BBD', 'ABEV', 'GGBR4.SA', 'PETR4.SA', 'VALE3.SA', 'ITSA4.SA', 'BBDC4.SA'
    ]

    # Category 3: Commodities (30 assets)
    COMMODITIES = [
        # Metals (10)
        'GC=F',   # Gold
        'SI=F',   # Silver
        'HG=F',   # Copper
        'PL=F',   # Platinum
        'PA=F',   # Palladium
        'ALI=F',  # Aluminum
        'IRON',   # Iron ore (via ETF if available)
        'ZINC',   # Zinc
        'NI=F',   # Nickel
        'TIN',    # Tin

        # Energy (10)
        'CL=F',   # Crude Oil WTI
        'BZ=F',   # Brent Oil
        'NG=F',   # Natural Gas
        'RB=F',   # Gasoline
        'HO=F',   # Heating Oil
        'QA=F',   # Propane
        'LS=F',   # Gas Oil
        'B0=F',   # Diesel
        'CB=F',   # Crude Brent
        'WT=F',   # WTI Crude

        # Agriculture (10)
        'ZC=F',   # Corn
        'ZS=F',   # Soybeans
        'ZW=F',   # Wheat
        'KC=F',   # Coffee
        'SB=F',   # Sugar
        'CC=F',   # Cocoa
        'CT=F',   # Cotton
        'LBS=F',  # Lumber
        'OJ=F',   # Orange Juice
        'LE=F'    # Live Cattle
    ]

    # Category 4: Crypto (70 assets - extended)
    CRYPTO = [
        # Top 20
        'BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD',
        'DOGE-USD', 'SOL-USD', 'TRX-USD', 'DOT-USD', 'MATIC-USD',
        'LTC-USD', 'SHIB-USD', 'AVAX-USD', 'UNI-USD', 'LINK-USD',
        'ATOM-USD', 'XLM-USD', 'BCH-USD', 'FIL-USD', 'APT-USD',

        # 21-50
        'ETC-USD', 'NEAR-USD', 'VET-USD', 'ALGO-USD', 'ICP-USD',
        'QNT-USD', 'HBAR-USD', 'CRO-USD', 'APE-USD', 'LDO-USD',
        'ARB-USD', 'OP-USD', 'MKR-USD', 'AAVE-USD', 'GRT-USD',
        'SAND-USD', 'MANA-USD', 'AXS-USD', 'THETA-USD', 'XTZ-USD',
        'EOS-USD', 'FTM-USD', 'EGLD-USD', 'RUNE-USD', 'KLAY-USD',
        'FLOW-USD', 'CHZ-USD', 'ENJ-USD', 'ZEC-USD', 'DASH-USD',

        # 51-70
        'XMR-USD', 'NEO-USD', 'IOTA-USD', 'MIOTA-USD', 'QTUM-USD',
        'ZIL-USD', 'BAT-USD', 'COMP-USD', 'SNX-USD', 'YFI-USD',
        'UMA-USD', 'CRV-USD', 'SUSHI-USD', '1INCH-USD', 'REN-USD',
        'LRC-USD', 'OMG-USD', 'ANT-USD', 'KNC-USD', 'BNT-USD'
    ]

    CATEGORIES = {
        0: "US_Stocks",
        1: "European_Stocks",
        2: "Emerging_Markets",
        3: "Commodities",
        4: "Crypto"
    }

    def __init__(self, cache_dir: str = "../data/full_market_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Build asset mappings
        self.all_assets = []
        self.asset_to_id = {}
        self.asset_to_category = {}

        asset_id = 0
        for category_id, assets in enumerate([
            self.US_STOCKS,
            self.EUROPEAN_STOCKS,
            self.EMERGING_MARKETS,
            self.COMMODITIES,
            self.CRYPTO
        ]):
            for symbol in assets:
                self.all_assets.append(symbol)
                self.asset_to_id[symbol] = asset_id
                self.asset_to_category[symbol] = category_id
                asset_id += 1

        print(f"Initialized UniversalMarketLoader with {len(self.all_assets)} assets")
        print(f"Categories breakdown:")
        for cat_id, cat_name in self.CATEGORIES.items():
            count = sum(1 for cat in self.asset_to_category.values() if cat == cat_id)
            print(f"  {cat_name}: {count} assets")

    def download_all_assets(
        self,
        start_date: str = "2005-01-01",  # 20 years
        end_date: str = None,
        force_redownload: bool = False
    ) -> Dict[str, pd.DataFrame]:
        """
        Download all 600 assets.

        Returns:
            Dictionary mapping asset symbol to DataFrame
        """
        print(f"\nDownloading {len(self.all_assets)} assets from {start_date} to {end_date or 'today'}")
        print("=" * 80)

        all_data = {}
        failed_assets = []

        for symbol in tqdm(self.all_assets, desc="Downloading assets"):
            cache_file = self.cache_dir / f"{symbol.replace('=', '_').replace('-', '_').replace('.', '_')}.csv"

            # Check cache
            if cache_file.exists() and not force_redownload:
                try:
                    df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                    all_data[symbol] = df
                    continue
                except Exception as e:
                    print(f"Cache read failed for {symbol}: {e}")

            # Download
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(
                    start=start_date,
                    end=end_date,
                    interval='1d',
                    auto_adjust=True
                )

                if len(df) < 50:  # Minimum data requirement
                    print(f"Insufficient data for {symbol}: {len(df)} rows")
                    failed_assets.append(symbol)
                    continue

                # Keep only OHLCV
                df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()

                # Add metadata
                df['asset_id'] = self.asset_to_id[symbol]
                df['category_id'] = self.asset_to_category[symbol]
                df['symbol'] = symbol

                # Cache
                df.to_csv(cache_file)
                all_data[symbol] = df

                time.sleep(0.05)  # Rate limiting

            except Exception as e:
                print(f"Failed to download {symbol}: {e}")
                failed_assets.append(symbol)

        print(f"\nDownload complete:")
        print(f"  Successfully downloaded: {len(all_data)}/{len(self.all_assets)}")
        if failed_assets:
            print(f"  Failed assets ({len(failed_assets)}): saved to failed_assets.json")
            with open(self.cache_dir / "failed_assets.json", "w") as f:
                json.dump(failed_assets, f, indent=2)

        return all_data

    def prepare_training_data(
        self,
        all_data: Dict[str, pd.DataFrame],
        train_ratio: float = 0.7,
        val_ratio: float = 0.15
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Prepare training, validation, and test sets.
        Uses temporal split (NOT random) to prevent data leakage.
        """
        # Concatenate all data
        combined = pd.concat(all_data.values(), axis=0)
        combined = combined.sort_index()

        print(f"\nCombined dataset:")
        print(f"  Total candles: {len(combined):,}")
        print(f"  Date range: {combined.index.min()} to {combined.index.max()}")
        print(f"  Assets: {combined['asset_id'].nunique()}")
        print(f"  Categories: {combined['category_id'].nunique()}")

        # Temporal split
        n = len(combined)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train_df = combined.iloc[:train_end].copy()
        val_df = combined.iloc[train_end:val_end].copy()
        test_df = combined.iloc[val_end:].copy()

        print(f"\nSplit:")
        print(f"  Train: {len(train_df):,} ({100*train_ratio:.1f}%)")
        print(f"  Val:   {len(val_df):,} ({100*val_ratio:.1f}%)")
        print(f"  Test:  {len(test_df):,} ({100*(1-train_ratio-val_ratio):.1f}%)")

        return train_df, val_df, test_df

    def get_category_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get statistics per category."""
        stats = df.groupby('category_id').agg({
            'Close': 'count',
            'asset_id': 'nunique'
        }).rename(columns={'Close': 'num_candles', 'asset_id': 'num_assets'})

        stats['category_name'] = stats.index.map(self.CATEGORIES)
        return stats


if __name__ == "__main__":
    # Test the loader
    loader = UniversalMarketLoader()

    print("\nStarting download of 600 assets...")
    print("This may take 30-60 minutes depending on network speed.")

    all_data = loader.download_all_assets(
        start_date="2005-01-01",
        force_redownload=False
    )

    print("\nPreparing splits...")
    train_df, val_df, test_df = loader.prepare_training_data(all_data)

    print("\nCategory statistics (training set):")
    print(loader.get_category_stats(train_df))

    print("\nData loading test complete!")
    print(f"Total candles across all sets: {len(train_df) + len(val_df) + len(test_df):,}")
