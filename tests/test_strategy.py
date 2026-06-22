import os
import sys

import numpy as np
import pandas as pd

# Establish pathing resolution up into the parent quantitative project space
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "strategy_trainer"))

from src.features.extractor import FeatureExtractor
from strategy import get_signals


def test_feature_engineering_columns():
    """Tests if the V2 strategy successfully processes engineered data fields."""

    # 1. Instantiate a dummy tracking layout filled with 600 raw market candles
    # (XGBoost validation gate requires a minimum of 500 rows to execute safely)
    np.random.seed(42)
    df = pd.DataFrame(
        {
            "open": np.random.uniform(50000, 51000, 600),
            "high": np.random.uniform(51000, 52000, 600),
            "low": np.random.uniform(49000, 50000, 600),
            "close": np.random.uniform(50000, 51000, 600),
            "volume": np.random.uniform(10, 100, 600),
        }
    )

    # 2. Run raw market historical logs through our Feature Extractor layer
    extractor = FeatureExtractor()
    engineered_df = extractor.extract_features(df)

    # 3. Process the enriched dataset through our machine learning strategy matrix
    processed_df = get_signals(engineered_df)

    # 4. Assertions verifying our active V2 column outputs are present
    assert "signal" in processed_df.columns, "Missing core trading execution signal"
    assert "ml_signal" in processed_df.columns, "Missing underlying ML probability entry signal"

    # Verify that our data alignment constraints did not drop or shift baseline records
    assert len(processed_df) == 600, "Strategy evaluation structure dropped candle rows"

    # Verify that generated position outputs adhere strictly to clear risk bounds
    valid_signals = {1.0, 0.0}
    unique_signals = set(processed_df["signal"].unique())
    assert unique_signals.issubset(
        valid_signals
    ), f"Invalid position state generated: {unique_signals}"
