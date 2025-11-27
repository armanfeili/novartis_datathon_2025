import pandas as pd
from pathlib import Path
import logging
from .utils import timer

class DataManager:
    def __init__(self, config: dict):
        self.config = config
        # Determine if we are in Colab or local
        # This logic might need to be more robust or passed in
        self.is_colab = Path("/content/drive").exists()
        
        paths_cfg = config['drive'] if self.is_colab else config['local']
        self.raw_dir = Path(paths_cfg['raw_dir'])
        self.interim_dir = Path(paths_cfg['interim_dir'])
        self.processed_dir = Path(paths_cfg['processed_dir'])

    def load_raw_data(self) -> dict[str, pd.DataFrame]:
        """Load raw data files defined in config."""
        data = {}
        files = self.config.get('files', {})
        with timer("Load Raw Data"):
            for name, filename in files.items():
                if filename:
                    path = self.raw_dir / filename
                    if path.exists():
                        logging.info(f"Loading {name} from {path}")
                        data[name] = pd.read_csv(path)
                    else:
                        logging.warning(f"File not found: {path}")
        return data

    def make_interim(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Process raw data into interim format (cleaning, merging)."""
        with timer("Make Interim Data"):
            # Implement cleaning and merging logic here
            # df = ...
            pass
        return pd.DataFrame() # Placeholder

    def make_processed(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process interim data into final features."""
        with timer("Make Processed Data"):
            # Implement final processing logic here
            pass
        return df

    def save_processed(self, df: pd.DataFrame, filename: str = "train_processed.csv"):
        """Save processed data to disk."""
        path = self.processed_dir / filename
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        logging.info(f"Saved processed data to {path}")
