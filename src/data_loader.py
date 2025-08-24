"""
Amazon ESCI Data Loader File
Created on 2024-06-20 by @SnigdhaTiwari
"""
import sys
from pathlib import Path

# Adding project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd 
import requests  # HTTP downloads
from typing import Optional, Dict, List
import logging
from tqdm import tqdm  # progress bars

class SimpleConfig:
    def __init__(self):
        self.paths = self.PathConfig()
    
    class PathConfig:
        def __init__(self):
            self.project_root = Path(__file__).parent.parent.parent
            self.raw_data_dir = self.project_root / "data" / "raw"
            self.raw_data_dir.mkdir(parents=True, exist_ok=True)

def get_config():
    return SimpleConfig()

# Create logs directory if it doesn't exist
logs_dir = Path("logs")
logs_dir.mkdir(exist_ok=True)

# Set up professional production logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
    handlers=[
        logging.FileHandler(logs_dir / 'data_loader.log'), 
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ESCIDataLoader:
    """Data Loader for Amazon ESCI Dataset"""
    
    def __init__(self, config=None):
        if config is None:
            self.config = SimpleConfig()
        else:
            self.config = config
        self.data_urls = {
            'train': 'https://github.com/amazon-science/esci-data/raw/main/shopping_queries_dataset/shopping_queries_dataset_examples.parquet',
            'products': 'https://github.com/amazon-science/esci-data/raw/main/shopping_queries_dataset/shopping_queries_dataset_products.parquet'
        }
        logger.info(f"ESCI Data Loader initialised with {len(self.data_urls)} data sources")
    
    def download_file(self, url: str, filename: str) -> bool:
        """Error handling and progress tracking for file downloads"""
        filepath = self.config.paths.raw_data_dir / filename

        if filepath.exists():
            logger.info(f"File already exists: {filename}")
            return True
        
        try:
            logger.info(f"Downloading {filename} from {url}")

            # HTTP request with error handling
            response = requests.get(url, stream=True)
            response.raise_for_status()

            # Total size for progress bar
            total_size = int(response.headers.get('content-length', 0))

            # Download with progress bar - FIXED INDENTATION
            with open(filepath, 'wb') as file, tqdm(
                desc=filename,
                total=total_size, 
                unit='iB',
                unit_scale=True, 
                unit_divisor=1024
            ) as progress_bar:
                
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        file.write(chunk)
                        progress_bar.update(len(chunk))
            
            logger.info(f"Successfully downloaded: {filename}")
            return True
        
        except requests.exceptions.RequestException as e: 
            logger.error(f"Download failed for {filename}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error downloading {filename}: {e}")
            return False
    
    def load_dataset(self, dataset_name: str) -> Optional[pd.DataFrame]:
        """Load dataset with validation and error handling"""
        if dataset_name not in self.data_urls:
            logger.error(f"Unknown dataset: {dataset_name}")
            return None
        
        # Download if needed
        url = self.data_urls[dataset_name]
        filename = f"{dataset_name}.parquet"

        if not self.download_file(url, filename):
            return None
        
        # Load the data
        filepath = self.config.paths.raw_data_dir / filename

        try:
            logger.info(f"Loading {dataset_name} dataset...")
            df = pd.read_parquet(filepath)

            if df.empty:
                logger.warning(f"Dataset {dataset_name} is empty!")
                return None
            
            logger.info(f"Loaded {dataset_name}: {df.shape[0]:,} rows, {df.shape[1]} columns")

            # Log memory usage
            memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
            logger.info(f"Memory usage: {memory_mb:.1f} MB")
            return df
        
        except Exception as e:
            logger.error(f"Failed to load {dataset_name}: {e}")
            return None
        
    def load_all_datasets(self) -> Dict[str, pd.DataFrame]:
        """Load all available datasets"""
        datasets = {}
        for dataset_name in self.data_urls.keys():
            df = self.load_dataset(dataset_name)
            if df is not None:
                datasets[dataset_name] = df
        logger.info(f"Successfully loaded {len(datasets)} datasets")
        return datasets
    
# Testing pattern
if __name__ == "__main__":
    """Test the data loader when run directly"""

    print("🧪 Testing ESCI Data Loader...")

    # Initialize loader
    loader = ESCIDataLoader()

    # Test loading datasets
    datasets = loader.load_all_datasets()

    # Display results
    for name, df in datasets.items():
        print(f"\n📊 Dataset: {name}")
        print(f"   Shape: {df.shape}")
        print(f"   Columns: {list(df.columns)}")
        print(f"   Sample Data:")
        print(df.head(2).to_string())
        
    print(f"\n🎉 Testing complete! Loaded {len(datasets)} datasets")
