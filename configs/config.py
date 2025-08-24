"""
Amazon Recommender System Configuration File
Created on 2024-06-20 by @SnigdhaTiwari
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class ModelConfig:
    """Two-Tower Model Configuration"""

    # core architecture parameters
    embedding_dim: int = 128 # dimension of the embedding vectors
    user_feature_dim: int = 256 # user feature input size
    item_feature_dim: int = 512 # item feature input size

    # training parameters
    batch_size: int = 256 # batch size for training
    learning_rate: float = 0.001 # learning rate
    num_epochs: int = 10 # training epochs - prevents overfitting
    negative_samples: int = 5 # no. negative samples per positive

    # model architecture
    hidden_layers: list = None # hidden layers size
    dropout_rate: float = 0.2 # dropout rate for regularization - prevents overfitting

    def __post_init__(self): # runs automatically after object creation
        if self.hidden_layers is None:
            self.hidden_layers = [256, 128]

@dataclass
class PathConfig:
    """File and Directory Paths"""

    project_root: Path = Path(__file__).parent.parent # gets the path to the current files and goes up two levels

    # data directories
    data_dir: Path = project_root / "data"
    raw_data_dir: Path = data_dir / "raw"
    processed_data_dir: Path = data_dir / "processed"
    embeddings_dir: Path = data_dir / "embeddings"

    # model and output directories
    models_dir: Path = project_root / "models"
    logs_dir: Path = project_root / "logs"

    # specific file paths
    train_data_path: Optional[str] = None
    test_data_path: Optional[str] = None

    def create_directories(self):
        """Create all necessary directories"""
        dirs = [self.data_dir, 
                self.raw_data_dir, 
                self.processed_data_dir, 
                self.embeddings_dir, 
                self.models_dir, 
                self.logs_dir]
        
        for directory in dirs:
            directory.mkdir(parents=True, exist_ok=True)
            print(f" Created/Verified directory: {directory}")

@dataclass
class AppConfig:
    """Application Settings"""

    # streamlit app settings
    page_title: str = "Amazon Product Recommender"
    page_icon: str = "🛍️"
    layout: str = "wide"

    # recommendation settings
    max_recommendations: int = 20
    min_recommendations: int = 5

    # performance settings
    cache_timeout: int = 3600 # 1 hour cache
    model_device: str = "cpu"

    # data processing settings
    max_sequence_length: int = 512 

@dataclass
class Config:
    """Master Configuration"""
    model: ModelConfig = None
    paths: PathConfig = None
    app: AppConfig = None
   

    def __post_init__(self):
        """Initialise after creation"""

        if self.model is None:
            self.model = ModelConfig()
        if self.paths is None:
            self.paths = PathConfig()
        if self.app is None:
            self.app = AppConfig()

        self.paths.create_directories()
        print("All configurations are set and directories are created.")

    def summary(self):
        """Print a summary of the configurations"""
        print(f" Embedding Dimension: {self.model.embedding_dim}")
        print(f" Batch Size: {self.model.batch_size}")
        print(f" Data Directory: {self.paths.data_dir}")
        print(f" Device: {self.app.model_device}")
        print(f" Max Recommendations: {self.app.max_recommendations}")

# global config instance
def get_config() -> 'Config':
    """Get the global configuration instance"""
    return Config()
    
if __name__ == "__main__":
    # test the configuration when run directly
    config = get_config()
    config.summary()