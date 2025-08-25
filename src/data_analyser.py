"""
Amazon ESCI Data Analyser File
Created on 2025-08-24 by @SnigdhaTiwari
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set up logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AmazonDataAnalyser:  # Fixed class name
    """Amazon ESCI dataset analyser"""

    def __init__(self, datasets: Dict[str, pd.DataFrame]):
        """Initialise analyser with loaded datasets"""

        self.datasets = datasets
        self.train_df = datasets.get('train')
        self.products_df = datasets.get('products')

        # Validate required datasets
        if self.train_df is None:
            raise ValueError("Training dataset is required for analysis")
        
        logger.info(f"Analyser initialised with {len(datasets)} datasets")
        logger.info(f"Training data: {self.train_df.shape[0]:,} rows")
        if self.products_df is not None:
            logger.info(f"Product data: {self.products_df.shape[0]:,} rows")

        # Set up visualisation style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

    def generate_dataset_profile(self) -> Dict:
        """Generate dataset profile and stats"""
        logger.info("Generating comprehensive dataset profile...")
        
        profile = {
            'datasets': {}, 
            'analysis_timestamp': pd.Timestamp.now().isoformat(), 
            'business_insights': []
        }

        # Analyse training datasets
        if self.train_df is not None:
            train_profile = self._profile_single_dataset('train', self.train_df)
            profile['datasets']['train'] = train_profile
        
        # Analyse products dataset
        if self.products_df is not None:
            products_profile = self._profile_single_dataset('products', self.products_df)
            profile['datasets']['products'] = products_profile
        
        # Generate cross-dataset insights
        profile['cross_dataset_analysis'] = self._analyze_dataset_relationships()

        logger.info("Dataset profiling complete")
        return profile
    
    def _profile_single_dataset(self, name: str, df: pd.DataFrame) -> Dict:
        """Profile individual dataset with comprehensive metrics"""
        logger.info(f"Profiling {name} dataset...")

        profile = {
            # Basic stats
            'shape': df.shape, 
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024, 
            'column_count': len(df.columns), 
            'row_count': len(df), 

            # Data quality metrics
            'missing_values': df.isnull().sum().to_dict(),
            'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
            'duplicate_rows': df.duplicated().sum(),
            'duplicate_percentage': df.duplicated().sum() / len(df) * 100,

            # Column analysis
            'column_types': df.dtypes.astype(str).to_dict(),
            'unique_counts': df.nunique().to_dict(),

            # Text analysis
            'text_analysis': {},

            # Business metrics
            'business_metrics': self._extract_business_metrics(name, df)
        }

        # Analyse text columns
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].dtype == 'object':
                profile['text_analysis'][col] = {
                    'avg_length': df[col].astype(str).str.len().mean(),
                    'max_length': df[col].astype(str).str.len().max(),
                    'min_length': df[col].astype(str).str.len().min(),
                    'empty_strings': (df[col] == '').sum()
                }
        
        return profile  # Fixed: moved outside the loop
        
    def _extract_business_metrics(self, dataset_name: str, df: pd.DataFrame) -> Dict:
        """Extract business specific metrics from dataset"""
        metrics = {}

        if dataset_name == 'train' and 'esci_label' in df.columns:
            # ESCI label distribution
            label_dist = df['esci_label'].value_counts()
            metrics['esci_distribution'] = label_dist.to_dict()
            metrics['esci_percentages'] = (label_dist / len(df) * 100).to_dict()

            # Query analysis
            if 'query' in df.columns:
                metrics['unique_queries'] = df['query'].nunique()
                metrics['avg_query_length'] = df['query'].str.len().mean()  # Fixed typo
                metrics['queries_per_product'] = len(df) / df['product_id'].nunique() if 'product_id' in df.columns else None
            
        return metrics
        
    def create_esci_analysis_dashboard(self) -> Dict:  # Fixed: moved to class level
        """Create ESCI analysis dashboard"""
        logger.info("Creating ESCI analysis dashboard...")
        
        if 'esci_label' not in self.train_df.columns:
            logger.warning("ESCI labels not found in dataset")
            return {}
        
        esci_counts = self.train_df['esci_label'].value_counts()
        esci_percentages = self.train_df['esci_label'].value_counts(normalize=True) * 100

        fig = make_subplots(
            rows=2, 
            cols=2, 
            subplot_titles=['ESCI Label Distribution', 'Query Length by Label', 
                          'Products per Query by Label', 'Label Quality Score'], 
            specs=[[{"type": "pie"}, {"type": "box"}], 
                   [{"type": "bar"}, {"type": "scatter"}]]            
        )
        
        # Pie chart of ESCI distribution
        fig.add_trace(
            go.Pie(labels=esci_counts.index, 
                   values=esci_counts.values, 
                   name="ESCI Distribution"), 
            row=1, 
            col=1
        )

        # Box plot of query lengths by label
        for label in esci_counts.index:
            label_data = self.train_df[self.train_df['esci_label'] == label]
            fig.add_trace(
                go.Box(y=label_data['query'].str.len(), 
                       name=f"Label {label}"), 
                row=1, 
                col=2
            )
            
        # Business insights generation
        insights = self._generate_esci_insights(esci_counts, esci_percentages)
        
        # Save visualization
        fig.update_layout(height=800, showlegend=True, 
                         title_text="Amazon ESCI Analysis Dashboard")
    
        results = {
            'esci_distribution': esci_counts.to_dict(),
            'esci_percentages': esci_percentages.to_dict(),
            'business_insights': insights,
            'visualization': fig,
            'quality_score': self._calculate_recommendation_quality_score(esci_percentages)
        }
        
        logger.info("✅ ESCI analysis dashboard created")
        return results
    
    def _generate_esci_insights(self, counts: pd.Series, percentages: pd.Series) -> List[str]:
        """Generate business insights from ESCI analysis"""
        
        insights = []
        
        # Analyze Exact matches
        if 'E' in percentages.index:
            exact_pct = percentages['E']
            if exact_pct < 20:
                insights.append(f"⚠️ Only {exact_pct:.1f}% exact matches - opportunity to improve search relevance")
            elif exact_pct > 40:
                insights.append(f"✅ Strong exact match rate at {exact_pct:.1f}% - good search quality")
                
        # Analyze Irrelevant results
        if 'I' in percentages.index:
            irrelevant_pct = percentages['I']
            if irrelevant_pct > 30:
                insights.append(f"🚨 High irrelevant results at {irrelevant_pct:.1f}% - major improvement opportunity")
                revenue_impact = irrelevant_pct * 0.15  # Assume 15% conversion loss per irrelevant result
                insights.append(f"💰 Potential revenue impact: ~{revenue_impact:.1f}% conversion loss")
                
        # Cross-selling opportunities
        if 'C' in percentages.index:
            complement_pct = percentages['C']
            insights.append(f"🛒 {complement_pct:.1f}% complement opportunities for cross-selling")
            
        return insights
    
    def _calculate_recommendation_quality_score(self, percentages: pd.Series) -> float:
        """Calculate overall recommendation system quality score"""
        
        # Weighted quality score
        weights = {'E': 1.0, 'S': 0.8, 'C': 0.6, 'I': -0.5}  # Business value weights
        score = sum(percentages.get(label, 0) * weight for label, weight in weights.items())
        
        return max(0, min(100, score))  # Normalize to 0-100 scale           
    
    def generate_query_analysis(self) -> Dict:
        """Analyze search query patterns and insights"""
        
        logger.info("🔍 Analyzing query patterns...")
        
        if 'query' not in self.train_df.columns:
            return {}
            
        # Query statistics
        queries = self.train_df['query'].str.lower().str.strip()
        
        analysis = {
            'total_queries': len(queries),
            'unique_queries': queries.nunique(),
            'avg_query_length': queries.str.len().mean(),
            'most_common_queries': queries.value_counts().head(10).to_dict(),
            'query_length_distribution': queries.str.len().describe().to_dict(),
            'business_insights': []
        }
        
        # Generate insights
        if analysis['avg_query_length'] < 10:
            analysis['business_insights'].append("Short queries detected - users may need better search suggestions")
        if analysis['unique_queries'] / analysis['total_queries'] > 0.8:
            analysis['business_insights'].append("High query diversity - good product discovery coverage")
            
        return analysis
    
    def _analyze_dataset_relationships(self) -> Dict:
        """Analyze relationships between datasets"""
        
        relationships = {}
        
        if self.train_df is not None and self.products_df is not None:
            # Find common products
            train_products = set(self.train_df['product_id'].unique()) if 'product_id' in self.train_df.columns else set()
            product_ids = set(self.products_df['product_id'].unique()) if 'product_id' in self.products_df.columns else set()
            
            relationships['common_products'] = len(train_products.intersection(product_ids))
            relationships['train_only_products'] = len(train_products - product_ids)
            relationships['products_only_products'] = len(product_ids - train_products)
            
        return relationships

# Helper function to run complete analysis
def run_complete_analysis(datasets: Dict[str, pd.DataFrame]) -> Dict:
    """
    Run complete analysis pipeline on Amazon datasets
    
    Args:
        datasets: Dictionary containing loaded datasets
        
    Returns:
        Complete analysis results
    """
    
    analyser = AmazonDataAnalyser(datasets)
    
    logger.info("🚀 Starting complete Amazon data analysis...")
    
    results = {
        'dataset_profile': analyser.generate_dataset_profile(),
        'esci_analysis': analyser.create_esci_analysis_dashboard(),
        'query_analysis': analyser.generate_query_analysis(),
        'summary_insights': []
    }
    
    # Generate executive summary
    if results['esci_analysis']:
        quality_score = results['esci_analysis'].get('quality_score', 0)
        results['summary_insights'].append(f"Overall recommendation quality score: {quality_score:.1f}/100")
        
    logger.info("✅ Complete analysis finished!")
    return results
