"""
Test Amazon Data Analyser
Run complete analysis on Amazon ESCI dataset
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.data_loader import ESCIDataLoader
from src.data_analyser import run_complete_analysis
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Run the complete analysis pipeline"""
    
    print("🚀 Amazon ESCI Data Analysis Pipeline")
    print("=" * 50)
    
    # Step 1: Load data
    print("\n📥 STEP 1: Loading Amazon datasets...")
    loader = ESCIDataLoader()
    datasets = loader.load_all_datasets()
    
    if not datasets:
        print("❌ No datasets loaded. Exiting.")
        return
        
    # Step 2: Run analysis
    print("\n🔬 STEP 2: Running comprehensive analysis...")
    try:
        results = run_complete_analysis(datasets)
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        return
    
    # Step 3: Display key insights
    print("\n📊 STEP 3: Analysis Results")
    print("=" * 50)
    
    # Dataset overview
    if 'dataset_profile' in results:
        profile = results['dataset_profile']
        print(f"\n📋 DATASET OVERVIEW:")
        for dataset_name, info in profile['datasets'].items():
            print(f"   📊 {dataset_name.upper()}: {info['row_count']:,} rows, {info['column_count']} columns")
            print(f"      💾 Memory: {info['memory_usage_mb']:.1f} MB")
            print(f"      🔍 Missing data: {sum(info['missing_values'].values()):,} cells")
    
    # ESCI Analysis
    if 'esci_analysis' in results and results['esci_analysis']:
        esci = results['esci_analysis']
        print(f"\n🎯 RECOMMENDATION QUALITY ANALYSIS:")
        print(f"   📈 Quality Score: {esci.get('quality_score', 0):.1f}/100")
        
        if 'esci_percentages' in esci:
            print(f"   📊 ESCI Label Distribution:")
            for label, pct in esci['esci_percentages'].items():
                label_names = {'E': 'Exact', 'S': 'Substitute', 'C': 'Complement', 'I': 'Irrelevant'}
                print(f"      {label_names.get(label, label)}: {pct:.1f}%")
        
        if 'business_insights' in esci:
            print(f"\n💡 KEY BUSINESS INSIGHTS:")
            for insight in esci['business_insights']:
                print(f"   • {insight}")
    
    # Query Analysis
    if 'query_analysis' in results and results['query_analysis']:
        query = results['query_analysis']
        print(f"\n🔍 SEARCH QUERY ANALYSIS:")
        print(f"   📝 Total queries: {query.get('total_queries', 0):,}")
        print(f"   🔢 Unique queries: {query.get('unique_queries', 0):,}")
        print(f"   📏 Average query length: {query.get('avg_query_length', 0):.1f} characters")
        
        if 'most_common_queries' in query:
            print(f"\n🏆 TOP SEARCH QUERIES:")
            for i, (search_query, count) in enumerate(list(query['most_common_queries'].items())[:5], 1):
                print(f"   {i}. '{search_query}' ({count:,} times)")
    
    # Executive Summary
    if 'summary_insights' in results:
        print(f"\n🎯 EXECUTIVE SUMMARY:")
        for insight in results['summary_insights']:
            print(f"   📌 {insight}")
    
    print(f"\n🎉 Analysis complete! Check logs for detailed information.")

if __name__ == "__main__":
    main()
