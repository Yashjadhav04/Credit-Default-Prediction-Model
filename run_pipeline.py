#!/usr/bin/env python3
"""
Single Command Execution - Fresh EDA PD Model
Run the complete pipeline with one command
"""

from harness import PDModelHarness
import sys
import os


def main():
    """Execute complete PD model pipeline."""
    
    print("""
    ╔═══════════════════════════════════════════════════════════════════╗
    ║                  FRESH EDA PD MODEL PIPELINE                      ║
    ║                                                                   ║
    ║  Complete Probability of Default modeling using Fresh EDA data   ║
    ║  Excludes: leverage_w, leverage_z                                ║
    ╚═══════════════════════════════════════════════════════════════════╝
    """)
    
    # Configuration
    config = {
        'data_path': '/Users/yj/Downloads/ML TRIAL /Default_final copy.csv',
        'results_dir': '../results',
        'n_features': 12,
        'start_year': 2009,
        'end_year': 2012,
        'prediction_sample': 50000
    }
    
    # Print configuration
    print("📋 CONFIGURATION:")
    print("-" * 70)
    print(f"  Data Source:        {config['data_path']}")
    print(f"  Results Directory:  {config['results_dir']}")
    print(f"  Features Selected:  {config['n_features']}")
    print(f"  Training Period:    {config['start_year']}-{config['end_year']}")
    print(f"  Prediction Sample:  {config['prediction_sample']:,}")
    print("-" * 70)
    
    # Check if data file exists
    if not os.path.exists(config['data_path']):
        print(f"\n❌ ERROR: Data file not found: {config['data_path']}")
        print("   Please ensure 'Default_final copy.csv' is in the parent directory.")
        return 1
    
    # Initialize and run pipeline
    try:
        harness = PDModelHarness(
            data_path=config['data_path'],
            results_dir=config['results_dir']
        )
        
        success = harness.run_complete_pipeline(
            n_features=config['n_features'],
            start_year=config['start_year'],
            end_year=config['end_year'],
            prediction_sample=config['prediction_sample'],
            use_latest_model=True
        )
        
        if success:
            print("\n" + "=" * 70)
            print("🎉 SUCCESS! Pipeline completed successfully.")
            print("=" * 70)
            print(f"\n📁 Results saved to: {config['results_dir']}/")
            print("\nGenerated files:")
            print("  - fresh_eda_model_*.pkl (trained models)")
            print("  - fresh_eda_performance_*.json (metrics)")
            print("  - fresh_eda_predictions_*.csv (PD scores)")
            print("  - fresh_eda_comprehensive_report.json (full report)")
            print("\n" + "=" * 70)
            return 0
        else:
            print("\n" + "=" * 70)
            print("⚠️  Pipeline completed with errors.")
            print("=" * 70)
            print("\nCheck the output above for details.")
            return 1
            
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
