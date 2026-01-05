"""
Interactive Data Collection Setup

This script guides you through:
1. Checking API key status
2. Providing setup instructions
3. Collecting all available data
4. Saving results

Run this first to get all public data!
"""

import os
import sys
from pathlib import Path

# Try to load .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, will use environment variables only

def check_api_key(env_var: str) -> bool:
    """Check if API key is set"""
    return bool(os.getenv(env_var))

def print_header(text: str):
    """Print formatted header"""
    print("\n" + "="*80)
    print(text.center(80))
    print("="*80 + "\n")

def main():
    print_header("QE ANALYSIS - PUBLIC DATA COLLECTION SETUP")
    
    print("This script will help you collect all publicly available data.")
    print("You'll need at least one FREE API key to get started.\n")
    
    # Check API keys
    print("Checking API keys...")
    print("-" * 80)
    
    keys_status = {
        'FRED_API_KEY': {
            'name': 'FRED (Federal Reserve) - FREE',
            'required': True,
            'url': 'https://fred.stlouisfed.org/docs/api/api_key.html',
            'set': check_api_key('FRED_API_KEY')
        }
    }
    
    # No-API-key sources
    print("\nNo API Key Required:")
    print("  ✓ US Treasury (direct download)")
    print("  ✓ FRBNY (manual download instructions provided)")
    print("\nNOT Collected (requires paid subscriptions):")
    print("  ✗ High-frequency intraday data (Bloomberg/Refinitiv)")
    print("  ✗ Treasury bid-ask spreads (Bloomberg/TRACE)")
    print()
    
    # Print status
    has_required = True
    for env_var, info in keys_status.items():
        status = "✓ SET" if info['set'] else "✗ NOT SET"
        required = "(REQUIRED)" if info['required'] else "(optional)"
        print(f"{status:12} {info['name']:40} {required}")
        
        if info['required'] and not info['set']:
            has_required = False
    
    print("-" * 80)
    
    # Guide user
    if not has_required:
        print("\n⚠️  FRED API key is required to continue.")
        print("\nTo get your FREE FRED API key:")
        print("1. Visit: https://fred.stlouisfed.org/docs/api/api_key.html")
        print("2. Create account (2 minutes)")
        print("3. Copy your API key")
        print("4. Set environment variable:")
        print("\n   Windows PowerShell:")
        print('   $env:FRED_API_KEY="your_key_here"')
        print("\n   Windows CMD:")
        print('   set FRED_API_KEY=your_key_here')
        print("\n   Linux/Mac:")
        print('   export FRED_API_KEY="your_key_here"')
        print("\n5. Run this script again")
        print("\nFor detailed setup instructions, see: FREE_API_KEYS_SETUP.md")
        return 1
    
    # Ready to collect
    print("\n✓ Ready to collect data!")
    print("\nAvailable data sources:")
    for env_var, info in keys_status.items():
        if info['set']:
            print(f"  • {info['name']}")
    
    # Ask to proceed
    print("\nThis will collect:")
    print("  • 60+ economic time series from FRED")
    print("  • Daily Treasury yields (2008-2023)")
    print("  • Federal Reserve balance sheet data")
    print("  • Fiscal indicators")
    print("  • Financial conditions indices")
    print("\nEstimated time: 2-5 minutes")
    print("Output location: output/public_data/")
    
    response = input("\nProceed with data collection? [Y/n]: ").strip().lower()
    
    if response in ['', 'y', 'yes']:
        print("\nStarting data collection...")
        print("(This may take a few minutes)\n")
        
        # Run collection script
        import subprocess
        result = subprocess.run([sys.executable, 'collect_all_public_data.py'])
        
        if result.returncode == 0:
            print_header("DATA COLLECTION COMPLETE!")
            print("✓ Data saved to: output/public_data/")
            print("✓ Log saved to: data_collection.log")
            print("\nNext steps:")
            print("1. Review collected data: dir output\\public_data")
            print("2. Check log for any issues: type data_collection.log")
            print("3. Convert to quarterly: python convert_to_quarterly.py")
            print("4. Run analysis: python run_analysis_pipeline.py")
            return 0
        else:
            print("\n✗ Data collection encountered errors.")
            print("Check data_collection.log for details.")
            return 1
    else:
        print("\nData collection cancelled.")
        print("Run this script again when ready.")
        return 0

if __name__ == '__main__':
    sys.exit(main())
