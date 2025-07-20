"""Test script to verify pandas functionality."""
import time
import pandas as pd

def test_pandas():
    print("Testing pandas import and basic functionality...")
    start_time = time.time()
    
    # Test basic DataFrame creation
    data = {
        'Name': ['Alice', 'Bob', 'Charlie'],
        'Age': [25, 30, 35],
        'City': ['New York', 'San Francisco', 'Los Angeles']
    }
    
    try:
        df = pd.DataFrame(data)
        print("\nDataFrame created successfully:")
        print(df)
        print("\nBasic statistics:")
        print(df.describe())
        
        # Test a simple operation
        df['Age_Plus_10'] = df['Age'] + 10
        print("\nAfter adding Age_Plus_10 column:")
        print(df)
        
        print(f"\nPandas version: {pd.__version__}")
        print(f"Test completed in {time.time() - start_time:.2f} seconds")
        return True
    except Exception as e:
        print(f"\nError during pandas test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Starting pandas test...")
    success = test_pandas()
    if success:
        print("\n✅ Pandas test completed successfully!")
    else:
        print("\n❌ Pandas test failed!")
