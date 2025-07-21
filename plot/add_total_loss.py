"""
Script to add a total_loss column to CSV files containing loss data.
The total_loss is calculated as the sum of curl_loss, taco_loss, and reward_loss.
Also includes preprocessing functionality to interpolate data points for plotting.

Usage:
python plot/add_total_loss.py --csv_path data_plot/loss_visualization/button-press
python plot/add_total_loss.py --csv_path data_plot/loss_visualization --recursive
python plot/add_total_loss.py --csv_path data_plot/loss_visualization --recursive --processing
"""

import argparse
import pandas as pd
import os
import sys
import numpy as np
from scipy.interpolate import interp1d


def clean_and_process_csv_data(csv_file_path):
    """
    Clean CSV data by removing unnecessary columns and averaging duplicate buffer_size values.
    
    Parameters:
    - csv_file_path: Path to the CSV file
    
    Returns:
    - bool: True if successful, False otherwise
    """
    try:
        # Read the CSV file
        df = pd.read_csv(csv_file_path)
        
        print(f"Original shape: {df.shape}")
        print(f"Original columns: {list(df.columns)}")
        
        # Remove unnamed columns (they are just counters)
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        
        # Remove _step column if it exists
        if '_step' in df.columns:
            df = df.drop('_step', axis=1)
        
        # Check if we have the required columns
        required_columns = ['buffer_size', 'curl_loss', 'taco_loss', 'reward_loss']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"ERROR: File {csv_file_path} is missing required columns: {missing_columns}")
            print(f"Available columns: {list(df.columns)}")
            return False
        
        # Group by buffer_size and take the mean of other columns
        # This handles cases where there are multiple entries for the same buffer_size
        loss_columns = ['curl_loss', 'taco_loss', 'reward_loss']
        
        # Check if there are duplicate buffer_size values
        duplicate_counts = df['buffer_size'].value_counts()
        duplicates = duplicate_counts[duplicate_counts > 1]
        
        if len(duplicates) > 0:
            print(f"Found {len(duplicates)} buffer_size values with duplicates:")
            for buffer_size, count in duplicates.head().items():
                print(f"  buffer_size {buffer_size}: {count} entries")
            
            # Group by buffer_size and take mean of loss columns
            df_grouped = df.groupby('buffer_size')[loss_columns].mean().reset_index()
            
            print(f"After grouping by buffer_size - shape: {df_grouped.shape}")
        else:
            df_grouped = df[['buffer_size'] + loss_columns].copy()
        
        # Sort by buffer_size
        df_grouped = df_grouped.sort_values('buffer_size').reset_index(drop=True)
        
        # Save the cleaned data
        df_grouped.to_csv(csv_file_path, index=False)
        
        print(f"SUCCESS: Cleaned and processed data in {csv_file_path}")
        print(f"Final shape: {df_grouped.shape}")
        return True
        
    except Exception as e:
        print(f"ERROR: Failed to clean and process {csv_file_path}: {str(e)}")
        return False


def add_total_loss_to_csv(csv_file_path):
    """
    Add total_loss column to a single CSV file.
    
    Parameters:
    - csv_file_path: Path to the CSV file
    
    Returns:
    - bool: True if successful, False otherwise
    """
    try:
        # Read the CSV file
        df = pd.read_csv(csv_file_path)
        
        # Check if required columns exist
        required_columns = ['curl_loss', 'taco_loss', 'reward_loss']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"WARNING: File {csv_file_path} is missing columns: {missing_columns}")
            print(f"Available columns: {list(df.columns)}")
            return False
        
        # Check if total_loss column already exists
        if 'total_loss' in df.columns:
            print(f"INFO: File {csv_file_path} already has total_loss column. Updating it.")
        
        # Calculate total_loss
        df['total_loss'] = df['curl_loss'] + df['taco_loss'] + df['reward_loss']
        
        # Ensure proper column ordering: buffer_size, loss columns, total_loss
        column_order = ['buffer_size', 'curl_loss', 'taco_loss', 'reward_loss', 'total_loss']
        
        # Add any other columns that might exist
        other_columns = [col for col in df.columns if col not in column_order]
        final_columns = column_order + other_columns
        
        # Filter to only include columns that actually exist
        final_columns = [col for col in final_columns if col in df.columns]
        
        df = df[final_columns]
        
        # Save the updated CSV
        df.to_csv(csv_file_path, index=False)
        
        print(f"SUCCESS: Added total_loss column to {csv_file_path}")
        return True
        
    except Exception as e:
        print(f"ERROR: Failed to add total_loss to {csv_file_path}: {str(e)}")
        return False


def preprocess_csv_for_plotting(csv_file_path, x_key, n_points, min_x, max_x, keys_to_interpolate):
    """
    Preprocess a CSV file by interpolating data points for plotting.
    
    Parameters:
    - csv_file_path: Path to the CSV file
    - x_key: Column name to use as x-axis
    - n_points: Number of points to interpolate
    - min_x: Minimum x value
    - max_x: Maximum x value (None for auto)
    - keys_to_interpolate: List of column names to interpolate
    
    Returns:
    - bool: True if successful, False otherwise
    """
    try:
        # Read the CSV file
        df = pd.read_csv(csv_file_path)
        
        # Check if required columns exist
        required_columns = [x_key] + keys_to_interpolate
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"WARNING: File {csv_file_path} is missing columns for preprocessing: {missing_columns}")
            print(f"Available columns: {list(df.columns)}")
            return False
        
        # Check if we have enough data points
        if len(df) < 2:
            print(f"WARNING: File {csv_file_path} has less than 2 data points, skipping interpolation")
            return False
        
        # Determine x-axis range
        x_min = min_x if min_x is not None else int(df.iloc[0][x_key])
        x_max = max_x if max_x is not None else int(df.iloc[-1][x_key])
        
        # Create interpolated x-axis
        x_axis = np.linspace(x_min, x_max, n_points, dtype=int)
        
        # Create new dataframe with interpolated x values
        new_df = pd.DataFrame({x_key: x_axis})
        
        # Interpolate each specified column
        for key in keys_to_interpolate:
            if key in df.columns:
                try:
                    # Use interpolation to get values at new x points
                    f = interp1d(df[x_key], df[key], fill_value="extrapolate")
                    new_df[key] = f(x_axis)
                except Exception as e:
                    print(f"WARNING: Failed to interpolate {key}: {str(e)}")
                    # If interpolation fails, use forward fill
                    new_df[key] = df[key].iloc[0] if len(df) > 0 else 0
        
        # Save the preprocessed CSV
        new_df.to_csv(csv_file_path, index=False)
        
        print(f"SUCCESS: Preprocessed {csv_file_path} with {n_points} interpolated points")
        return True
        
    except Exception as e:
        print(f"ERROR: Failed to preprocess {csv_file_path}: {str(e)}")
        return False


def process_directory(csv_path, recursive=False, clean_data=True, add_loss=True, preprocess=False, 
                     x_key='buffer_size', n_points=1000, min_x=100, max_x=100000, 
                     keys_to_interpolate=None):
    """
    Process all CSV files in a directory.
    
    Parameters:
    - csv_path: Path to the directory containing CSV files
    - recursive: If True, search subdirectories recursively
    - clean_data: If True, clean and process raw data first
    - add_loss: If True, add total_loss column
    - preprocess: If True, preprocess data for plotting
    - x_key: Column name to use as x-axis for preprocessing
    - n_points: Number of points to interpolate
    - min_x: Minimum x value
    - max_x: Maximum x value (None for auto)
    - keys_to_interpolate: List of column names to interpolate
    
    Returns:
    - tuple: (successful_files, failed_files, skipped_files)
    """
    if not os.path.exists(csv_path):
        print(f"ERROR: Path {csv_path} does not exist")
        return 0, 0, 0
    
    if keys_to_interpolate is None:
        keys_to_interpolate = ['curl_loss', 'taco_loss', 'reward_loss', 'total_loss']
    
    successful_files = 0
    failed_files = 0
    skipped_files = 0
    
    if recursive:
        # Walk through all subdirectories
        for root, dirs, files in os.walk(csv_path):
            csv_files = [f for f in files if f.endswith('.csv')]
            
            if csv_files:
                print(f"\nProcessing directory: {root}")
                print(f"Found {len(csv_files)} CSV files")
                
                for filename in csv_files:
                    file_path = os.path.join(root, filename)
                    success = True
                    
                    print(f"\n--- Processing {filename} ---")
                    
                    # Step 1: Clean and process raw data if requested
                    if clean_data:
                        success = success and clean_and_process_csv_data(file_path)
                    
                    # Step 2: Add total_loss column if requested
                    if add_loss and success:
                        success = success and add_total_loss_to_csv(file_path)
                    
                    # Step 3: Preprocess for plotting if requested
                    if preprocess and success:
                        success = success and preprocess_csv_for_plotting(
                            file_path, x_key, n_points, min_x, max_x, keys_to_interpolate)
                    
                    if success:
                        successful_files += 1
                    else:
                        failed_files += 1
    else:
        # Process only files in the specified directory
        if os.path.isfile(csv_path):
            # Single file provided
            if csv_path.endswith('.csv'):
                success = True
                
                print(f"\n--- Processing single file {csv_path} ---")
                
                # Step 1: Clean and process raw data if requested
                if clean_data:
                    success = success and clean_and_process_csv_data(csv_path)
                
                # Step 2: Add total_loss column if requested
                if add_loss and success:
                    success = success and add_total_loss_to_csv(csv_path)
                
                # Step 3: Preprocess for plotting if requested
                if preprocess and success:
                    success = success and preprocess_csv_for_plotting(
                        csv_path, x_key, n_points, min_x, max_x, keys_to_interpolate)
                
                if success:
                    successful_files += 1
                else:
                    failed_files += 1
            else:
                print(f"ERROR: {csv_path} is not a CSV file")
                failed_files += 1
        else:
            # Directory provided
            csv_files = [f for f in os.listdir(csv_path) if f.endswith('.csv')]
            
            if not csv_files:
                print(f"WARNING: No CSV files found in {csv_path}")
                return 0, 0, 0
            
            print(f"Found {len(csv_files)} CSV files in {csv_path}")
            
            for filename in csv_files:
                file_path = os.path.join(csv_path, filename)
                success = True
                
                print(f"\n--- Processing {filename} ---")
                
                # Step 1: Clean and process raw data if requested
                if clean_data:
                    success = success and clean_and_process_csv_data(file_path)
                
                # Step 2: Add total_loss column if requested
                if add_loss and success:
                    success = success and add_total_loss_to_csv(file_path)
                
                # Step 3: Preprocess for plotting if requested
                if preprocess and success:
                    success = success and preprocess_csv_for_plotting(
                        file_path, x_key, n_points, min_x, max_x, keys_to_interpolate)
                
                if success:
                    successful_files += 1
                else:
                    failed_files += 1
    
    return successful_files, failed_files, skipped_files


def main():
    parser = argparse.ArgumentParser(
        description="Process CSV files containing loss data: clean data, add total_loss column, and optionally preprocess for plotting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Clean data and add total_loss column
  python plot/add_total_loss.py --csv_path data_plot/loss_visualization/button-press
  
  # Clean, add total_loss, and preprocess for plotting
  python plot/add_total_loss.py --csv_path data_plot/loss_visualization --recursive --processing
  
  # Only add total_loss (skip cleaning)
  python plot/add_total_loss.py --csv_path data_plot/loss_visualization --no_clean_data
        """
    )
    
    parser.add_argument('--csv_path', type=str, required=True,
                        help='Path to CSV file or directory containing CSV files')
    parser.add_argument('--recursive', action='store_true', default=False,
                        help='Process CSV files in subdirectories recursively')
    parser.add_argument('--dry_run', action='store_true', default=False,
                        help='Show what would be processed without making changes')
    parser.add_argument('--processing', action='store_true', default=False,
                        help='Enable preprocessing of data for plotting (interpolation)')
    parser.add_argument('--no_clean_data', action='store_true', default=False,
                        help='Skip data cleaning step (remove unnamed columns, average duplicates)')
    parser.add_argument('--add_loss', action='store_true', default=True,
                        help='Add total_loss column (default: True)')
    parser.add_argument('--x_key', type=str, default='buffer_size',
                        help='Column name to use as x-axis for preprocessing (default: buffer_size)')
    parser.add_argument('--n_points', type=int, default=1000,
                        help='Number of points to interpolate (default: 1000)')
    parser.add_argument('--min_x', type=int, default=100,
                        help='Minimum x value for interpolation (default: 100)')
    parser.add_argument('--max_x', type=int, default=100000,
                        help='Maximum x value for interpolation (default: 100000)')
    parser.add_argument('--keys_to_interpolate', type=str, 
                        default='curl_loss,taco_loss,reward_loss,total_loss',
                        help='Comma-separated list of columns to interpolate (default: curl_loss,taco_loss,reward_loss,total_loss)')

    args = parser.parse_args()
    
    # Parse keys to interpolate
    keys_to_interpolate = [key.strip() for key in args.keys_to_interpolate.split(',')]
    
    # Determine what operations to perform
    clean_data = not args.no_clean_data
    
    print(f"=== LOSS DATA PROCESSING SCRIPT ===")
    print(f"CSV Path: {args.csv_path}")
    print(f"Recursive: {args.recursive}")
    print(f"Dry Run: {args.dry_run}")
    print(f"Clean Data: {clean_data}")
    print(f"Add Loss Column: {args.add_loss}")
    print(f"Preprocessing: {args.processing}")
    if args.processing:
        print(f"X-axis Key: {args.x_key}")
        print(f"Number of Points: {args.n_points}")
        print(f"Min X: {args.min_x}")
        print(f"Max X: {args.max_x}")
        print(f"Keys to Interpolate: {keys_to_interpolate}")
    print(f"{'='*50}")
    
    if args.dry_run:
        print("DRY RUN MODE: No files will be modified")
        
        if os.path.isfile(args.csv_path):
            if args.csv_path.endswith('.csv'):
                print(f"Would process file: {args.csv_path}")
            else:
                print(f"ERROR: {args.csv_path} is not a CSV file")
        else:
            if args.recursive:
                for root, dirs, files in os.walk(args.csv_path):
                    csv_files = [f for f in files if f.endswith('.csv')]
                    if csv_files:
                        print(f"Would process {len(csv_files)} files in: {root}")
                        for f in csv_files:
                            print(f"  - {f}")
            else:
                csv_files = [f for f in os.listdir(args.csv_path) if f.endswith('.csv')]
                print(f"Would process {len(csv_files)} files in: {args.csv_path}")
                for f in csv_files:
                    print(f"  - {f}")
        return
    
    # Process the files
    successful, failed, skipped = process_directory(
        args.csv_path, 
        recursive=args.recursive,
        clean_data=clean_data,
        add_loss=args.add_loss,
        preprocess=args.processing,
        x_key=args.x_key,
        n_points=args.n_points,
        min_x=args.min_x,
        max_x=args.max_x,
        keys_to_interpolate=keys_to_interpolate
    )
    
    print(f"\n{'='*50}")
    print(f"PROCESSING COMPLETE")
    print(f"Successfully processed: {successful} files")
    print(f"Failed to process: {failed} files")
    print(f"Skipped files: {skipped} files")
    print(f"{'='*50}")
    
    if failed > 0:
        print(f"WARNING: {failed} files failed to process. Check the error messages above.")
        sys.exit(1)
    else:
        print("All files processed successfully!")


if __name__ == "__main__":
    main()
