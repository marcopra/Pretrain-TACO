import argparse
import pandas as pd
import wandb
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import json
import os
import os.path as osp
import numpy as np
from scipy.interpolate import interp1d

def read_csvs_from_directory(directory):
    dataframes = []

    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            df = pd.read_csv(os.path.join(directory, filename))
            dataframes.append(df)
    return dataframes

def flatten_dict(d):
    """Flatten a nested dictionary."""
    items = []
    for key, value in d.items():
        if isinstance(value, dict):
            # The name of the nested keys are <parent_key>/<child_key>
            for sub_key, sub_value in value.items():
                new_key = f"{key}/{sub_key}"
                if isinstance(sub_value, dict):
                    items.extend(flatten_dict(sub_value).items())
                else:
                    items.append((new_key, sub_value))
        else:
            items.append((key, value))
    return dict(items)

def main():
    

    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, default="data_plot/", help='csv folder') 
    parser.add_argument('--keys', type=str, default="eval/episode_reward,eval/success_rate", help='Data to be saved')
    parser.add_argument('--x-key', type=str, default="buffer_size", help='X axis key')
    parser.add_argument('--filter_by_tag', type=str, default="benchmark", help='Filter by tag')
    parser.add_argument('--download', action='store_true', default=True)
    parser.add_argument('--processing', action='store_true', default=True)
    parser.add_argument('--project', type=str, default='taco_metaworld', help='csv folder') 
    parser.add_argument('--n_points', type=int, default=1000, help='Number of points to plot')
    parser.add_argument('--max_x', type=int, default=200_000, help='maximum x axis value of points to plot')

    args = parser.parse_args()
    keys = args.keys.split(",")

    print("Start Downloading")
    if args.download:
        
        os.makedirs(args.csv_path, exist_ok=True)
        api = wandb.Api()
        runs = api.runs(args.project) 
        runs = [run for run in runs if run.tags and args.filter_by_tag in run.tags]
        all_keys = keys.copy()
        all_keys.append(args.x_key)
        for run in runs:
            print(flatten_dict(run.config)["agent/pretrained_path"])
            
            history = run.history(keys=all_keys)  


            run_name = f'{flatten_dict(run.config)["agent/pretrained_path"].split("/")[-1]}___{run.id}'
            history.to_csv(f"{args.csv_path}/{run_name}.csv")
            print(f"saved {run_name}.csv")
            
        print("saving DONE")
    
    print("Start Processing")
    if args.processing:
        
        directory = args.csv_path
        print("reading...")
        runs = read_csvs_from_directory(directory)
        filenames = os.listdir(directory)
        print("reading DONE")
        for run, filename in zip(runs, filenames):
            
            print("processing: ", filename)
            history = run
           
            x_axis = np.linspace(int(history.iloc[0][args.x_key]), int(history.iloc[-1][args.x_key]) if args.max_x is None else args.max_x, args.n_points, dtype = int)
            new_history = pd.DataFrame({
                    str(args.x_key): x_axis,
                })
            
            # Convert the 'buffer_size' column to numeric
            for key in keys:
                
                f = interp1d(history[args.x_key], history[key], fill_value="extrapolate")
                new_history[key] = f(x_axis)

            print("saving...")
            new_history.to_csv(f"{directory}/{filename}")
            
            print("saving DONE")
        
        # all_datasets = preprocess_data(all_datasets, args.k)
  

if __name__ == "__main__":
    main()