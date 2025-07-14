"""
python /home/mprattico/Pretrain-TACO/plot/download_data_from_wandb.py --csv_path data_plot/shelf-place/exp --filter_by_config agent/no_taco!=true,env_name=shelf-place-v2 --filter_by_tags MT50 --group_by_config agent/pretrained_path
# NO MT20 because not finished yet
python /home/mprattico/Pretrain-TACO/plot/download_data_from_wandb.py --csv_path data_plot/mt/basketball --filter_by_config agent/no_taco!=true,env_name=basketball-v2 --filter_by_tags MT1,MT10,MT30,MT40,MT45 --group_by_config agent/pretrained_path
python /home/mprattico/Pretrain-TACO/plot/download_data_from_wandb.py --csv_path data_plot/mt/bin-picking --filter_by_config agent/no_taco!=true,env_name=bin-picking-v2 --filter_by_tags MT1,MT10,MT30,MT40,MT45 --group_by_config agent/pretrained_path
python /home/mprattico/Pretrain-TACO/plot/download_data_from_wandb.py --csv_path data_plot/mt/button-press --filter_by_config agent/no_taco!=true,env_name=button-press-v2 --filter_by_tags MT1,MT10,MT30,MT40,MT45 --group_by_config agent/pretrained_path
python /home/mprattico/Pretrain-TACO/plot/download_data_from_wandb.py --csv_path data_plot/mt/push --filter_by_config agent/no_taco!=true,env_name=push-v2 --filter_by_tags MT1,MT10,MT30,MT40,MT45 --group_by_config agent/pretrained_path
python /home/mprattico/Pretrain-TACO/plot/download_data_from_wandb.py --csv_path data_plot/mt/shelf-place --filter_by_config agent/no_taco!=true,env_name=shelf-place-v2 --filter_by_tags MT1,MT10,MT30,MT40,MT45 --group_by_config agent/pretrained_path

python /home/mprattico/Pretrain-TACO/plot/download_data_from_wandb.py --csv_path data_plot/mt/basketball --filter_by_config agent/no_taco!=true,env_name=basketball-v2 --filter_by_tags     MT20 --group_by_config agent/pretrained_path
python /home/mprattico/Pretrain-TACO/plot/download_data_from_wandb.py --csv_path data_plot/mt/bin-picking --filter_by_config agent/no_taco!=true,env_name=bin-picking-v2 --filter_by_tags   MT20 --group_by_config agent/pretrained_path
python /home/mprattico/Pretrain-TACO/plot/download_data_from_wandb.py --csv_path data_plot/mt/button-press --filter_by_config agent/no_taco!=true,env_name=button-press-v2 --filter_by_tags MT20 --group_by_config agent/pretrained_path
python /home/mprattico/Pretrain-TACO/plot/download_data_from_wandb.py --csv_path data_plot/mt/push --filter_by_config agent/no_taco!=true,env_name=push-v2 --filter_by_tags                 MT20 --group_by_config agent/pretrained_path
python /home/mprattico/Pretrain-TACO/plot/download_data_from_wandb.py --csv_path data_plot/mt/shelf-place --filter_by_config agent/no_taco!=true,env_name=shelf-place-v2 --filter_by_tags   MT20 --group_by_config agent/pretrained_path

python /home/mprattico/Pretrain-TACO/plot/download_data_from_wandb.py --csv_path data_plot/mt/all --filter_by_config agent/no_taco!=true,env_name=basketball-v2 --filter_by_tags     MT20 --group_by_config agent/pretrained_path
python /home/mprattico/Pretrain-TACO/plot/download_data_from_wandb.py --csv_path data_plot/mt/all --filter_by_config agent/no_taco!=true,env_name=bin-picking-v2 --filter_by_tags   MT20 --group_by_config agent/pretrained_path
python /home/mprattico/Pretrain-TACO/plot/download_data_from_wandb.py --csv_path data_plot/mt/all --filter_by_config agent/no_taco!=true,env_name=button-press-v2 --filter_by_tags MT20 --group_by_config agent/pretrained_path
python /home/mprattico/Pretrain-TACO/plot/download_data_from_wandb.py --csv_path data_plot/mt/all --filter_by_config agent/no_taco!=true,env_name=shelf-place-v2 --filter_by_tags   MT20 --group_by_config agent/pretrained_path
python /home/mprattico/Pretrain-TACO/plot/download_data_from_wandb.py --csv_path data_plot/mt/all --filter_by_config agent/no_taco!=true,env_name=push-v2 --filter_by_tags                 MT20 --group_by_config agent/pretrained_path

python /home/mprattico/Pretrain-TACO/plot/download_data_from_wandb.py --csv_path data_plot/push --project taco_metaworld_resnet_ed4ct --group_by_config agent/pretrained_path
python /home/mprattico/Pretrain-TACO/plot/download_data_from_wandb.py --csv_path data_plot/push --project taco_metaworld_resnet_ed4ct --filter_by_config env_name=push-v3,batch_size=1024 --group_by_config agent/pretrained_path

python /home/mprattico/Pretrain-TACO/plot/download_data_from_wandb.py --csv_path data_plot/exp/basketball --filter_by_config agent/no_taco!=true,env_name=basketball-v2 --filter_by_tags     MT50 --group_by_config agent/pretrained_path --project taco_metaworld_debug 
python /home/mprattico/Pretrain-TACO/plot/download_data_from_wandb.py --csv_path data_plot/exp/bin-picking --filter_by_config agent/no_taco!=true,env_name=bin-picking-v2 --filter_by_tags   MT50 --group_by_config agent/pretrained_path --project taco_metaworld_debug
python /home/mprattico/Pretrain-TACO/plot/download_data_from_wandb.py --csv_path data_plot/exp/button-press --filter_by_config agent/no_taco!=true,env_name=button-press-v2 --filter_by_tags MT50 --group_by_config agent/pretrained_path --project taco_metaworld_debug
python /home/mprattico/Pretrain-TACO/plot/download_data_from_wandb.py --csv_path data_plot/exp/push --filter_by_config agent/no_taco!=true,env_name=push-v2 --filter_by_tags                 MT50 --group_by_config agent/pretrained_path --project taco_metaworld_debug
python /home/mprattico/Pretrain-TACO/plot/download_data_from_wandb.py --csv_path data_plot/exp/shelf-place --filter_by_config agent/no_taco!=true,env_name=shelf-place-v2 --filter_by_tags   MT50 --group_by_config agent/pretrained_path --project taco_metaworld_debug

"""
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
    csv_filenames = []

    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            df = pd.read_csv(os.path.join(directory, filename))
            dataframes.append(df)
            csv_filenames.append(filename)
    
    return dataframes, csv_filenames

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

def check_config_match(run_config, config_filters):
    """Check if a run's config matches the specified filters."""
    flattened_config = flatten_dict(run_config)
    for key, filter_info in config_filters.items():
        operator, value = filter_info
        
        # Check if the key exists in the flattened config
        if key not in flattened_config:
            # If using != and the key doesn't exist, that's actually a match
            if operator == "!=":
                continue
            return False
        
        # Convert the config value to string for comparison
        config_value = str(flattened_config[key]).lower()
        filter_value = str(value).lower()
        
        # Apply the appropriate comparison based on operator
        if operator == "==" and config_value != filter_value:
            return False
        elif operator == "!=" and config_value == filter_value:
            return False
    
    return True

def parse_config_filters(config_filter_str):
    """Parse the config filter string into a dictionary."""
    if not config_filter_str:
        return {}
    
    filters = {}
    filter_pairs = config_filter_str.split(',')
    for pair in filter_pairs:
        # Check for inequality operator
        if "!=" in pair:
            key, value = pair.split("!=", 1)
            operator = "!="
        # Check for equality operator (explicit or implicit)
        elif "==" in pair:
            key, value = pair.split("==", 1)
            operator = "=="
        elif "=" in pair:
            key, value = pair.split("=", 1)
            operator = "=="
        else:
            continue
            
        # Try to convert value to appropriate type
        if value.lower() == 'true':
            value = True
        elif value.lower() == 'false':
            value = False
        elif value.isdigit():
            value = int(value)
        elif value.replace('.', '', 1).isdigit() and value.count('.') <= 1:
            value = float(value)
        
        # Store both the operator and the value
        filters[key.strip()] = (operator, value)
    
    return filters

def parse_tag_filters(tags_filter_str):
    """Parse tag filter string into inclusion and exclusion lists."""
    if not tags_filter_str:
        return [], []
    
    include_tags = []
    exclude_tags = []
    
    tag_filters = tags_filter_str.split(',')
    for tag_filter in tag_filters:
        if tag_filter.endswith('!='):
            # Remove the != operator and add to exclude list
            exclude_tags.append(tag_filter[:-2].strip())
        elif '!=' in tag_filter:
            # Format: "tag!=value"
            exclude_tags.append(tag_filter.split('!=')[1].strip())
        elif tag_filter.endswith('='):
            # Format: "tag="
            include_tags.append(tag_filter[:-1].strip())
        elif '=' in tag_filter:
            # Format: "tag=value"
            include_tags.append(tag_filter.split('=')[1].strip())
        else:
            # Simple tag name
            include_tags.append(tag_filter.strip())
    
    return include_tags, exclude_tags

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, default="data_plot/", help='csv folder') 
    parser.add_argument('--keys', type=str, default="eval/episode_reward,eval/success_rate", help='Data to be saved')
    parser.add_argument('--x-key', type=str, default="buffer_size", help='X axis key')
    parser.add_argument('--filter_by_tags', type=str, default="", 
                      help='Filter by tags (comma-separated list). Use tag!= to exclude a tag')
    parser.add_argument('--filter_by_config', type=str, default="", help='Filter by config parameters (format: key1=value1,key2=value2)')
    parser.add_argument('--download', action='store_true', default=True, help='Download data from wandb')
    parser.add_argument('--processing', action='store_true', default=True)
    parser.add_argument('--project', type=str, default='taco_metaworld', help='Project name') 
    parser.add_argument('--entity', type=str, default=None, help='WandB entity/team name (optional)')
    parser.add_argument('--n_points', type=int, default=1000, help='Number of points to plot')
    parser.add_argument('--max_x', type=int, default=100_000, help='maximum x axis value of points to plot')
    parser.add_argument('--min_x', type=int, default=0, help='minimum x axis value of points to plot')
    parser.add_argument('--group_by_config', type=str, default="pretrained_path", help='Config parameter to use for grouping and naming saved files')
    parser.add_argument('--max_runs_per_group', type=int, default=7, 
                        help='Maximum number of runs to download per group (based on group_by_config). If None, download all runs')

    args = parser.parse_args()
    keys = args.keys.split(",")
    include_tags, exclude_tags = parse_tag_filters(args.filter_by_tags)
    config_filters = parse_config_filters(args.filter_by_config)

    print("Start Downloading")
    if args.download:
        
        os.makedirs(args.csv_path, exist_ok=True)
        
        # Initialize WandB API with debugging
        try:
            api = wandb.Api()
            print(f"WandB API inizializzata correttamente")
            print(f"Current user: {api.viewer}")
        except Exception as e:
            print(f"Errore nell'inizializzazione dell'API WandB: {e}")
            print("Prova a fare login con: wandb login")
            return
        
        # Construct project path - if no entity specified, try with user's entity
        if args.entity:
            project_path = f"{args.entity}/{args.project}"
        else:
            # Try to get user's default entity
            try:
                user_entity = api.viewer['entity']
                project_path = f"{user_entity}/{args.project}"
                print(f"Nessuna entità specificata, usando l'entità dell'utente: {user_entity}")
            except:
                project_path = args.project
        
        print(f"Tentativo di accesso al progetto: {project_path}")
        
        try:
            # Get project info first
            project = api.project(project_path)
            print(f"Progetto trovato: {project.name}")
            print(f"Entità del progetto: {project.entity}")
        except Exception as e:
            print(f"Errore nell'accesso al progetto {project_path}: {e}")
            print("Progetti disponibili:")
            try:
                for proj in api.projects():
                    print(f"  - {proj.entity}/{proj.name}")
            except Exception as proj_e:
                print(f"Errore nel listare i progetti: {proj_e}")
            return
        
        # Get runs with detailed debugging
        try:
            print(f"Recupero runs dal progetto {project_path}...")
            runs = api.runs(project_path)
            runs_list = list(runs)
            print(f"Numero totale di runs trovati: {len(runs_list)}")
            
            if len(runs_list) == 0:
                print("NESSUN RUN TROVATO!")
                print("Possibili cause:")
                print("1. Il progetto non contiene run")
                print("2. Non hai i permessi per vedere i run")
                print("3. Il nome del progetto o dell'entità è errato")
                return
            else:
                print(f"Primi 5 run trovati:")
                for i, run in enumerate(runs_list[:5]):
                    print(f"  {i+1}. ID: {run.id}, Nome: {run.name}, Tags: {run.tags}")
            
        except Exception as e:
            print(f"Errore nel recupero dei runs: {e}")
            return
        
        runs = runs_list
        
        # Filter runs by tags and config
        filtered_runs = []
        for run in runs:
            # Check tags - must have at least one included tag (if any specified)
            # AND must not have any excluded tags
            tags_include_match = not include_tags or any(tag in run.tags for tag in include_tags)
            tags_exclude_match = not exclude_tags or not any(tag in run.tags for tag in exclude_tags)
            tags_match = tags_include_match and tags_exclude_match
            
            # Check if the run's config matches all specified config filters
            config_match = check_config_match(run.config, config_filters)
            print(f"Run ID: {run.id}, Tags: {run.tags}, Config: {run.config}")
            print(f"Tags match: {tags_match}, Config match: {config_match}")
            if tags_match and config_match:
                filtered_runs.append(run)
        
        print(f"Runs dopo filtrazione: {len(filtered_runs)}")
        runs = filtered_runs
        
        # Group runs by the specified config parameter if max_runs_per_group is set
        if args.max_runs_per_group is not None:
            print(f"Limiting to {args.max_runs_per_group} runs per group (grouped by {args.group_by_config})")
            
            # Group runs by the config parameter
            grouped_runs = {}
            for run in runs:
                flattened_config = flatten_dict(run.config)
                if args.group_by_config in flattened_config:
                    param_value = str(flattened_config[args.group_by_config]).split("/")[-1]
                else:
                    param_value = "unknown"
                
                if param_value not in grouped_runs:
                    grouped_runs[param_value] = []
                grouped_runs[param_value].append(run)
            
            # Limit the number of runs per group
            limited_runs = []
            for group_name, group_runs in grouped_runs.items():
                selected_runs = group_runs[:args.max_runs_per_group]
                limited_runs.extend(selected_runs)
                print(f"Group '{group_name}': selected {len(selected_runs)} out of {len(group_runs)} runs")
            
            runs = limited_runs
            print(f"Total runs after limiting: {len(runs)}")
        
        all_keys = keys.copy()
        all_keys.append(args.x_key)
        for run in runs:
            print(flatten_dict(run.config)["agent/pretrained_path"])
            
            history = run.history(keys=all_keys)  

            flattened_config = flatten_dict(run.config)
            if args.group_by_config in flattened_config:
                param_value = str(flattened_config[args.group_by_config]).split("/")[-1]
                print(f"param_value: {param_value}")
            else:
                # Fallback if the parameter doesn't exist
                param_value = "unknown"
                
            run_name = f'{param_value}___{run.id}'
            history.to_csv(f"{args.csv_path}/{run_name}.csv")
            print(f"saved {run_name}.csv")
            
        print("saving DONE")
    
    print("Start Processing")
    if args.processing:
        
        directory = args.csv_path
        print("reading...")
        runs, filenames = read_csvs_from_directory(directory)
        print("reading DONE")
        for run, filename in zip(runs, filenames):
            
            print("processing: ", filename)
            history = run
           
            # x_axis = np.linspace(int(history.iloc[0][args.x_key]), int(history.iloc[-1][args.x_key]) if args.max_x is None else args.max_x, args.n_points, dtype = int)
            
            x_axis = np.linspace(int(history.iloc[0][args.x_key]) if args.min_x is None else args.min_x, int(history.iloc[-1][args.x_key]) if args.max_x is None else args.max_x, args.n_points, dtype = int)
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