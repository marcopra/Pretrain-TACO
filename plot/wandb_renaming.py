import wandb
import os
import re
import argparse

def refactor_path(path):
    """
    Refactor the path according to the following rules:
    - If path contains 'MT50', modify to include 'MT50' folder
    - If path contains just 'MT', modify to include 'MT10' folder
    - Otherwise, modify to include 'ST' folder
    """
    # Check if path already contains MT50, MT10, or ST folders
    if '/models/MT50/' in path or '/models/MT10/' in path or '/models/ST/' in path:
        return path
    
    # Extract the base directory up to /models/
    base_dir_match = re.match(r'(.*?/models)/', path)
    if not base_dir_match:
        print(f"Warning: Path does not contain '/models/': {path}")
        return path
    
    base_dir = base_dir_match.group(1)
    
    # Extract filename and remaining path after /models/
    remaining_path = path[len(base_dir) + 1:]  # +1 for the trailing slash
    
    # Determine which subfolder to add
    if 'MT50' in path:
        subfolder = 'MT50'
    elif 'MT' in path:
        subfolder = 'MT10'
    else:
        subfolder = 'ST'
    
    # Construct the new path by inserting the subfolder
    new_path = os.path.join(base_dir, subfolder, remaining_path)
    
    # Check if the file would exist at the new path
    if not os.path.exists(new_path):
        print(f"Warning: File would not exist at refactored path: {new_path}")
        raise FileNotFoundError(f"File does not exist at refactored path: {new_path}")
    
    print(f"Refactored path: {path} -> {new_path}")
    return new_path

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Modify pretrained_path parameter in wandb runs')
    parser.add_argument('--project', required=True, type=str, help='Wandb project name')
    args = parser.parse_args()
    
    # Set up WandB API
    api = wandb.Api()
    
    project_name = args.project
    
    print(f"Fetching runs from {project_name}...")
    runs = api.runs(f"{project_name}")
    
    modified_runs = 0
    skipped_runs = 0
    
    for run in runs:
        # Skip running runs
        # if run.state == 'running':
        #     print(f"Skipping running run {run.id}")
        #     skipped_runs += 1
        #     continue
        
        # Check if agent dictionary exists and contains pretrained_path
        if 'agent' in run.config and 'pretrained_path' in run.config['agent']:
            original_path = run.config['agent']['pretrained_path']
            
            # Refactor the path
            refactored_path = refactor_path(original_path)
            
            # Skip if path didn't change
            if original_path == refactored_path:
                print(f"Path unchanged for run {run.id}: {original_path}")
                skipped_runs += 1
                continue
            
            # Check if the file exists in the refactored path
            if not os.path.exists(refactored_path):
                print(f"Warning: File does not exist at {refactored_path}")
                skipped_runs += 1
                continue
            
            # Update the pretrained_path parameter
            try:
                # Check both top-level and nested pretrained_path
                if 'pretrained_path' in run.config:
                    print(f"Updating run {run.id}: changing top-level pretrained_path from {run.config['pretrained_path']} to {refactored_path}")
                    run.config['pretrained_path'] = refactored_path
                
                # Always update the agent.pretrained_path
                print(f"Updating run {run.id}: changing agent.pretrained_path from {run.config['agent']['pretrained_path']} to {refactored_path}")
                run.config['agent']['pretrained_path'] = refactored_path
                
                run.update()
                modified_runs += 1
                print(f"Successfully updated run {run.id}")
            except Exception as e:
                print(f"Error updating run {run.id}: {e}")
                skipped_runs += 1
        else:
            print(f"Run {run.id} does not have agent.pretrained_path in config")
            skipped_runs += 1
    
    print(f"Updated {modified_runs} runs successfully. Skipped {skipped_runs} runs.")

if __name__ == "__main__":
    main()
