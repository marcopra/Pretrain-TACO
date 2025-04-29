import wandb
import argparse
import sys

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Copy agent.pretrained_path to pretrained_path if the latter is none')
    parser.add_argument('--project', required=True, type=str, help='Wandb project name')
    parser.add_argument('--dry-run', action='store_true', help='Print what would be done without making changes')
    args = parser.parse_args()
    
    # Set up WandB API
    api = wandb.Api()
    
    project_name = args.project
    dry_run = args.dry_run
    
    print(f"Fetching runs from {project_name}...")
    runs = api.runs(f"{project_name}")
    
    modified_runs = 0
    skipped_runs = 0
    
    for run in runs:
        print(f"Processing run {run.id} (state: {run.state})")
        
        # Check if agent dictionary exists and contains pretrained_path
        if 'agent' in run.config and 'pretrained_path' in run.config['agent']:
            agent_path = run.config['agent']['pretrained_path']
            
            # Check if pretrained_path is 'none' or None
            if 'pretrained_path' not in run.config or run.config['pretrained_path'] == 'none' or run.config['pretrained_path'] is None:
                print(f"Run {run.id}: agent.pretrained_path={agent_path}, pretrained_path is none or missing")
                
                if not dry_run:
                    try:
                        # Update the pretrained_path parameter
                        run.config['pretrained_path'] = agent_path
                        run.update()
                        modified_runs += 1
                        print(f"Successfully updated run {run.id}")
                    except Exception as e:
                        print(f"Error updating run {run.id}: {e}")
                        skipped_runs += 1
                else:
                    print(f"[DRY RUN] Would update run {run.id} pretrained_path to {agent_path}")
                    modified_runs += 1
            else:
                print(f"Run {run.id} already has pretrained_path set to {run.config['pretrained_path']}")
                skipped_runs += 1
        else:
            print(f"Run {run.id} does not have agent.pretrained_path in config")
            skipped_runs += 1
    
    verb = "Would update" if dry_run else "Updated"
    print(f"{verb} {modified_runs} runs successfully. Skipped {skipped_runs} runs.")
    
    if dry_run:
        print("This was a dry run. Use without --dry-run to make actual changes.")

if __name__ == "__main__":
    main()
