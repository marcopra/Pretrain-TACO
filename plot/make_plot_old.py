import wandb

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import json
import os
# from tueplots.bundles import neurips2024
import os.path as osp
import numpy as np
from scipy.interpolate import interp1d


DIV_LINE_WIDTH = 50

# Global vars for tracking and labeling data at load time.
exp_idx = 0
units = dict()

# def get_datasets(logdir, condition=None):
#     """
#     Recursively look through logdir for output files produced by
#     spinup.logx.Logger. 

#     Assumes that any file "progress.txt" is a valid hit. 
#     """
#     global exp_idx
#     global units
#     datasets = []
#     for root, _, files in os.walk(logdir):
#         if 'progress.txt' in files:
#             exp_name = None
#             try:
#                 config_path = open(os.path.join(root,'config.json'))
#                 config = json.load(config_path)
#                 if 'exp_name' in config:
#                     exp_name = config['exp_name']
#             except:
#                 print('No file named config.json')
#             condition1 = condition or exp_name or 'exp'
#             condition2 = condition1 + '-' + str(exp_idx)
#             exp_idx += 1
#             if condition1 not in units:
#                 units[condition1] = 0
#             unit = units[condition1]
#             units[condition1] += 1

#             try:
#                 exp_data = pd.read_table(os.path.join(root,'progress.txt'))
#             except:
#                 print('Could not read from %s'%os.path.join(root,'progress.txt'))
#                 continue
#             performance = 'AverageTestEpRet' if 'AverageTestEpRet' in exp_data else 'AverageEpRet'
#             exp_data.insert(len(exp_data.columns),'Unit',unit)
#             exp_data.insert(len(exp_data.columns),'Condition1',condition1)
#             exp_data.insert(len(exp_data.columns),'Condition2',condition2)
#             exp_data.insert(len(exp_data.columns),'Performance',exp_data[performance])
#             datasets.append(exp_data)
#     return datasets


# def get_all_datasets(all_logdirs, legend=None, select=None, exclude=None):
#     """
#     For every entry in all_logdirs,
#         1) check if the entry is a real directory and if it is, 
#            pull data from it; 

#         2) if not, check to see if the entry is a prefix for a 
#            real directory, and pull data from that.
#     """
#     logdirs = []
#     for logdir in all_logdirs:
#         if osp.isdir(logdir) and logdir[-1]==os.sep:
#             logdirs += [logdir]
#         else:
#             basedir = osp.dirname(logdir)
#             fulldir = lambda x : osp.join(basedir, x)
#             prefix = logdir.split(os.sep)[-1]
#             listdir= os.listdir(basedir)
#             logdirs += sorted([fulldir(x) for x in listdir if prefix in x])

#     """
#     Enforce selection rules, which check logdirs for certain substrings.
#     Makes it easier to look at graphs from particular ablations, if you
#     launch many jobs at once with similar names.
#     """
#     if select is not None:
#         logdirs = [log for log in logdirs if all(x in log for x in select)]
#     if exclude is not None:
#         logdirs = [log for log in logdirs if all(not(x in log) for x in exclude)]

#     # Verify logdirs
#     print('Plotting from...\n' + '='*DIV_LINE_WIDTH + '\n')
#     for logdir in logdirs:
#         print(logdir)
#     print('\n' + '='*DIV_LINE_WIDTH)

#     # Make sure the legend is compatible with the logdirs
#     assert not(legend) or (len(legend) == len(logdirs)), \
#         "Must give a legend title for each set of experiments."

#     # Load data from logdirs
#     data = []
#     if legend:
#         for log, leg in zip(logdirs, legend):
#             data += get_datasets(log, leg)
#     else:
#         for log in logdirs:
#             data += get_datasets(log)
#     return data


# import matplotlib.ticker as ticker

# def make_plots(data, legend=None, xaxis=None, values=None, count=False,  
#                font_scale=1.5, smooth=1, select=None, exclude=None, estimator='mean', log_scale=False, env = None):
#     values = values if isinstance(values, list) else [values]

#     estimator = getattr(np, estimator)      # choose what to show on main curve: mean? max? min?
#     for value in values:
#         plt.figure()
#         plot_data(data, xaxis=xaxis, value=value, condition="", smooth=smooth, estimator=estimator, log_scale=log_scale, env = env)
#         # neurips2024( usetex=True, rel_width=1.0, nrows=1, ncols=1, family='serif')
#         # Increase the linewidth for 'POWR (Ours)'
#         # Get all lines in the plot
#         ax = plt.gca()
#         lines = ax.get_lines()

#         # List of labels in hue_order (replace with your specific order)
#         hue_order = []



#     plt.savefig("prova.png", format='png')


# def plot_data(data, xaxis='Timesteps', value="Reward", condition="", smooth=1, log_scale=False, env = None,  **kwargs):
#     if smooth > 1:
#         y = np.ones(smooth)
#         for datum in data:
#             x = np.asarray(datum[value])
#             z = np.ones(len(x))
#             smoothed_x = np.convolve(x,y,'same') / np.convolve(z,y,'same')
#             datum[value] = smoothed_x

#     if isinstance(data, list):
#         data = pd.concat(data, ignore_index=True)
#     sns.set_style("darkgrid")
#     sns.lineplot(data=data, x=xaxis, y=value, hue=condition, hue_order=['A2C', 'DQN', 'TRPO', 'PPO', 'POWR (Ours)'], **kwargs)# TODO attenzione ai nomi di algoritmi
    
#     plt.ylabel('Reward')

#     ax = plt.gca()
#     plt.title(f"{env}")


#     if log_scale:
        
#         ax.set_xscale('log')
    
#         plt.xlabel('Timestep (logscale)')
#     else:
#         def format_func(value, ticker):
#             if np.isnan(value) or value <= 0:
#                 return value
#             else:
#                 return "{:.1f}".format(value / 10**5)


#         ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_func))
           
#         plt.xlabel('Timestep (1e5)')

#     plt.tight_layout(pad=0.5)

def preprocess_data(data_list, k):
    for data in data_list:
        if data.iloc[0]['Condition1'] == 'POWR (Ours)':
            for index, row in data.iterrows():
                if row['timestep'] > k:
                    # Get the value before modification
                    value_before = data.loc[data['timestep'] < k, 'train_mean_reward']
                    # Replace all values before k with the value at k
                    data.loc[data['timestep'] < k, 'train_mean_reward'] = -200 #row['train_mean_reward']
                    # Get the value after modification
                    value_after = data.loc[data['timestep'] < k, 'train_mean_reward']
                    print(f"Algorithm: {row['Condition1']}") # , Timestep: {row['timestep']}, Value before: {value_before}, Value after: {value_after}")
                    break
    return data_list

def read_csvs_from_directory(directory):
    """
    Read CSV files from directory and group them by algorithm name.
    Algorithm name is determined by splitting the filename by '___' and taking the first part.
    """
    grouped_dataframes = {}
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            algorithm_name = filename.split("___")[0]
            df = pd.read_csv(os.path.join(directory, filename))
            
            if algorithm_name not in grouped_dataframes:
                grouped_dataframes[algorithm_name] = []
            
            grouped_dataframes[algorithm_name].append(df)
    
    return grouped_dataframes

def aggregate_runs(dataframes, x_column, y_column, aggregator='mean'):
    """
    Aggregate multiple runs into a single dataframe.
    
    Parameters:
    - dataframes: List of dataframes to aggregate
    - x_column: Column name for x-axis
    - y_column: Column name for y-axis values to aggregate
    - aggregator: Function or string specifying how to aggregate
    
    Returns:
    - Aggregated dataframe with columns: x_column, y_mean, y_min, y_max, y_std, y_stderr
    """
    # Convert string aggregator to numpy function
    if isinstance(aggregator, str):
        if aggregator == 'mean':
            agg_func = np.mean
        elif aggregator == 'median':
            agg_func = np.median
        elif aggregator == 'min':
            agg_func = np.min
        elif aggregator == 'max':
            agg_func = np.max
        else:
            raise ValueError(f"Unknown aggregator: {aggregator}")
    else:
        agg_func = aggregator  # Use the provided function directly
    
    # Collect all x values across dataframes
    all_x_values = set()
    for df in dataframes:
        all_x_values.update(df[x_column].values)
    
    all_x_values = sorted(list(all_x_values))
    
    # Initialize arrays to store aggregated values
    y_values = []
    
    # For each x value, collect all corresponding y values from all dataframes
    for x_val in all_x_values:
        y_for_x = []
        for df in dataframes:
            # Get rows where x_column equals x_val
            rows = df[df[x_column] == x_val]
            if not rows.empty:
                y_for_x.extend(rows[y_column].values)
        
        if y_for_x:
            y_values.append(y_for_x)
        else:
            y_values.append([np.nan])
    
    # Create the result dataframe
    result = pd.DataFrame({
        x_column: all_x_values,
        'y_mean': [np.mean(y) for y in y_values],
        'y_median': [np.median(y) if len(y) > 0 and not all(np.isnan(y)) else np.nan for y in y_values],
        'y_min': [np.min(y) if len(y) > 0 and not all(np.isnan(y)) else np.nan for y in y_values],
        'y_max': [np.max(y) if len(y) > 0 and not all(np.isnan(y)) else np.nan for y in y_values],
        'y_std': [np.std(y) if len(y) > 1 and not all(np.isnan(y)) else np.nan for y in y_values],
        'y_stderr': [np.std(y) / np.sqrt(len(y)) if len(y) > 1 and not all(np.isnan(y)) else np.nan for y in y_values],
        'y_count': [len(y) for y in y_values],
        'y_aggregated': [agg_func(y) if len(y) > 0 and not all(np.isnan(y)) else np.nan for y in y_values]
    })
    
    return result

def create_value_formatter(scale_factor=None):
    """
    Create a formatter function for axis tick values.
    
    Parameters:
    - scale_factor: Value by which to divide the tick values (e.g., 1e5, 1e3)
    
    Returns:
    - A formatter function for matplotlib ticker
    """
    def format_func(value, ticker):
        if np.isnan(value) or value < 0:
            return value
        elif scale_factor is None:
            return value
        else:
            # Divide by scale factor
            scaled_value = value / scale_factor
            # Check if the value has no decimal part (is a whole number)
            if scaled_value == int(scaled_value):
                return "{:d}".format(int(scaled_value))
            else:
                return "{:.1f}".format(scaled_value)
    
    return format_func

def print_color_menu():
    """
    Stampa un menu colorato dei colori disponibili per seaborn.
    """
    # Colori base disponibili in seaborn/matplotlib
    colors = {
        'blue': '\033[94m',
        'green': '\033[92m', 
        'red': '\033[91m',
        'orange': '\033[93m',
        'purple': '\033[95m',
        'cyan': '\033[96m',
        'pink': '\033[95m',
        'brown': '\033[33m',
        'gray': '\033[90m',
        'olive': '\033[93m'
    }
    
    reset_color = '\033[0m'
    
    print("\n=== MENU COLORI DISPONIBILI ===")
    print("Seleziona un colore digitando il nome corrispondente:")
    
    for color_name, color_code in colors.items():
        print(f"{color_code}● {color_name}{reset_color}")
    
    print("\nPuoi anche usare codici esadecimali (es: #FF5733) o nomi matplotlib standard.")
    print("====================================\n")

def load_or_create_color_mapping(csv_path, algorithm_names, force_color=False):
    """
    Carica o crea un mapping dei colori per gli algoritmi.
    
    Parameters:
    - csv_path: Percorso della cartella contenente i CSV
    - algorithm_names: Lista dei nomi degli algoritmi (dopo rinominazione)
    - force_color: Se True, ignora il file esistente e chiede nuovi colori
    
    Returns:
    - Dizionario con il mapping nome_algoritmo -> colore
    """
    mapping_file = os.path.join(csv_path, "algorithm_color_mapping.txt")
    mapping = {}
    
    # Se force_color è True, chiedi sempre all'utente i nuovi colori
    if force_color:
        print("Force color attivato: creazione di un nuovo mapping colori...")
        if os.path.exists(mapping_file):
            attempts = 0
            max_attempts = 3
            
            while attempts < max_attempts:
                response = input(f"Il file {mapping_file} esiste già. Vuoi sovrascriverlo? [y/n]: ").strip().lower()
                if response == 'y':
                    break
                elif response == 'n':
                    print("Operazione annullata.")
                    return {}  # Restituisci mapping vuoto per usare colori default
                else:
                    attempts += 1
                    print(f"Risposta non valida. Tentativi rimanenti: {max_attempts - attempts}")
            
            if attempts == max_attempts:
                raise ValueError("Troppi tentativi falliti. Operazione annullata.")
        
        # Crea nuovo mapping da zero
        print(f"Algoritmi per cui assegnare colori: {algorithm_names}")
        for alg_name in algorithm_names:
            mapping[alg_name] = get_algorithm_color(alg_name)
        
        # Salva il mapping
        save_color_mapping(mapping_file, mapping, overwrite=True)
        return mapping
    
    # Prova a caricare il mapping esistente se force_color è False
    if os.path.exists(mapping_file):
        try:
            with open(mapping_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and ':' in line:
                        algorithm, color = line.split(':', 1)
                        mapping[algorithm.strip()] = color.strip()
            print(f"Caricato mapping colori esistente da {mapping_file}")
            
            # Verifica se tutti gli algoritmi sono nel mapping
            missing_algorithms = [alg for alg in algorithm_names if alg not in mapping]
            if missing_algorithms:
                print(f"Algoritmi mancanti nel mapping colori: {missing_algorithms}")
                # Aggiungi gli algoritmi mancanti
                for alg in missing_algorithms:
                    mapping[alg] = get_algorithm_color(alg)
                save_color_mapping(mapping_file, mapping, overwrite=True)
            
            return mapping
        except Exception as e:
            print(f"Errore nel caricamento del mapping colori: {e}")
            print("Creazione di un nuovo mapping colori...")
    
    # Crea un nuovo mapping
    print("Creazione mapping colori algoritmi...")
    print(f"Algoritmi per cui assegnare colori: {algorithm_names}")
    
    for alg_name in algorithm_names:
        mapping[alg_name] = get_algorithm_color(alg_name)
    
    # Salva il mapping
    save_color_mapping(mapping_file, mapping, overwrite=False)
    
    return mapping

def get_algorithm_color(algorithm_name):
    """
    Chiede all'utente che colore assegnare a un algoritmo e conferma la scelta.
    
    Parameters:
    - algorithm_name: Nome dell'algoritmo
    
    Returns:
    - Colore confermato dall'utente
    """
    while True:
        print_color_menu()
        color = input(f"Che colore vuoi assegnare a '{algorithm_name}'? ").strip()
        if not color:
            print("Devi inserire un colore. Riprova...")
            continue
        
        # Chiedi conferma
        confirm = input(f"Confermi di assegnare il colore '{color}' a '{algorithm_name}'? [Y/n]: ").strip().lower()
        if confirm == '' or confirm == 'y':
            return color
        elif confirm == 'n':
            print("Riprova...")
            continue
        else:
            print("Risposta non valida. Riprova...")

def save_color_mapping(mapping_file, mapping, overwrite=False):
    """
    Salva il mapping dei colori in un file.
    
    Parameters:
    - mapping_file: Percorso del file di mapping
    - mapping: Dizionario con il mapping
    - overwrite: Se True, forza la sovrascrittura senza chiedere
    """
    should_save = True
    
    if os.path.exists(mapping_file) and not overwrite:
        attempts = 0
        max_attempts = 3
        
        while attempts < max_attempts:
            response = input(f"Il file {mapping_file} esiste già. Vuoi sovrascriverlo? [y/n]: ").strip().lower()
            if response == 'y':
                should_save = True
                break
            elif response == 'n':
                should_save = False
                break
            else:
                attempts += 1
                print(f"Risposta non valida. Tentativi rimanenti: {max_attempts - attempts}")
        
        if attempts == max_attempts:
            raise ValueError("Troppi tentativi falliti. Operazione annullata.")
    
    if should_save:
        with open(mapping_file, 'w') as f:
            for algorithm, color in mapping.items():
                f.write(f"{algorithm}: {color}\n")
        print(f"Mapping colori salvato in {mapping_file}")
    else:
        print("Mapping colori non salvato.")

def plot_data_with_ci(aggregated_data, xaxis='buffer_size', yaxis='y_aggregated', 
                     ci_type='std_err', log_x=False, log_y=False, 
                     x_min=None, x_max=None, y_min=None, y_max=None,
                     x_scale_factor=None, y_scale_factor=None,
                     x_label=None, y_label=None, plot_title=None, 
                     color_mapping=None, **kwargs):
    """
    Plot aggregated data with confidence intervals.
    
    Parameters:
    - aggregated_data: Dictionary where keys are algorithm names and values are 
                      aggregated dataframes with statistics
    - xaxis: Column name for x-axis
    - yaxis: Column to use for the main plot line (typically 'y_aggregated')
    - ci_type: Type of confidence interval to show ('min_max', 'std', 'std_err', 'samples', or None)
    - log_x: Whether to use logarithmic scale for x-axis
    - log_y: Whether to use logarithmic scale for y-axis
    - x_min, x_max: Limits for x-axis
    - y_min, y_max: Limits for y-axis
    - x_scale_factor: Value to divide x-axis ticks by (e.g., 1e5)
    - y_scale_factor: Value to divide y-axis ticks by (e.g., 1e3)
    - x_label: Custom x-axis label
    - y_label: Custom y-axis label
    - plot_title: Custom plot title
    - color_mapping: Dictionary mapping algorithm names to colors
    """
    sns.set_style("darkgrid")
    plt.figure(figsize=(10, 6))
    
    # Use custom colors if provided, otherwise use default seaborn palette
    if color_mapping:
        colors = [color_mapping.get(alg, sns.color_palette("tab10")[i]) 
                 for i, alg in enumerate(aggregated_data.keys())]
    else:
        colors = sns.color_palette("tab10", len(aggregated_data))
    
    for i, (algorithm, data) in enumerate(aggregated_data.items()):
        color = colors[i]
        
        # Plot the main line
        plt.plot(data[xaxis], data[yaxis], label=algorithm, color=color, linewidth=2)
        
        # Add confidence interval
        if ci_type == 'min_max':
            plt.fill_between(data[xaxis], data['y_min'], data['y_max'], 
                             color=color, alpha=0.2)
        elif ci_type == 'std':
            plt.fill_between(data[xaxis], data[yaxis] - data['y_std'], 
                             data[yaxis] + data['y_std'], color=color, alpha=0.2)
        elif ci_type == 'std_err':
            plt.fill_between(data[xaxis], data[yaxis] - data['y_stderr'], 
                             data[yaxis] + data['y_stderr'], color=color, alpha=0.2)
        elif ci_type == 'samples' and 'y_count' in data.columns:
            # Adjust alpha based on sample count
            plt.fill_between(data[xaxis], data['y_min'], data['y_max'], 
                             color=color, alpha=0.1)
    
    ax = plt.gca()
    
    # Set x-axis properties
    if log_x:
        ax.set_xscale('log')
        x_axis_label = x_label or 'Timestep (logscale)'
        if x_label and x_scale_factor is not None:
            x_scale_notation = f"1e{int(np.log10(x_scale_factor))}"
            x_axis_label = f"{x_label} (logscale, {x_scale_notation})"
    else:
        # Format x-axis ticks if scale factor is provided
        if x_scale_factor is not None:
            x_formatter = create_value_formatter(x_scale_factor)
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(x_formatter))
            x_scale_notation = f"1e{int(np.log10(x_scale_factor))}"
            
            # If custom label provided, append scale notation
            if x_label:
                x_axis_label = f"{x_label} ({x_scale_notation})"
            else:
                x_axis_label = f"{xaxis.capitalize()} ({x_scale_notation})"
        else:
            x_axis_label = x_label or xaxis.capitalize()
    
    plt.xlabel(x_axis_label)
    
    # Set y-axis properties
    if log_y:
        ax.set_yscale('log')
        y_axis_label = y_label or 'Reward (logscale)'
        if y_label and y_scale_factor is not None:
            y_scale_notation = f"1e{int(np.log10(y_scale_factor))}"
            y_axis_label = f"{y_label} (logscale, {y_scale_notation})"
    else:
        # Format y-axis ticks if scale factor is provided
        if y_scale_factor is not None:
            y_formatter = create_value_formatter(y_scale_factor)
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(y_formatter))
            y_scale_notation = f"1e{int(np.log10(y_scale_factor))}"
            
            # If custom label provided, append scale notation
            if y_label:
                y_axis_label = f"{y_label} ({y_scale_notation})"
            else:
                y_axis_label = f"Reward ({y_scale_notation})"
        else:
            y_axis_label = y_label or 'Reward'
    
    plt.ylabel(y_axis_label)
    
    # Set custom title or default
    if plot_title is not None:
        plt.title(plot_title)
    
    # Set axis limits if provided
    if x_min is not None:
        ax.set_xlim(left=x_min)
    if x_max is not None:
        ax.set_xlim(right=x_max)
    if y_min is not None:
        ax.set_ylim(bottom=y_min)
    if y_max is not None:
        ax.set_ylim(top=y_max)
    
    plt.legend()
    plt.tight_layout(pad=0.5)

def apply_name_mapping(grouped_dataframes, name_mapping):
    """
    Applica il mapping dei nomi ai dataframe raggruppati.
    Aggrega i dataframe per i nomi rinominati se ci sono duplicati.
    """
    renamed_dataframes = {}
    
    for original_name, dataframes in grouped_dataframes.items():
        new_name = name_mapping.get(original_name, original_name)
        
        # Se il nome rinominato esiste già, aggiungi i dataframe a quelli esistenti
        if new_name in renamed_dataframes:
            renamed_dataframes[new_name].extend(dataframes)
        else:
            renamed_dataframes[new_name] = dataframes.copy()
    
    # Stampa informazioni sull'aggregazione
    print("\n=== AGGREGAZIONE ALGORITMI ===")
    for renamed, dataframes in renamed_dataframes.items():
        original_names = [orig for orig, new in name_mapping.items() if new == renamed]
        if not original_names:  # Se non c'è mapping, usa il nome stesso
            original_names = [renamed]
        print(f"'{renamed}': {len(dataframes)} run da {len(original_names)} algoritmi originali")
        if len(original_names) > 1:
            print(f"  Algoritmi originali: {original_names}")
    print("===============================\n")
    
    return renamed_dataframes

def load_or_create_name_mapping(csv_path, algorithm_names, force_rename=False):
    """
    Carica o crea un mapping dei nomi degli algoritmi.
    """
    mapping_file = os.path.join(csv_path, "algorithm_name_mapping.txt")
    mapping = {}
    
    if force_rename:
        print("Force rename attivato: creazione di un nuovo mapping...")
        if os.path.exists(mapping_file):
            attempts = 0
            max_attempts = 3
            
            while attempts < max_attempts:
                response = input(f"Il file {mapping_file} esiste già. Vuoi sovrascriverlo? [y/n]: ").strip().lower()
                if response == 'y':
                    break
                elif response == 'n':
                    print("Operazione annullata.")
                    return {name: name for name in algorithm_names}
                else:
                    attempts += 1
                    print(f"Risposta non valida. Tentativi rimanenti: {max_attempts - attempts}")
            
            if attempts == max_attempts:
                raise ValueError("Troppi tentativi falliti. Operazione annullata.")
        
        print(f"Algoritmi trovati: {algorithm_names}")
        for alg_name in algorithm_names:
            mapping[alg_name] = get_renamed_algorithm(alg_name)
        
        save_name_mapping(mapping_file, mapping, overwrite=True)
        return mapping
    
    if os.path.exists(mapping_file):
        try:
            with open(mapping_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and ':' in line:
                        original, renamed = line.split(':', 1)
                        original = original.strip()
                        renamed = renamed.strip()
                        
                        # Solo aggiungi il mapping se l'algoritmo originale esiste
                        if original in algorithm_names:
                            mapping[original] = renamed
            
            print(f"Caricato mapping esistente da {mapping_file}")
            
            missing_algorithms = [alg for alg in algorithm_names if alg not in mapping]
            if missing_algorithms:
                print(f"Algoritmi mancanti nel mapping: {missing_algorithms}")
                for alg in missing_algorithms:
                    mapping[alg] = get_renamed_algorithm(alg)
                save_name_mapping(mapping_file, mapping, overwrite=True)
            
            return mapping
        except Exception as e:
            print(f"Errore nel caricamento del mapping: {e}")
            print("Creazione di un nuovo mapping...")
    
    print("Creazione mapping nomi algoritmi...")
    print(f"Algoritmi trovati: {algorithm_names}")
    
    for alg_name in algorithm_names:
        mapping[alg_name] = get_renamed_algorithm(alg_name)
    
    save_name_mapping(mapping_file, mapping, overwrite=False)
    return mapping

def get_renamed_algorithm(original_name):
    """
    Chiede all'utente come rinominare un algoritmo e conferma la scelta.
    
    Parameters:
    - original_name: Nome originale dell'algoritmo
    
    Returns:
    - Nome rinominato confermato dall'utente
    """
    while True:
        new_name = input(f"Come vuoi rinominare '{original_name}'? (premi invio per mantenere il nome originale): ").strip()
        if not new_name:
            new_name = original_name
        
        # Chiedi conferma
        confirm = input(f"Confermi di rinominare '{original_name}' in '{new_name}'? [Y/n]: ").strip().lower()
        if confirm == '' or confirm == 'y':
            return new_name
        elif confirm == 'n':
            print("Riprova...")
            continue
        else:
            print("Risposta non valida. Riprova...")

def save_name_mapping(mapping_file, mapping, overwrite=False):
    """
    Salva il mapping dei nomi in un file.
    
    Parameters:
    - mapping_file: Percorso del file di mapping
    - mapping: Dizionario con il mapping
    - overwrite: Se True, forza la sovrascrittura senza chiedere
    """
    should_save = True
    
    if os.path.exists(mapping_file) and not overwrite:
        attempts = 0
        max_attempts = 3
        
        while attempts < max_attempts:
            response = input(f"Il file {mapping_file} esiste già. Vuoi sovrascriverlo? [y/n]: ").strip().lower()
            if response == 'y':
                should_save = True
                break
            elif response == 'n':
                should_save = False
                break
            else:
                attempts += 1
                print(f"Risposta non valida. Tentativi rimanenti: {max_attempts - attempts}")
        
        if attempts == max_attempts:
            raise ValueError("Troppi tentativi falliti. Operazione annullata.")
    
    if should_save:
        with open(mapping_file, 'w') as f:
            for original, renamed in mapping.items():
                f.write(f"{original}: {renamed}\n")
        print(f"Mapping salvato in {mapping_file}")
    else:
        print("Mapping non salvato.")

def apply_name_mapping(grouped_dataframes, name_mapping):
    """
    Applica il mapping dei nomi ai dataframe raggruppati.
    Aggrega i dataframe per i nomi rinominati se ci sono duplicati.
    """
    renamed_dataframes = {}
    
    for original_name, dataframes in grouped_dataframes.items():
        new_name = name_mapping.get(original_name, original_name)
        
        # Se il nome rinominato esiste già, aggiungi i dataframe a quelli esistenti
        if new_name in renamed_dataframes:
            renamed_dataframes[new_name].extend(dataframes)
        else:
            renamed_dataframes[new_name] = dataframes.copy()
    
    # Stampa informazioni sull'aggregazione
    print("\n=== AGGREGAZIONE ALGORITMI ===")
    for renamed, dataframes in renamed_dataframes.items():
        original_names = [orig for orig, new in name_mapping.items() if new == renamed]
        if not original_names:  # Se non c'è mapping, usa il nome stesso
            original_names = [renamed]
        print(f"'{renamed}': {len(dataframes)} run da {len(original_names)} algoritmi originali")
        if len(original_names) > 1:
            print(f"  Algoritmi originali: {original_names}")
    print("===============================\n")
    
    return renamed_dataframes

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--xaxis', '-x', default='buffer_size', help='Column name for x-axis data')
    parser.add_argument('--value', '-y', default='eval/episode_reward', help='Column name for y-axis data')
    parser.add_argument('--est', default='median', choices=['mean', 'median', 'min', 'max'], 
                        help='Aggregation function')
    parser.add_argument('--ci', default='std_err', choices=['min_max', 'std', 'std_err', 'samples', 'none'],
                        help='Confidence interval type')
    
    # Axis scaling options
    parser.add_argument('--log_x', action='store_true', default=False, help='Use log scale for x-axis')
    parser.add_argument('--log_y', action='store_true', default=False, help='Use log scale for y-axis')
    
    # Axis limits
    parser.add_argument('--x_min', type=float, default=0, help='Minimum value for x-axis')
    parser.add_argument('--x_max', type=float, default=100_000, help='Maximum value for x-axis')
    parser.add_argument('--y_min', type=float, default=0, help='Minimum value for y-axis')
    parser.add_argument('--y_max', type=float, default=None, help='Maximum value for y-axis')
    
    # Axis formatting
    parser.add_argument('--x_scale', type=float, default=1e4, help='Scale factor for x-axis values (e.g., 1e5)')
    parser.add_argument('--y_scale', type=float, default=None, help='Scale factor for y-axis values (e.g., 1e3)')
    parser.add_argument('--x_label', type=str, default='Buffer Size', help='Custom x-axis label')
    parser.add_argument('--y_label', type=str, default=None, help='Custom y-axis label')
    
    # Plot title
    parser.add_argument('--title', type=str, default=None, help='Plot title')
    
    parser.add_argument('--csv_path', type=str, default="data_plot/", help='csv folder')
    parser.add_argument('--output', type=str, default="plot.png", help='Output filename')
    
    # Rename options
    parser.add_argument('--rename', action='store_true', default=False, 
                        help='Enable algorithm name renaming with interactive input')
    parser.add_argument('--force-rename', action='store_true', default=False,
                        help='Force renaming even if mapping file exists')
    
    # Color mapping options
    parser.add_argument('--color', action='store_true', default=False,
                        help='Enable algorithm color mapping with interactive input')
    parser.add_argument('--force-color', action='store_true', default=False,
                        help='Force color mapping even if mapping file exists')
    
    args = parser.parse_args()

    # Read and group CSV files by algorithm
    grouped_dataframes = read_csvs_from_directory(args.csv_path)
    
    # Apply renaming if requested
    if args.rename:
        algorithm_names = list(grouped_dataframes.keys())
        name_mapping = load_or_create_name_mapping(
            args.csv_path, 
            algorithm_names, 
            force_rename=getattr(args, 'force_rename', False)
        )
        grouped_dataframes = apply_name_mapping(grouped_dataframes, name_mapping)
    
    # Apply color mapping if requested
    color_mapping = None
    if args.color:
        algorithm_names = list(grouped_dataframes.keys())  # Usa i nomi dopo la rinominazione
        color_mapping = load_or_create_color_mapping(
            args.csv_path,
            algorithm_names,
            force_color=getattr(args, 'force_color', False)
        )
    
    # Aggregate runs for each algorithm
    aggregated_data = {}
    for algorithm, dataframes in grouped_dataframes.items():
        aggregated_data[algorithm] = aggregate_runs(
            dataframes, 
            x_column=args.xaxis, 
            y_column=args.value, 
            aggregator=args.est
        )
    
    # Plot the aggregated data with confidence intervals
    print("Creating plot...")
    plot_data_with_ci(
        aggregated_data, 
        xaxis=args.xaxis, 
        yaxis='y_aggregated',
        ci_type=args.ci if args.ci.lower() != 'none' else None,
        log_x=args.log_x,
        log_y=args.log_y,
        x_min=args.x_min,
        x_max=args.x_max,
        y_min=args.y_min,
        y_max=args.y_max,
        x_scale_factor=args.x_scale,
        y_scale_factor=args.y_scale,
        x_label=args.x_label,
        y_label=args.y_label,
        plot_title=args.title,
        color_mapping=color_mapping
    )
    
    # Create output path in the same folder as csv_path
    output_path = os.path.join(args.csv_path, args.output)
    
    # Save the plot
    plt.savefig(output_path, format='png', dpi=300)
    print(f"Plot saved as {output_path}")

if __name__ == "__main__":
    main()