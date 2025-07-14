"""
python plot/make_barplot.py --csv_path data_plot/push/exp --rename --color --baseline --custom_order --bar_type std_err --error_bar_percentile 95 --fixed_x 100000 --y_min 0 --baseline_error_bars -y eval/success_rate
python plot/make_barplot.py --csv_path data_plot/exp/all --rename --color --baseline --custom_order --bar_type std_err --error_bar_percentile 95 --fixed_x 100000 --y_min 0
"""

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import json
import os
import numpy as np


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

def find_closest_x_value(df, x_column, target_x):
    """
    Trova il valore più vicino al target_x nel dataframe.
    """
    if x_column not in df.columns:
        return None
    
    x_values = df[x_column].dropna().unique()
    if len(x_values) == 0:
        return None
    
    closest_value = min(x_values, key=lambda x: abs(x - target_x))
    return closest_value

def prepare_barplot_data_at_x(grouped_dataframes, x_column, y_column, fixed_x):
    """
    Prepare data for barplot by collecting y values at a specific x value for each algorithm.
    """
    barplot_data = []
    
    print(f"Cercando dati per x = {fixed_x}...")
    
    for algorithm, dataframes in grouped_dataframes.items():
        algorithm_values = []
        
        for df in dataframes:
            if x_column in df.columns and y_column in df.columns:
                closest_x = find_closest_x_value(df, x_column, fixed_x)
                
                if closest_x is not None:
                    matching_rows = df[df[x_column] == closest_x]
                    if not matching_rows.empty:
                        y_values = matching_rows[y_column].dropna()
                        algorithm_values.extend(y_values.values)
                        
                        if closest_x != fixed_x:
                            print(f"  {algorithm}: usando x = {closest_x} (più vicino a {fixed_x})")
        
        for value in algorithm_values:
            barplot_data.append({
                'algorithm': algorithm,
                'value': value
            })
        
        if algorithm_values:
            print(f"  {algorithm}: trovati {len(algorithm_values)} valori")
        else:
            print(f"  {algorithm}: nessun valore trovato per x = {fixed_x}")
    
    return pd.DataFrame(barplot_data)

def apply_name_mapping(grouped_dataframes, name_mapping):
    """
    Applica il mapping dei nomi ai dataframe raggruppati.
    """
    renamed_dataframes = {}
    
    for original_name, dataframes in grouped_dataframes.items():
        new_name = name_mapping.get(original_name, original_name)
        
        if new_name in renamed_dataframes:
            renamed_dataframes[new_name].extend(dataframes)
        else:
            renamed_dataframes[new_name] = dataframes.copy()
    
    print("\n=== AGGREGAZIONE ALGORITMI ===")
    for renamed, dataframes in renamed_dataframes.items():
        original_names = [orig for orig, new in name_mapping.items() if new == renamed]
        if not original_names:
            original_names = [renamed]
        print(f"'{renamed}': {len(dataframes)} run da {len(original_names)} algoritmi originali")
        if len(original_names) > 1:
            print(f"  Algoritmi originali: {original_names}")
    print("===============================\n")
    
    return renamed_dataframes

def load_or_create_barplot_name_mapping(csv_path, algorithm_names, force_rename=False):
    """
    Carica o crea un mapping dei nomi degli algoritmi per barplot.
    """
    mapping_file = os.path.join(csv_path, "algorithm_name_mapping_bar.txt")
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
    
    print("Creazione mapping nomi algoritmi per barplot...")
    print(f"Algoritmi trovati: {algorithm_names}")
    
    for alg_name in algorithm_names:
        mapping[alg_name] = get_renamed_algorithm(alg_name)
    
    save_name_mapping(mapping_file, mapping, overwrite=False)
    return mapping

def get_renamed_algorithm(original_name):
    """
    Chiede all'utente come rinominare un algoritmo e conferma la scelta.
    """
    while True:
        new_name = input(f"Come vuoi rinominare '{original_name}'? (premi invio per mantenere il nome originale): ").strip()
        if not new_name:
            new_name = original_name
        
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

def load_or_create_baseline_mapping(csv_path, algorithm_names, force_baseline=False):
    """
    Carica o crea un mapping per le baseline.
    """
    mapping_file = os.path.join(csv_path, "baseline_mapping_bar.txt")
    baselines = []
    
    if force_baseline:
        print("Force baseline attivato: creazione di un nuovo mapping...")
        if os.path.exists(mapping_file):
            attempts = 0
            max_attempts = 3
            
            while attempts < max_attempts:
                response = input(f"Il file {mapping_file} esiste già. Vuoi sovrascriverlo? [y/n]: ").strip().lower()
                if response == 'y':
                    break
                elif response == 'n':
                    print("Operazione annullata.")
                    return []
                else:
                    attempts += 1
                    print(f"Risposta non valida. Tentativi rimanenti: {max_attempts - attempts}")
            
            if attempts == max_attempts:
                raise ValueError("Troppi tentativi falliti. Operazione annullata.")
        
        baselines = get_baseline_selection(algorithm_names)
        save_baseline_mapping(mapping_file, baselines, overwrite=True)
        return baselines
    
    if os.path.exists(mapping_file):
        try:
            with open(mapping_file, 'r') as f:
                content = f.read().strip()
                if content:
                    baselines = [name.strip() for name in content.split(',')]
            print(f"Caricato mapping baseline esistente da {mapping_file}: {baselines}")
            return baselines
        except Exception as e:
            print(f"Errore nel caricamento del mapping baseline: {e}")
            print("Creazione di un nuovo mapping...")
    
    print("Creazione mapping baseline...")
    baselines = get_baseline_selection(algorithm_names)
    save_baseline_mapping(mapping_file, baselines, overwrite=False)
    return baselines

def get_baseline_selection(algorithm_names):
    """
    Chiede all'utente di selezionare le baseline.
    """
    print("\n=== SELEZIONE BASELINE ===")
    print("Algoritmi disponibili:")
    for i, name in enumerate(algorithm_names, 1):
        print(f"{i}. {name}")
    
    while True:
        selection = input("Seleziona le baseline (es: 1,3 per algoritmi 1 e 3, o lascia vuoto per nessuna baseline): ").strip()
        
        if not selection:
            return []
        
        try:
            indices = [int(x.strip()) for x in selection.split(',')]
            baselines = []
            for idx in indices:
                if 1 <= idx <= len(algorithm_names):
                    baselines.append(algorithm_names[idx - 1])
                else:
                    print(f"Indice {idx} non valido. Deve essere tra 1 e {len(algorithm_names)}")
                    break
            else:
                confirm = input(f"Confermi la selezione delle baseline: {baselines}? [Y/n]: ").strip().lower()
                if confirm == '' or confirm == 'y':
                    return baselines
                else:
                    print("Riprova...")
        except ValueError:
            print("Formato non valido. Usa numeri separati da virgole (es: 1,3)")

def save_baseline_mapping(mapping_file, baselines, overwrite=False):
    """
    Salva il mapping delle baseline in un file.
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
            f.write(','.join(baselines))
        print(f"Mapping baseline salvato in {mapping_file}")
    else:
        print("Mapping baseline non salvato.")

def load_or_create_custom_order(csv_path, algorithm_names, force_order=False):
    """
    Carica o crea un ordinamento personalizzato.
    """
    mapping_file = os.path.join(csv_path, "custom_order_bar.txt")
    custom_order = []
    
    if force_order:
        print("Force order attivato: creazione di un nuovo ordinamento...")
        if os.path.exists(mapping_file):
            attempts = 0
            max_attempts = 3
            
            while attempts < max_attempts:
                response = input(f"Il file {mapping_file} esiste già. Vuoi sovrascriverlo? [y/n]: ").strip().lower()
                if response == 'y':
                    break
                elif response == 'n':
                    print("Operazione annullata.")
                    return algorithm_names
                else:
                    attempts += 1
                    print(f"Risposta non valida. Tentativi rimanenti: {max_attempts - attempts}")
            
            if attempts == max_attempts:
                raise ValueError("Troppi tentativi falliti. Operazione annullata.")
        
        custom_order = get_custom_order_selection(algorithm_names)
        save_custom_order(mapping_file, custom_order, overwrite=True)
        return custom_order
    
    if os.path.exists(mapping_file):
        try:
            with open(mapping_file, 'r') as f:
                content = f.read().strip()
                if content:
                    custom_order = [name.strip() for name in content.split(',')]
            print(f"Caricato ordinamento personalizzato da {mapping_file}: {custom_order}")
            
            missing = [alg for alg in algorithm_names if alg not in custom_order]
            if missing:
                print(f"Algoritmi mancanti nell'ordinamento: {missing}")
                custom_order = get_custom_order_selection(algorithm_names)
                save_custom_order(mapping_file, custom_order, overwrite=True)
            
            return custom_order
        except Exception as e:
            print(f"Errore nel caricamento dell'ordinamento: {e}")
            print("Creazione di un nuovo ordinamento...")
    
    print("Creazione ordinamento personalizzato...")
    custom_order = get_custom_order_selection(algorithm_names)
    save_custom_order(mapping_file, custom_order, overwrite=False)
    return custom_order

def get_custom_order_selection(algorithm_names):
    """
    Chiede all'utente di definire un ordinamento personalizzato.
    """
    print("\n=== ORDINAMENTO PERSONALIZZATO ===")
    print("Algoritmi disponibili:")
    for i, name in enumerate(algorithm_names, 1):
        print(f"{i}. {name}")
    
    while True:
        selection = input(f"Inserisci l'ordine desiderato (es: 4,1,3,2 per riordinare, o lascia vuoto per ordine originale): ").strip()
        
        if not selection:
            return algorithm_names
        
        try:
            indices = [int(x.strip()) for x in selection.split(',')]
            
            if len(indices) != len(algorithm_names):
                print(f"Devi specificare tutti i {len(algorithm_names)} algoritmi")
                continue
            
            if set(indices) != set(range(1, len(algorithm_names) + 1)):
                print("Devi usare tutti i numeri da 1 a {} esattamente una volta".format(len(algorithm_names)))
                continue
            
            custom_order = [algorithm_names[idx - 1] for idx in indices]
            
            confirm = input(f"Confermi l'ordinamento: {custom_order}? [Y/n]: ").strip().lower()
            if confirm == '' or confirm == 'y':
                return custom_order
            else:
                print("Riprova...")
        except ValueError:
            print("Formato non valido. Usa numeri separati da virgole")

def save_custom_order(mapping_file, custom_order, overwrite=False):
    """
    Salva l'ordinamento personalizzato in un file.
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
            f.write(','.join(custom_order))
        print(f"Ordinamento personalizzato salvato in {mapping_file}")
    else:
        print("Ordinamento personalizzato non salvato.")

def load_or_create_color_mapping(csv_path, algorithm_names, force_color=False):
    """
    Carica o crea un mapping dei colori per gli algoritmi.
    """
    mapping_file = os.path.join(csv_path, "algorithm_color_mapping.txt")
    mapping = {}
    
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
                    return {}
                else:
                    attempts += 1
                    print(f"Risposta non valida. Tentativi rimanenti: {max_attempts - attempts}")
            
            if attempts == max_attempts:
                raise ValueError("Troppi tentativi falliti. Operazione annullata.")
        
        print(f"Algoritmi per cui assegnare colori: {algorithm_names}")
        for alg_name in algorithm_names:
            mapping[alg_name] = get_algorithm_color(alg_name)
        
        save_color_mapping(mapping_file, mapping, overwrite=True)
        return mapping
    
    if os.path.exists(mapping_file):
        try:
            with open(mapping_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and ':' in line:
                        algorithm, color = line.split(':', 1)
                        mapping[algorithm.strip()] = color.strip()
            print(f"Caricato mapping colori esistente da {mapping_file}")
            
            missing_algorithms = [alg for alg in algorithm_names if alg not in mapping]
            if missing_algorithms:
                print(f"Algoritmi mancanti nel mapping colori: {missing_algorithms}")
                for alg in missing_algorithms:
                    mapping[alg] = get_algorithm_color(alg)
                save_color_mapping(mapping_file, mapping, overwrite=True)
            
            return mapping
        except Exception as e:
            print(f"Errore nel caricamento del mapping colori: {e}")
            print("Creazione di un nuovo mapping colori...")
    
    print("Creazione mapping colori algoritmi...")
    print(f"Algoritmi per cui assegnare colori: {algorithm_names}")
    
    for alg_name in algorithm_names:
        mapping[alg_name] = get_algorithm_color(alg_name)
    
    save_color_mapping(mapping_file, mapping, overwrite=False)
    return mapping

def print_color_menu():
    """
    Stampa un menu colorato dei colori disponibili per seaborn.
    """
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

def get_algorithm_color(algorithm_name):
    """
    Chiede all'utente che colore assegnare a un algoritmo e conferma la scelta.
    """
    while True:
        print_color_menu()
        color = input(f"Che colore vuoi assegnare a '{algorithm_name}'? ").strip()
        if not color:
            print("Devi inserire un colore. Riprova...")
            continue
        
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

def calculate_interquartile_mean(values):
    """
    Calculate the interquartile mean (IQM) - 25% trimmed mean.
    """
    if len(values) == 0:
        return np.nan
    
    values = np.array(values)
    values = values[~np.isnan(values)]
    
    if len(values) == 0:
        return np.nan
    
    if len(values) < 4:
        return np.mean(values)
    
    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)
    
    iqr_values = values[(values >= q1) & (values <= q3)]
    
    if len(iqr_values) == 0:
        return np.mean(values)
    
    return np.mean(iqr_values)

def detect_task_structure(csv_path):
    """
    Detect if the path contains single task or multiple tasks.
    """
    if not os.path.exists(csv_path):
        raise ValueError(f"Path {csv_path} does not exist")
    
    csv_files = [f for f in os.listdir(csv_path) if f.endswith('.csv')]
    subdirs = [d for d in os.listdir(csv_path) 
               if os.path.isdir(os.path.join(csv_path, d))]
    
    if csv_files and not subdirs:
        return True, {'single_task': csv_path}
    elif subdirs and not csv_files:
        task_data = {}
        for subdir in subdirs:
            subdir_path = os.path.join(csv_path, subdir)
            subdir_csvs = [f for f in os.listdir(subdir_path) if f.endswith('.csv')]
            if subdir_csvs:
                task_data[subdir] = subdir_path
        return False, task_data
    else:
        raise ValueError(f"Ambiguous directory structure in {csv_path}. "
                        "Directory should contain either CSV files directly (single task) "
                        "or subdirectories with CSV files (multi-task), but not both.")

def read_csvs_with_task_awareness(csv_path):
    """
    Read CSV files with task awareness for stratified bootstrap.
    """
    is_single_task, task_data = detect_task_structure(csv_path)
    
    grouped_dataframes = {}
    
    if is_single_task:
        print(f"Detected single task in: {csv_path}")
        task_name = os.path.basename(csv_path)
        
        for filename in os.listdir(csv_path):
            if filename.endswith(".csv"):
                algorithm_name = filename.split("___")[0]
                df = pd.read_csv(os.path.join(csv_path, filename))
                
                if algorithm_name not in grouped_dataframes:
                    grouped_dataframes[algorithm_name] = []
                
                grouped_dataframes[algorithm_name].append((df, task_name))
    
    else:
        print(f"Detected multi-task structure with tasks: {list(task_data.keys())}")
        
        for task_name, task_path in task_data.items():
            print(f"  Reading task: {task_name}")
            
            for filename in os.listdir(task_path):
                if filename.endswith(".csv"):
                    algorithm_name = filename.split("___")[0]
                    df = pd.read_csv(os.path.join(task_path, filename))
                    
                    if algorithm_name not in grouped_dataframes:
                        grouped_dataframes[algorithm_name] = []
                    
                    grouped_dataframes[algorithm_name].append((df, task_name))
    
    task_structure = {
        'is_single_task': is_single_task,
        'task_data': task_data
    }
    
    return grouped_dataframes, task_structure

def extract_dataframes_only(grouped_dataframes_with_tasks):
    """
    Extract only dataframes from the task-aware structure for compatibility.
    """
    grouped_dataframes = {}
    for algorithm, df_task_pairs in grouped_dataframes_with_tasks.items():
        grouped_dataframes[algorithm] = [df for df, task_name in df_task_pairs]
    return grouped_dataframes

def prepare_barplot_data_at_x_with_bootstrap(grouped_dataframes_with_tasks, x_column, y_column, fixed_x, central_line='median', n_bootstrap=1000):
    """
    Prepare data for barplot using stratified bootstrap at a specific x value.
    """
    if central_line == 'mean':
        agg_func = np.mean
    elif central_line == 'median':
        agg_func = np.median
    elif central_line == 'min':
        agg_func = np.min
    elif central_line == 'max':
        agg_func = np.max
    elif central_line == 'iqm':
        agg_func = calculate_interquartile_mean
    else:
        agg_func = np.median
    
    barplot_data = []
    
    print(f"Performing stratified bootstrap for barplot at x = {fixed_x} with {n_bootstrap} samples...")
    
    for algorithm, df_task_pairs in grouped_dataframes_with_tasks.items():
        task_values = {}
        
        for df, task_name in df_task_pairs:
            if x_column in df.columns and y_column in df.columns:
                closest_x = find_closest_x_value(df, x_column, fixed_x)
                
                if closest_x is not None:
                    matching_rows = df[df[x_column] == closest_x]
                    if not matching_rows.empty:
                        y_values = matching_rows[y_column].dropna().values
                        if len(y_values) > 0:
                            if task_name not in task_values:
                                task_values[task_name] = []
                            task_values[task_name].extend(y_values)
        
        if not task_values:
            print(f"  {algorithm}: no data found at x = {fixed_x}")
            continue
        
        bootstrap_stats = []
        original_values = []
        
        for task_vals in task_values.values():
            original_values.extend(task_vals)
        
        for _ in range(n_bootstrap):
            bootstrap_sample = []
            
            for task_name, values in task_values.items():
                if len(values) > 0:
                    resampled = np.random.choice(values, size=len(values), replace=True)
                    bootstrap_sample.extend(resampled)
            
            if bootstrap_sample:
                bootstrap_stats.append(agg_func(bootstrap_sample))
        
        if bootstrap_stats:
            bootstrap_mean = np.mean(bootstrap_stats)
            bootstrap_ci_2_5 = np.percentile(bootstrap_stats, 2.5)
            bootstrap_ci_97_5 = np.percentile(bootstrap_stats, 97.5)
            bootstrap_ci_5 = np.percentile(bootstrap_stats, 5)
            bootstrap_ci_95 = np.percentile(bootstrap_stats, 95)
            
            barplot_data.append({
                'algorithm': algorithm,
                'value': agg_func(original_values),
                'bootstrap_mean': bootstrap_mean,
                'bootstrap_ci_2_5': bootstrap_ci_2_5,
                'bootstrap_ci_97_5': bootstrap_ci_97_5,
                'bootstrap_ci_5': bootstrap_ci_5,
                'bootstrap_ci_95': bootstrap_ci_95,
                'n_tasks': len(task_values),
                'n_runs': len(original_values)
            })
            
            print(f"  {algorithm}: {len(task_values)} tasks, {len(original_values)} runs, "
                  f"bootstrap CI 95%: [{bootstrap_ci_2_5:.2f}, {bootstrap_ci_97_5:.2f}]")
    
    return pd.DataFrame(barplot_data)

def create_barplot(data, x_column='algorithm', y_column='value', 
                  custom_order=None, baselines=None, log_y=False,
                  y_min=None, y_max=None, y_scale_factor=None,
                  y_label=None, plot_title=None, color_mapping=None, 
                  central_line='median', bar_type='std_err', error_bar_percentile=95,
                  show_baseline_error_bars=False, show_tendency_line=False, 
                  use_bootstrap=False, **kwargs):
    """
    Crea un barplot con opzioni personalizzate.
    
    Parameters:
    - central_line: Tipo di valore centrale per l'altezza della barra ('mean', 'median', 'min', 'max', 'iqm')
    - bar_type: Tipo di barre di errore ('std_err', 'std', 'confidence_interval', 'quartile', 'min_max')
    - error_bar_percentile: Percentile per confidence_interval (es. 95 per 95% CI). Ignorato per altri bar_type
    - show_baseline_error_bars: Se True, mostra barre di errore per le baseline
    - show_tendency_line: Se True, mostra una linea che connette le altezze delle barre
    - use_bootstrap: Se True, usa statistiche bootstrap per le barre di errore
    """
    sns.set_style("darkgrid")
    plt.figure(figsize=(12, 8))
    
    # Rimuovi i dati delle baseline dal dataset principale per il barplot
    plot_data = data.copy()
    if baselines:
        plot_data = plot_data[~plot_data[x_column].isin(baselines)]
    
    # Applica ordinamento personalizzato se specificato (escludendo le baseline)
    if custom_order:
        bar_order = [alg for alg in custom_order if not baselines or alg not in baselines]
        if bar_order:
            plot_data[x_column] = pd.Categorical(plot_data[x_column], categories=bar_order, ordered=True)
            plot_data = plot_data.sort_values(x_column)
    
    # Se non ci sono dati per il barplot (solo baseline), salta la creazione del barplot
    if len(plot_data) > 0:
        algorithms = plot_data[x_column].unique()
        
        x_positions = []
        bar_heights = []
        error_lower = []
        error_upper = []
        colors = []
        labels = []
        
        for i, algorithm in enumerate(algorithms):
            alg_data_rows = plot_data[plot_data[x_column] == algorithm]
            
            if len(alg_data_rows) == 0:
                continue
            
            labels.append(algorithm)
            x_positions.append(i)
            
            # Get the bar height and error bars
            if use_bootstrap and 'bootstrap_mean' in alg_data_rows.columns:
                # Use bootstrap statistics for bar height
                if central_line == 'mean':
                    bar_height = alg_data_rows['bootstrap_mean'].iloc[0]
                else:
                    # For other central lines, use the original calculated value
                    bar_height = alg_data_rows['value'].iloc[0]
                
                # Calculate error bars based on bar_type
                if bar_type == 'confidence_interval':
                    if error_bar_percentile == 95:
                        error_lower.append(alg_data_rows['bootstrap_ci_2_5'].iloc[0])
                        error_upper.append(alg_data_rows['bootstrap_ci_97_5'].iloc[0])
                    elif error_bar_percentile == 90:
                        error_lower.append(alg_data_rows['bootstrap_ci_5'].iloc[0])
                        error_upper.append(alg_data_rows['bootstrap_ci_95'].iloc[0])
                    else:
                        # Fallback to regular percentiles for other confidence levels
                        # Get original data for this algorithm
                        alg_values = []
                        for _, row in alg_data_rows.iterrows():
                            # This is a simplified fallback - in practice you'd need access to original data
                            alg_values.append(row[y_column])
                        lower_percentile = (100 - error_bar_percentile) / 2
                        upper_percentile = 100 - lower_percentile
                        error_lower.append(np.percentile(alg_values, lower_percentile))
                        error_upper.append(np.percentile(alg_values, upper_percentile))
                else:
                    # For other bar_types with bootstrap, we need original data
                    # This is a limitation - we should store more statistics in bootstrap
                    # For now, fallback to confidence intervals
                    error_lower.append(alg_data_rows['bootstrap_ci_5'].iloc[0])
                    error_upper.append(alg_data_rows['bootstrap_ci_95'].iloc[0])
            else:
                # Use original non-bootstrap logic
                alg_data = alg_data_rows[y_column].dropna()
                
                # Calcola altezza della barra (valore centrale)
                if central_line == 'mean':
                    bar_height = np.mean(alg_data)
                elif central_line == 'median':
                    bar_height = np.median(alg_data)
                elif central_line == 'min':
                    bar_height = np.min(alg_data)
                elif central_line == 'max':
                    bar_height = np.max(alg_data)
                elif central_line == 'iqm':
                    bar_height = calculate_interquartile_mean(alg_data)
                else:
                    bar_height = np.median(alg_data)  # default
                
                # Calcola barre di errore in base al bar_type
                if bar_type == 'std_err':
                    # Standard error
                    std_err = np.std(alg_data) / np.sqrt(len(alg_data))
                    error_lower.append(bar_height - std_err)
                    error_upper.append(bar_height + std_err)
                elif bar_type == 'std':
                    # Standard deviation
                    std_val = np.std(alg_data)
                    error_lower.append(bar_height - std_val)
                    error_upper.append(bar_height + std_val)
                elif bar_type == 'confidence_interval':
                    # Confidence interval basato su error_bar_percentile
                    lower_percentile = (100 - error_bar_percentile) / 2
                    upper_percentile = 100 - lower_percentile
                    error_lower.append(np.percentile(alg_data, lower_percentile))
                    error_upper.append(np.percentile(alg_data, upper_percentile))
                elif bar_type == 'quartile':
                    # Quartili (25% - 75%)
                    q25 = np.percentile(alg_data, 25)
                    q75 = np.percentile(alg_data, 75)
                    error_lower.append(q25)
                    error_upper.append(q75)
                elif bar_type == 'min_max':
                    # Min-Max range
                    error_lower.append(np.min(alg_data))
                    error_upper.append(np.max(alg_data))
                else:
                    # Default: standard error
                    std_err = np.std(alg_data) / np.sqrt(len(alg_data))
                    error_lower.append(bar_height - std_err)
                    error_upper.append(bar_height + std_err)
            
            bar_heights.append(bar_height)
            
            # Determina il colore
            if color_mapping and algorithm in color_mapping:
                colors.append(color_mapping[algorithm])
            else:
                colors.append(sns.color_palette("tab10")[i % 10])
        
        # Crea il barplot
        ax = plt.gca()
        
        # Disegna le barre
        bars = plt.bar(x_positions, bar_heights, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
        
        # Disegna le barre di errore
        for x, height, e_lower, e_upper, color in zip(x_positions, bar_heights, error_lower, error_upper, colors):
            if e_lower is not None and e_upper is not None:
                # Calcola l'errore rispetto alla barra
                yerr_lower = height - e_lower
                yerr_upper = e_upper - height
                
                plt.errorbar(x, height, yerr=[[yerr_lower], [yerr_upper]], 
                           color='black', capsize=5, capthick=2, linewidth=2)
        
        # Disegna la linea di tendenza se richiesta
        if show_tendency_line and len(x_positions) > 1:
            plt.plot(x_positions, bar_heights, color='gray', linewidth=2, alpha=0.6, 
                    linestyle='-', marker='o', markersize=4)
        
        # Imposta le etichette dell'asse x
        plt.xticks(x_positions, labels, rotation=45, ha='right')
    
    # Aggiungi linee baseline se specificate
    if baselines:
        ax = plt.gca()
        
        for baseline_name in baselines:
            if baseline_name in data[x_column].values:
                baseline_data = data[data[x_column] == baseline_name][y_column]
                
                # Calcola il valore centrale della baseline
                if central_line == 'mean':
                    baseline_value = baseline_data.mean()
                elif central_line == 'median':
                    baseline_value = baseline_data.median()
                elif central_line == 'min':
                    baseline_value = baseline_data.min()
                elif central_line == 'max':
                    baseline_value = baseline_data.max()
                else:
                    baseline_value = baseline_data.median()
                
                # Disegna la linea orizzontale della baseline
                plt.axhline(y=baseline_value, color='red', linestyle='--', alpha=0.8, linewidth=2,
                           label=f'Baseline: {baseline_name}')
                
                # Aggiungi barre di errore per la baseline se richiesto
                if show_baseline_error_bars and len(baseline_data) > 1:
                    if error_bar_percentile > 0:
                        lower_percentile = (100 - error_bar_percentile) / 2
                        upper_percentile = 100 - lower_percentile
                        baseline_lower = np.percentile(baseline_data, lower_percentile)
                        baseline_upper = np.percentile(baseline_data, upper_percentile)
                        
                        # Aggiungi zona ombreggiata per l'intervallo di confidenza della baseline
                        plt.axhspan(baseline_lower, baseline_upper, alpha=0.2, color='red',
                                   label=f'{baseline_name} {error_bar_percentile}% CI')
                        
                        print(f"Baseline {baseline_name}: valore={baseline_value:.2f}, "
                              f"intervallo=[{baseline_lower:.2f}, {baseline_upper:.2f}] "
                              f"({error_bar_percentile}% CI)")
    
    # Imposta scala logaritmica se richiesta
    ax = plt.gca()
    if log_y:
        ax.set_yscale('log')
        y_axis_label = y_label or 'Value (logscale)'
        if y_label and y_scale_factor is not None:
            y_scale_notation = f"1e{int(np.log10(y_scale_factor))}"
            y_axis_label = f"{y_label} (logscale, {y_scale_notation})"
    else:
        # Formatta l'asse y se specificato un fattore di scala
        if y_scale_factor is not None:
            def format_func(value, ticker):
                if np.isnan(value) or value < 0:
                    return value
                elif y_scale_factor is None:
                    return value
                else:
                    scaled_value = value / y_scale_factor
                    if scaled_value == int(scaled_value):
                        return "{:d}".format(int(scaled_value))
                    else:
                        return "{:.1f}".format(scaled_value)
            
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(format_func))
            y_scale_notation = f"1e{int(np.log10(y_scale_factor))}"
            
            if y_label:
                y_axis_label = f"{y_label} ({y_scale_notation})"
            else:
                y_axis_label = f"Value ({y_scale_notation})"
        else:
            y_axis_label = y_label or 'Value'
    
    plt.ylabel(y_axis_label)
    plt.xlabel('Algorithm')
    
    # Imposta titolo personalizzato se specificato
    if plot_title is not None:
        plt.title(plot_title)
    
    # Imposta limiti dell'asse y se specificati
    if y_min is not None:
        ax.set_ylim(bottom=y_min)
    if y_max is not None:
        ax.set_ylim(top=y_max)
    
    # Aggiungi legenda se ci sono baseline
    if baselines:
        plt.legend()
    
    plt.tight_layout()

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--xaxis', '-x', default='buffer_size', help='Column name for x-axis data')
    parser.add_argument('--value', '-y', default='eval/episode_reward', help='Column name for y-axis data')
    parser.add_argument('--fixed_x', type=float, required=True, 
                        help='Fixed x value for barplot comparison (e.g., 100000)')
    
    # Barplot configuration
    parser.add_argument('--central_line', default='median', choices=['mean', 'median', 'min', 'max', 'iqm'],
                        help='Type of central value for bar height')
    parser.add_argument('--bar_type', default='std_err', choices=['std_err', 'std', 'confidence_interval', 'ci', 'quartile', 'min_max'],
                        help='Type of error bars in barplot (ci is short for confidence_interval)')
    parser.add_argument('--error_bar_percentile', type=float, default=95.0,
                        help='Percentile for confidence_interval type (e.g., 95 for 95% CI). Ignorato per altri bar_types')
    
    # Bootstrap options
    parser.add_argument('--bootstrap', action='store_true', default=False,
                        help='Use stratified bootstrap for confidence intervals')
    parser.add_argument('--n_bootstrap', type=int, default=10000,
                        help='Number of bootstrap samples (default: 10000)')
    
    # Baseline error bars option
    parser.add_argument('--baseline_error_bars', action='store_true', default=False,
                        help='Show error bars/bands for baseline confidence intervals')
    
    # Tendency line option
    parser.add_argument('--tendency_line', action='store_true', default=False,
                        help='Show a line connecting the bar heights')
    
    # Axis scaling options
    parser.add_argument('--log_y', action='store_true', default=False, help='Use log scale for y-axis')
    
    # Axis limits
    parser.add_argument('--y_min', type=float, default=None, help='Minimum value for y-axis')
    parser.add_argument('--y_max', type=float, default=None, help='Maximum value for y-axis')
    
    # Axis formatting
    parser.add_argument('--y_scale', type=float, default=None, help='Scale factor for y-axis values (e.g., 1e3)')
    parser.add_argument('--y_label', type=str, default=None, help='Custom y-axis label')
    
    # Plot title
    parser.add_argument('--title', type=str, default=None, help='Plot title')
    
    parser.add_argument('--csv_path', type=str, default="data_plot/", help='csv folder')
    parser.add_argument('--output', type=str, default="barplot.png", help='Output filename')
    
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
    
    # Baseline options
    parser.add_argument('--baseline', action='store_true', default=False,
                        help='Enable baseline horizontal lines')
    parser.add_argument('--force-baseline', action='store_true', default=False,
                        help='Force baseline selection even if mapping file exists')
    
    # Custom order options
    parser.add_argument('--custom_order', action='store_true', default=False,
                        help='Enable custom ordering of algorithms')
    parser.add_argument('--force-order', action='store_true', default=False,
                        help='Force custom order selection even if mapping file exists')
    
    args = parser.parse_args()

    # Map 'ci' to 'confidence_interval' for backward compatibility
    if args.bar_type == 'ci':
        args.bar_type = 'confidence_interval'

    # Always use task-aware reading to handle both single and multi-task structures
    grouped_dataframes_with_tasks, task_structure = read_csvs_with_task_awareness(args.csv_path)
    print(f"Task structure: {task_structure}")
    
    # Extract dataframes for compatibility with existing functions
    grouped_dataframes = extract_dataframes_only(grouped_dataframes_with_tasks)
    
    # Apply renaming if requested
    if args.rename:
        algorithm_names = list(grouped_dataframes.keys())
        name_mapping = load_or_create_barplot_name_mapping(
            args.csv_path, 
            algorithm_names, 
            force_rename=getattr(args, 'force_rename', False)
        )
        
        if args.bootstrap:
            # Apply renaming to task-aware structure
            renamed_dataframes_with_tasks = {}
            for original_name, df_task_pairs in grouped_dataframes_with_tasks.items():
                new_name = name_mapping.get(original_name, original_name)
                if new_name in renamed_dataframes_with_tasks:
                    renamed_dataframes_with_tasks[new_name].extend(df_task_pairs)
                else:
                    renamed_dataframes_with_tasks[new_name] = df_task_pairs.copy()
            grouped_dataframes_with_tasks = renamed_dataframes_with_tasks
        
        grouped_dataframes = apply_name_mapping(grouped_dataframes, name_mapping)
    
    # Prepare data for barplot
    if args.bootstrap:
        # Use stratified bootstrap method
        barplot_data = prepare_barplot_data_at_x_with_bootstrap(
            grouped_dataframes_with_tasks, 
            args.xaxis, 
            args.value, 
            args.fixed_x,
            central_line=args.central_line,
            n_bootstrap=args.n_bootstrap
        )
        use_bootstrap_in_plot = True
    else:
        # Use traditional method but with task-aware data
        barplot_data = prepare_barplot_data_at_x(grouped_dataframes, args.xaxis, args.value, args.fixed_x)
        use_bootstrap_in_plot = False
    
    if barplot_data.empty:
        print(f"ERRORE: Nessun dato trovato per x = {args.fixed_x}")
        print("Verifica che il valore di --fixed_x sia presente nei tuoi dati.")
        print(f"Struttura task rilevata: {task_structure}")
        print(f"Algoritmi trovati: {list(grouped_dataframes.keys())}")
        return
    
    # Get final algorithm names (after renaming)
    final_algorithm_names = list(set(barplot_data['algorithm'].values))
    print(f"Algoritmi nel dataset finale: {final_algorithm_names}")
    
    # Apply color mapping if requested
    color_mapping = None
    if args.color:
        color_mapping = load_or_create_color_mapping(
            args.csv_path,
            final_algorithm_names,
            force_color=getattr(args, 'force_color', False)
        )
    
    # Handle baseline selection if requested
    baselines = None
    if args.baseline:
        baselines = load_or_create_baseline_mapping(
            args.csv_path,
            final_algorithm_names,
            force_baseline=getattr(args, 'force_baseline', False)
        )
    
    # Handle custom ordering if requested
    custom_order = None
    if args.custom_order:
        custom_order = load_or_create_custom_order(
            args.csv_path,
            final_algorithm_names,
            force_order=getattr(args, 'force_order', False)
        )
    
    # Create the barplot
    print(f"Creating barplot for x = {args.fixed_x}...")
    create_barplot(
        barplot_data,
        x_column='algorithm',
        y_column='value',
        custom_order=custom_order,
        baselines=baselines,
        log_y=args.log_y,
        y_min=args.y_min,
        y_max=args.y_max,
        y_scale_factor=args.y_scale,
        y_label=args.y_label,
        plot_title=args.title or f"Comparison at {args.xaxis} = {args.fixed_x}",
        color_mapping=color_mapping,
        central_line=args.central_line,
        bar_type=args.bar_type,
        error_bar_percentile=args.error_bar_percentile,
        show_baseline_error_bars=args.baseline_error_bars,
        show_tendency_line=args.tendency_line,
        use_bootstrap=use_bootstrap_in_plot
    )
    
    # Create output path in the same folder as csv_path
    output_path = os.path.join(args.csv_path, args.output)
    
    # Save the plot
    plt.savefig(output_path, format='png', dpi=300)
    print(f"Barplot saved as {output_path}")


if __name__ == "__main__":
    main()