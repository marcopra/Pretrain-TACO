"""
python plot/make_boxplot.py --csv_path data_plot/push/exp --rename --color --baseline --custom_order --box_type std_err --whiskers_percentile 90 --fixed_x 100000 --y_min 0 --baseline_whiskers -y eval/success_rate
python plot/make_boxplot.py --csv_path data_plot/push/exp --rename --color --baseline --custom_order --box_type std_err --whiskers_percentile 90 --fixed_x 100000 --y_min 0 -y eval/success_rate
python plot/make_boxplot.py --csv_path data_plot/push/exp --rename --color --baseline --custom_order --box_type std_err --whiskers_percentile 90 --fixed_x 100000 --y_min 0 --baseline_whiskers -y eval/success_rate
python plot/make_boxplot.py --csv_path data_plot/push/exp --rename --color --baseline --custom_order --box_type std_err --whiskers_percentile 90 --fixed_x 100000 --y_min 0 -y eval/success_rate
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
    
    Parameters:
    - df: DataFrame contenente i dati
    - x_column: Nome della colonna x
    - target_x: Valore target da cercare
    
    Returns:
    - Il valore più vicino al target_x presente nel dataframe
    """
    if x_column not in df.columns:
        return None
    
    x_values = df[x_column].dropna().unique()
    if len(x_values) == 0:
        return None
    
    # Trova il valore più vicino
    closest_value = min(x_values, key=lambda x: abs(x - target_x))
    return closest_value

def prepare_boxplot_data(grouped_dataframes, y_column):
    """
    Prepare data for boxplot by collecting all y values for each algorithm.
    
    Parameters:
    - grouped_dataframes: Dictionary with algorithm names as keys and list of dataframes as values
    - y_column: Column name to extract values from
    
    Returns:
    - List of dictionaries with 'algorithm' and 'value' columns
    """
    boxplot_data = []
    
    for algorithm, dataframes in grouped_dataframes.items():
        for df in dataframes:
            if y_column in df.columns:
                for value in df[y_column].dropna():
                    boxplot_data.append({
                        'algorithm': algorithm,
                        'value': value
                    })
    
    return pd.DataFrame(boxplot_data)

def prepare_boxplot_data_at_x(grouped_dataframes, x_column, y_column, fixed_x):
    """
    Prepare data for boxplot by collecting y values at a specific x value for each algorithm.
    
    Parameters:
    - grouped_dataframes: Dictionary with algorithm names as keys and list of dataframes as values
    - x_column: Column name for x-axis data
    - y_column: Column name to extract values from
    - fixed_x: Specific x value to extract data for
    
    Returns:
    - DataFrame with 'algorithm' and 'value' columns for the specified x value
    """
    boxplot_data = []
    
    print(f"Cercando dati per x = {fixed_x}...")
    
    for algorithm, dataframes in grouped_dataframes.items():
        algorithm_values = []
        
        for df in dataframes:
            if x_column in df.columns and y_column in df.columns:
                # Trova il valore x più vicino al target
                closest_x = find_closest_x_value(df, x_column, fixed_x)
                
                if closest_x is not None:
                    # Estrai i valori y corrispondenti al closest_x
                    matching_rows = df[df[x_column] == closest_x]
                    if not matching_rows.empty:
                        y_values = matching_rows[y_column].dropna()
                        algorithm_values.extend(y_values.values)
                        
                        if closest_x != fixed_x:
                            print(f"  {algorithm}: usando x = {closest_x} (più vicino a {fixed_x})")
        
        # Aggiungi tutti i valori trovati per questo algoritmo
        for value in algorithm_values:
            boxplot_data.append({
                'algorithm': algorithm,
                'value': value
            })
        
        if algorithm_values:
            print(f"  {algorithm}: trovati {len(algorithm_values)} valori")
        else:
            print(f"  {algorithm}: nessun valore trovato per x = {fixed_x}")
    
    return pd.DataFrame(boxplot_data)

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

def load_or_create_boxplot_name_mapping(csv_path, algorithm_names, force_rename=False):
    """
    Carica o crea un mapping dei nomi degli algoritmi per boxplot.
    """
    mapping_file = os.path.join(csv_path, "algorithm_name_mapping_box.txt")
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
    
    print("Creazione mapping nomi algoritmi per boxplot...")
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

def load_or_create_baseline_mapping(csv_path, algorithm_names, force_baseline=False):
    """
    Carica o crea un mapping per le baseline.
    """
    mapping_file = os.path.join(csv_path, "baseline_mapping_box.txt")
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
    mapping_file = os.path.join(csv_path, "custom_order_box.txt")
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
            
            # Verifica che tutti gli algoritmi siano nell'ordinamento
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
    Carica o crea un mapping dei colori per gli algoritmi (usa lo stesso file di make_plot.py).
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

def create_boxplot(data, x_column='algorithm', y_column='value', 
                  custom_order=None, baselines=None, log_y=False,
                  y_min=None, y_max=None, y_scale_factor=None,
                  y_label=None, plot_title=None, color_mapping=None, 
                  central_line='median', box_type='std_err', whiskers_percentile=95,
                  show_baseline_whiskers=False, **kwargs):
    """
    Crea un boxplot con opzioni personalizzate.
    
    Parameters:
    - central_line: Tipo di linea centrale ('mean', 'median', 'min', 'max')
    - box_type: Tipo di box ('std_err', 'std', 'min_max')
    - whiskers_percentile: Percentile per i whiskers (es. 95 per 95% confidence)
    - show_baseline_whiskers: Se True, mostra whiskers verticali per le baseline
    """
    sns.set_style("darkgrid")
    plt.figure(figsize=(12, 8))
    
    # Rimuovi i dati delle baseline dal dataset principale per il boxplot
    plot_data = data.copy()
    if baselines:
        # Filtra i dati escludendo le baseline
        plot_data = plot_data[~plot_data[x_column].isin(baselines)]
    
    # Applica ordinamento personalizzato se specificato (escludendo le baseline)
    if custom_order:
        # Rimuovi le baseline dall'ordinamento per il boxplot
        box_order = [alg for alg in custom_order if not baselines or alg not in baselines]
        if box_order:  # Solo se ci sono algoritmi non-baseline
            plot_data[x_column] = pd.Categorical(plot_data[x_column], categories=box_order, ordered=True)
            plot_data = plot_data.sort_values(x_column)
    
    # Se non ci sono dati per il boxplot (solo baseline), salta la creazione del boxplot
    if len(plot_data) > 0:
        # Aggrega i dati per ogni algoritmo per calcolare statistiche personalizzate
        algorithms = plot_data[x_column].unique()
        
        # Prepara i dati per il boxplot personalizzato
        x_positions = []
        central_values = []
        box_lower = []
        box_upper = []
        whisker_lower = []
        whisker_upper = []
        colors = []
        labels = []
        
        for i, algorithm in enumerate(algorithms):
            alg_data = plot_data[plot_data[x_column] == algorithm][y_column].dropna()
            
            if len(alg_data) == 0:
                continue
            
            labels.append(algorithm)
            x_positions.append(i)
            
            # Calcola linea centrale
            if central_line == 'mean':
                central = np.mean(alg_data)
            elif central_line == 'median':
                central = np.median(alg_data)
            elif central_line == 'min':
                central = np.min(alg_data)
            elif central_line == 'max':
                central = np.max(alg_data)
            else:
                central = np.median(alg_data)  # default
            
            central_values.append(central)
            
            # Calcola box (area ombreggiata)
            if box_type == 'std_err':
                std_err = np.std(alg_data) / np.sqrt(len(alg_data))
                box_lower.append(central - std_err)
                box_upper.append(central + std_err)
            elif box_type == 'std':
                std_val = np.std(alg_data)
                box_lower.append(central - std_val)
                box_upper.append(central + std_val)
            elif box_type == 'min_max':
                box_lower.append(np.min(alg_data))
                box_upper.append(np.max(alg_data))
            else:
                # Default: quartili
                q25 = np.percentile(alg_data, 25)
                q75 = np.percentile(alg_data, 75)
                box_lower.append(q25)
                box_upper.append(q75)
            
            # Calcola whiskers (confidence intervals)
            lower_percentile = (100 - whiskers_percentile) / 2
            upper_percentile = 100 - lower_percentile
            whisker_lower.append(np.percentile(alg_data, lower_percentile))
            whisker_upper.append(np.percentile(alg_data, upper_percentile))
            
            # Determina il colore
            if color_mapping and algorithm in color_mapping:
                colors.append(color_mapping[algorithm])
            else:
                colors.append(sns.color_palette("tab10")[i % 10])
        
        # Crea il boxplot personalizzato
        ax = plt.gca()
        
        # Disegna i box (aree ombreggiate)
        for i, (x, lower, upper, color) in enumerate(zip(x_positions, box_lower, box_upper, colors)):
            width = 0.6
            rect = plt.Rectangle((x - width/2, lower), width, upper - lower, 
                               facecolor=color, alpha=0.3, edgecolor=color, linewidth=1)
            ax.add_patch(rect)
        
        # Disegna le linee centrali
        for x, central, color in zip(x_positions, central_values, colors):
            plt.plot([x - 0.3, x + 0.3], [central, central], color=color, linewidth=3)
        
        # Disegna i whiskers
        for x, w_lower, w_upper, central, color in zip(x_positions, whisker_lower, whisker_upper, central_values, colors):
            # Linee verticali dei whiskers
            plt.plot([x, x], [w_lower, w_upper], color=color, linewidth=1, alpha=0.7)
            # Cappucci dei whiskers
            plt.plot([x - 0.1, x + 0.1], [w_lower, w_lower], color=color, linewidth=1, alpha=0.7)
            plt.plot([x - 0.1, x + 0.1], [w_upper, w_upper], color=color, linewidth=1, alpha=0.7)
        
        # Imposta le etichette dell'asse x
        plt.xticks(x_positions, labels, rotation=45, ha='right')
    
    # Aggiungi linee baseline se specificate
    if baselines:
        ax = plt.gca()
        # Calcola la posizione x centrale del grafico
        if len(plot_data) > 0:
            # Se ci sono box, metti i whiskers baseline al centro
            center_x = (len(x_positions) - 1) / 2 if x_positions else 0
        else:
            # Se non ci sono box, metti i whiskers al centro dell'asse
            center_x = 0
        
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
                    baseline_value = baseline_data.median()  # default
                
                # Disegna la linea orizzontale della baseline
                plt.axhline(y=baseline_value, color='black', linestyle='--', alpha=0.7, 
                           label=f'Baseline: {baseline_name}')
                
                # Aggiungi whiskers verticali se richiesto
                if show_baseline_whiskers and len(baseline_data) > 1:
                    # Calcola i whiskers della baseline
                    lower_percentile = (100 - whiskers_percentile) / 2
                    upper_percentile = 100 - lower_percentile
                    baseline_lower = np.percentile(baseline_data, lower_percentile)
                    baseline_upper = np.percentile(baseline_data, upper_percentile)
                    
                    # Disegna whiskers verticali al centro del grafico
                    whisker_width = 0.2  # Larghezza dei whiskers
                    
                    # Linea verticale principale del whisker
                    plt.plot([center_x, center_x], [baseline_lower, baseline_upper], 
                            color='black', linewidth=2, alpha=0.8)
                    # Cappucci orizzontali dei whiskers
                    plt.plot([center_x - whisker_width/2, center_x + whisker_width/2], 
                            [baseline_lower, baseline_lower], 
                            color='black', linewidth=2, alpha=0.8)
                    plt.plot([center_x - whisker_width/2, center_x + whisker_width/2], 
                            [baseline_upper, baseline_upper], 
                            color='black', linewidth=2, alpha=0.8)
                    
                    print(f"Baseline {baseline_name}: centro={baseline_value:.2f}, "
                          f"whiskers=[{baseline_lower:.2f}, {baseline_upper:.2f}] "
                          f"({whiskers_percentile}% CI)")
    
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
                        help='Fixed x value for boxplot comparison (e.g., 100000)')
    
    # Boxplot configuration
    parser.add_argument('--central_line', default='median', choices=['mean', 'median', 'min', 'max'],
                        help='Type of central line in boxplot')
    parser.add_argument('--box_type', default='std_err', choices=['std_err', 'std', 'min_max', 'quartile'],
                        help='Type of box (shaded area) in boxplot')
    parser.add_argument('--whiskers_percentile', type=float, default=95.0,
                        help='Percentile for whiskers (e.g., 95 for 95% confidence interval)')
    
    # Baseline whiskers option
    parser.add_argument('--baseline_whiskers', action='store_true', default=False,
                        help='Show vertical whiskers for baseline confidence intervals at center of plot')
    
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
    parser.add_argument('--output', type=str, default="boxplot.png", help='Output filename')
    
    # Rename options
    parser.add_argument('--rename', action='store_true', default=False, 
                        help='Enable algorithm name renaming with interactive input')
    parser.add_argument('--force-rename', action='store_true', default=False,
                        help='Force renaming even if mapping file exists')
    
    # Color mapping options (usa lo stesso file di make_plot.py)
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

    # Read and group CSV files by algorithm
    grouped_dataframes = read_csvs_from_directory(args.csv_path)
    
    # Apply renaming if requested
    if args.rename:
        algorithm_names = list(grouped_dataframes.keys())
        name_mapping = load_or_create_boxplot_name_mapping(
            args.csv_path, 
            algorithm_names, 
            force_rename=getattr(args, 'force_rename', False)
        )
        grouped_dataframes = apply_name_mapping(grouped_dataframes, name_mapping)
    
    # Get final algorithm names (after renaming)
    final_algorithm_names = list(grouped_dataframes.keys())
    
    # Apply color mapping if requested (usa lo stesso file di make_plot.py)
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
    
    # Prepare data for boxplot at fixed x value
    boxplot_data = prepare_boxplot_data_at_x(grouped_dataframes, args.xaxis, args.value, args.fixed_x)
    
    if boxplot_data.empty:
        print(f"ERRORE: Nessun dato trovato per x = {args.fixed_x}")
        print("Verifica che il valore di --fixed_x sia presente nei tuoi dati.")
        return
    
    # Create the boxplot
    print(f"Creating boxplot for x = {args.fixed_x}...")
    create_boxplot(
        boxplot_data,
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
        box_type=args.box_type,
        whiskers_percentile=args.whiskers_percentile,
        show_baseline_whiskers=args.baseline_whiskers
    )
    
    # Create output path in the same folder as csv_path
    output_path = os.path.join(args.csv_path, args.output)
    
    # Save the plot
    plt.savefig(output_path, format='png', dpi=300)
    print(f"Boxplot saved as {output_path}")

if __name__ == "__main__":
    main()
