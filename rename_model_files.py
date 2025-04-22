import os
import glob

def rename_model_files():
    # Percorso della cartella models
    models_dir = "models"
    
    # Trova tutti i file nella cartella models
    model_files = glob.glob(os.path.join(models_dir, "*"))
    
    # Filtra solo i file che contengono il pattern _1OOD_config.json_
    files_to_rename = [f for f in model_files if "_1OOD_config.json_" in f]
    
    # Stampa i file che verranno rinominati
    print(f"Trovati {len(files_to_rename)} file da rinominare:")
    for file in files_to_rename:
        print(f"  - {os.path.basename(file)}")
    
    # Rinomina i file
    for old_path in files_to_rename:
        filename = os.path.basename(old_path)
        new_filename = filename.replace("_1OOD_config.json_", "_1OOD_push_config_")
        new_path = os.path.join(models_dir, new_filename)
        
        os.rename(old_path, new_path)
        print(f"Rinominato: {filename} -> {new_filename}")

if __name__ == "__main__":
    rename_model_files()
    print("Rinominazione completata.")
