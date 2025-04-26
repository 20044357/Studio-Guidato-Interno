import datetime
import json
import os
import time
import requests
import gzip
import csv
import pandas as pd

# Percorsi di directory per i dati e i file CSV generati
DATA_JSON_PATH = 'data/github_data'
DATA_CSV_PATH = 'data/github_csv'
DATA_CLEANED_CSV_PATH = 'data/github_cleaned_csv'

# Intestazioni per i file CSV generati
CSV_HEADERS = [
  "type_event", "id_actor", "login_actor", "url_actor",
  "url_repo", "id_repo", "name_repo", "action_payload", "created_at"
]

def clean_unused_files(directory, processed_files, file_extension):
    """
    Rimuove i file datati da una directory.

    Confronta i file presenti in 'input_dir' con quelli nell'elenco 'processed_files'
    e rimuove quelli NON presenti nell'elenco 'processed_files'.

    Parametri:
    - input_dir (str): Percorso della directory da pulire.
    - processed_files (list): Elenco dei file da mantenere.
    - file_extension (str): Estensione dei file da considerare.

    Ritorna:
    - None
    """
    for filename in os.listdir(directory):
        if filename.endswith(file_extension) and filename not in processed_files:
            file_path = os.path.join(directory, filename)
            try:
                os.remove(file_path)
                print(f"[INFO] Rimosso file obsoleto: {file_path}")
            except Exception as e:
                print(f"[ERROR] Impossibile rimuovere il file {file_path}: {e}")

def download_files():
    """
    Scarica i file .json.gz da GHArchive relativi all'ultimo intervallo 11:00 UTC -> 10:00 UTC disponibile.

    Confronta i file presenti in 'input_dir' con quelli nell'elenco 'processed_files'
    e rimuove quelli NON presenti nell'elenco 'processed_files'.

    Il Download:
    - Verifica se i file sono già presente nella directory, evitando di riscaricarli
    - Effettua il download dei file mancanti da https://data.gharchive.org/
    - Tiene traccia del tempo impiegato per il download (provvisorio)
    - Rimuove eventuali file datati dalla directory di destinaione (cleaned_unused_files(...))

    Ritorna:
    - None
    """
    os.makedirs(DATA_JSON_PATH, exist_ok=True)

    now = datetime.datetime.now(datetime.timezone.utc)
    reference_time = now.replace(hour=11, minute=3, second=0, microsecond=0)

    if now < reference_time:
        # Prima delle 11:03 UTC -> intervallo: 11:00 di 2 giorni fa -> alle 11:00 di ieri
        start = (now - datetime.timedelta(days=2)).replace(hour=11, minute=0, second=0, microsecond=0)
        end = (now - datetime.timedelta(days=1)).replace(hour=11, minute=0, second=0, microsecond=0)
    else:
        # Dopo le 11:03 UTC -> intervallo: 11:00 di ieri -> 10:00 di oggi 
        start = (now - datetime.timedelta(days=1)).replace(hour=11, minute=0, second=0, microsecond=0)
        end = now.replace(hour=11, minute=0, second=0, microsecond=0)

    print(f"[INFO] Download files del {start.date()} dalle {start.hour} alle {end.hour} di {end.date()}:")

    total_download_time = 0
    downloaded_files = set()
    current = start

    while current < end:
        filename = f'{current.strftime("%Y-%m-%d")}-{current.hour}.json.gz'
        filepath = os.path.join(DATA_JSON_PATH, filename)
        url = f'https://data.gharchive.org/{filename}'

        if os.path.exists(filepath):
            print(f"[INFO] File {filename} gia' presente, salto.")
            downloaded_files.add(filename)
        else:
            try:
                start_time = time.time()
                response = requests.get(url, stream=True)
                response.raise_for_status()

                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=4096):
                        f.write(chunk)

                elapsed = time.time() - start_time
                total_download_time += elapsed
                downloaded_files.add(filename)
                print(f"[INFO] {filename} scaricato in {elapsed:.2f}s.")
            except Exception as e:
                print(f"[ERROR] scaricando il file {filename}: {e}")
        
        current += datetime.timedelta(hours=1)

    clean_unused_files(DATA_JSON_PATH, downloaded_files, ".json.gz")
    print(f"[INFO] Download completato. Tempo totale: {total_download_time:.2f}s.")

def process_files():
    """
    Elabora i file .json.gz e li converte in file CSV.

    Elaborazione per ogni file:
    - Se il CSV è gia presente, l'elaborazione viene saltata (evita la duplicazione/sovrascrizione)
    - Gestisce errori come file corrotti, JSON malformati o chiavi mancanti
    - Rimuove eventuali file datati dalla directory di destinaione (cleaned_unused_files(...))

    Ritorna:
    - None
    """
    os.makedirs(DATA_CSV_PATH, exist_ok=True)
    generated_csv_files = set()

    for filename in os.listdir(DATA_JSON_PATH):
        if filename.endswith('.json.gz'):
            input_path = os.path.join(DATA_JSON_PATH, filename)
            output_filename = filename.replace('.json.gz', '.csv')
            output_path = os.path.join(DATA_CSV_PATH, output_filename)

            # Salta i file CSV già presenti nella directory
            if os.path.exists(output_path):
                print(f"[INFO] File {output_path} e' gia' presente, salto la creazione.")
                generated_csv_files.add(output_filename)
                continue

            try:
                # Elabora il file JSON e crea il file CSV
                with gzip.open(input_path, 'rb') as gzf:
                    data = gzf.readlines()

                with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
                    csv_writer = csv.writer(csvfile)
                    csv_writer.writerow(CSV_HEADERS)

                    for line in data:
                        try:
                            event = json.loads(line)
                            type_event = event["type"]
                            id_actor = event["actor"]["id"]
                            login_actor = event["actor"]["login"]
                            url_actor = event["actor"]["url"]
                            url_repo = event["repo"]["url"]
                            id_repo = event["repo"]["id"]
                            name_repo = event["repo"]["name"]
                            action_payload = event["payload"].get("action")
                            created_at = event["created_at"]
                            csv_writer.writerow([type_event, id_actor, login_actor, url_actor, url_repo, id_repo, name_repo, action_payload, created_at])
                        except json.JSONDecodeError as e:
                            print(f'[ERROR] File JSON in {input_path}: {e}')
                        except KeyError as e:
                            print(f'[ERROR] Chiave mancante in {input_path}: {e}')

                generated_csv_files.add(output_filename)
                print(f'[INFO] File CSV creato: {output_path}')
            except FileNotFoundError:
                print(f'[ERROR] File non trovato: {input_path}')
            except gzip.BadGzipFile:
                print(f'[ERROR]File gzip corrotto: {input_path}')
            except Exception as e:
                print(f'[ERROR] Errore inaspettato durante l\'elaborazione di {input_path}: {e}')

    clean_unused_files(DATA_CSV_PATH, generated_csv_files, file_extension=".csv")

def clean_csv_files():
    """
    Pulisce i file CSV grezzi e li salva nella directory 'data/github_cleaned_csv'.

    Pulizia per ogni file:
    - Se la versione pulita è già presente, la pulizia viene saltata.
    - Legge il file CSV grezzo.
    - Verifica la presenza della colonna 'type_event', saltando il file se manca.
    - Applica la funzione 'clean_data(df)'.
    - Salva il DataFrame pulito.
    - Gestisce errori di lettura.
    - Rimuove eventuali file datati dalla directory di destinazione.

    Ritorna:
    - None
    """
    os.makedirs(DATA_CLEANED_CSV_PATH, exist_ok=True)
    processed_cleaned_files = set()

    for filename in os.listdir(DATA_CSV_PATH):
        if filename.endswith('.csv'):
            input_path = os.path.join(DATA_CSV_PATH, filename)
            output_path = os.path.join(DATA_CLEANED_CSV_PATH, filename)
            output_filename = filename

            if os.path.exists(output_path):
                print(f"[INFO] File pulito {output_path} e' gia' presente, salto la pulizia.")
                processed_cleaned_files.add(output_filename)
                continue

            try:
                df = pd.read_csv(input_path)
                if 'type_event' not in df.columns:
                    print(f"[ERROR] La colonna 'type_event' non è presente nel file {filename}.")
                    continue

                cleaned_df = clean_data(df)
                cleaned_df.to_csv(output_path, index=False)
                print(f'[INFO] File CSV pulito: {output_path}')
                processed_cleaned_files.add(output_filename)

            except FileNotFoundError:
                print(f'[ERROR] File non trovato: {input_path}')
            except pd.errors.EmptyDataError:
                print(f'[ERROR] File CSV vuoto: {input_path}')
            except Exception as e:
                print(f'[ERROR] Errore durante la pulizia di {input_path}: {e}')

    # Rimuove eventuali file CSV "puliti" datati
    clean_unused_files(DATA_CLEANED_CSV_PATH, processed_cleaned_files, file_extension=".csv")

def clean_data(df):
    """
    Applica una serie di trasformazioni per pulire e normalizzare i dati degli eventi.

    Operazioni effettuate:
    - Converte il campo 'created_at' in formato datetime.
    - Esclude eventi associati ad attori 'bot'.
    - Raggruppa e ordina cronologicamente gli eventi per ogni repo.
    - Calcola la differenza di tempo tra eventi successivi per ogni repo.
    - Normalizza la colonna 'type_event' ('WatchEvent' con action 'started' diventa 'StarEvent').

    Parametri:
    - df: Il DataFrame contenente gli eventi GitHub.

    Ritorna:
    - DataFrame con i dati puliti e normalizzati.
    """
    # Formatta i dati
    df = df.copy()

    if 'created_at' in df.columns:
        df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')

    # Ordina per repo e per timestamp
    if 'id_repo' in df.columns and 'created_at' in df.columns:
        df = df.sort_values(by=['id_repo', 'created_at'])
        df['time_diff'] = df.groupby('id_repo')['created_at'].diff().dt.total_seconds()

    # Normalizza la colonna 'type_event'
    if 'type_event' in df.columns:
        df['type_event'] = df.apply(
            lambda row: 'StarEvent' if row['type_event'] == 'WatchEvent' and row.get('action_payload') == 'started' else row['type_event'],
            axis=1
        )
        
    return df

def main():
    download_files()
    process_files()
    clean_csv_files()

if __name__ == "__main__":
    main()
