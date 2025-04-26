import datetime
import os
import pandas as pd
import pm4py
from pm4py.algo.discovery.dfg import algorithm as dfg_algorithm
from pm4py.visualization.dfg import visualizer as dfg_visualizer

# Percorsi di directory per i dati
DATA_CLEANED_CSV_PATH = 'data/github_cleaned_csv'
DFG_PATH = 'data/process_models' 
TRENDING_REPO_FILE = 'data/trending/trending.csv'

def load_cleaned_csv():
    """
    Carica e concatena tutti i file CSV puliti presenti nella cartella.

    Ritorna:
    - Un unico DataFrame contenente tutti gli eventi concatenati.
      Se nessun file valido viene caricato, il programma termina.
    """
    data = []
    
    for filename in os.listdir(DATA_CLEANED_CSV_PATH):
        if filename.endswith('.csv'):
            file_path = os.path.join(DATA_CLEANED_CSV_PATH, filename)
            try:
                df = pd.read_csv(file_path)

                if 'created_at' in df.columns:
                    df['created_at'] = pd.to_datetime(
                    df['created_at'],
                    format='%Y-%m-%d %H:%M:%S%z',
                    utc=True,
                    errors='coerce'
                )
                
                data.append(df)
            except Exception as e:
                print(f'[ERROR] nel file {filename}: {e}')

    if not data:
        print("[ERROR] Nessun file CSV pulito trovato o caricabile.")
        exit()

    return pd.concat(data, ignore_index=True)

def extract_dfg_metrics(log):
    """
    Estrae metriche strutturali da un Directly-Follows Graph (DFG).

    Parametri:
    - log (DataFrame): Log di processo in formato compatibile con PM4Py.

    Ritorna:
    - Dizionario contenente:
      - num_nodi: Numero di attività distinte nel DFG;
      - num_archi: Numero di transizioni (coppie attività) nel DFG;
      - densità_dfg: Densità del grafo(il rapporto tra archi presenti
        e archi possibili (n*(n-1))).
    """
    dfg = dfg_algorithm.apply(log)

    num_nodi = len(set([x for x, _ in dfg.keys()] + [y for _, y in dfg.keys()]))
    num_archi = len(dfg)
    densità = num_archi / (num_nodi * (num_nodi - 1)) if num_nodi > 1 else 0

    return {
        'num_nodi': num_nodi,
        'num_archi': num_archi,
        'densità_dfg': densità
    }

def extract_features_from_log(log):
    """
    Estrae caratteristiche utili per il machine learning da un log di processo.

    Parametri:
    - log (DataFrame): Log di processo con colonne 'concept:name' e 'time:timestamp'.

    Ritorna:
    - Dizionario contenente:
      - percentuale_StarEvent: Frequenza relativa degli eventi 'StarEvent';
      - tempo_medio_eventi: Tempo medio (in secondi) tra un evento e il successivo.
    """
    # Ordina il log per timestamp
    log = log.sort_values('time:timestamp')

    # Calcola frequenza relativa degli eventi
    eventi_unici = log['concept:name'].value_counts(normalize=True).to_dict()

    # Calcola tempo medio tra eventi
    tempo_medio = (log['time:timestamp'].diff().mean().total_seconds()) if len(log) > 1 else 0

    return {
        'percentuale_StarEvent': eventi_unici.get('StarEvent', 0),
        'tempo_medio_eventi': tempo_medio
    }

def load_trending_repositories():
    """
    Carica la lista delle repository virali da un file CSV.

    Il file deve essere presente nel percorso specificato da TRENDING_REPO_FILE
    e deve contenere una colonna chiamata 'repository' con i nomi delle repository considerate virali.

    Operazioni svolte:
    - Legge il file CSV.
    - Rimuove eventuali valori NaN dalla colonna 'repository'.
    - Estrae solo i nomi univoci delle repository.
    - Controlla che la lista non sia vuota, altrimenti termina l'esecuzione.
    - Gestisce eventuali errori di lettura o assenza del file con un messaggio di errore e `exit(1)`.

    Ritorna:
        list: Una lista di stringhe con i nomi delle repository virali.

    Uscita con Errore:
        Il programma termina con exit(1) se:
        - Il file non esiste o non è leggibile.
        - La colonna 'repository' è vuota o assente.
        - Il file è vuoto
    """
    try:
        trending_df = pd.read_csv(TRENDING_REPO_FILE)
        trending_repos = trending_df['repository'].dropna().unique().tolist()
        if not trending_repos:
            print(f"[ERROR] Nessuna repository trovata nel file {TRENDING_REPO_FILE}: {e}")
            exit(1)
        return trending_repos
    except Exception as e:
        print(f"[ERROR] nel caricamento del file {TRENDING_REPO_FILE}: {e}")
        exit(1)

def load_non_trending_repositories(data, trending_repos):
    """
    Carica un numero limitato di repository non virali, escludendo le virali.

    Parametri:
    - data: DataFrame completo degli eventi
    - trending_repos: lista delle repository virali da escludere

    Ritorna:
    - Lista (massimo max_repos) di repository non virali.
    """
    max_repos = len(trending_repos) # per avere un addestramendo bilanciato
    repo_grouped = data.groupby('name_repo')

    filtered_repos = []
    for repo_name in repo_grouped.groups.keys():
        if repo_name not in trending_repos:
            filtered_repos.append(repo_name)
        if len(filtered_repos) >= max_repos:
            break

    return filtered_repos

def generate_process_models_and_features(dati, repo_list_virale, repo_list_non_virale):
    """
    Genera i modelli di processo (DFG) e le feature di machine learning per ciascuna repository.

    Parametri:
    - dati (pd.DataFrame): dataset degli eventi GitHub normalizzati.
    - repo_list_virale (list): lista di repository virali.        
    - repo_list_non_virale (list): lista di repository non virali.
    - date_str (str): data in formato 'YYYY_MM_DD' per aggiungere temporalità al nome della repo.

    Ritorna:
    - pd.DataFrame: dataframe contenente le feature estratte per ciascuna repository,
        incluse le metriche strutturali del DFG e un flag 'virale'.
    """
    df_features = []

    os.makedirs(DFG_PATH, exist_ok=True)

    for repo_name in repo_list_virale + repo_list_non_virale:
        dati_repo = dati[dati['name_repo'] == repo_name]

        if dati_repo.empty:
            print(f"[ERROR] Nessun evento trovato per la repository {repo_name}.")
            continue

        # Aggiunge il suffisso temporale al nome della repository per identificarla come un'entità separata
        date = datetime.datetime.now(datetime.timezone.utc)
        date_str = date.date().strftime('%Y-%m-%d')
        repo_name_temp = f"{repo_name}_{date_str}"

        dati_repo = dati_repo.rename(columns={'created_at': 'time:timestamp', 'type_event': 'concept:name'})
        dati_repo['case:concept:name'] = repo_name_temp

        log = pm4py.format_dataframe(dati_repo, case_id='case:concept:name', activity_key='concept:name', timestamp_key='time:timestamp')

        # Genera i dfg e le feature delle repo in esame
        dfg_metrics = extract_dfg_metrics(log)
        ml_features = extract_features_from_log(log)

        repo_features = {'repo_name': repo_name_temp, 'virale': repo_name in repo_list_virale}
        repo_features.update(dfg_metrics)
        repo_features.update(ml_features)

        df_features.append(repo_features)

        # Genera e salva il DFG
        safe_repo_name = repo_name_temp.replace("/", "__")
        file_path = os.path.join(DFG_PATH, f"{safe_repo_name}_dfg.png")
        try:
            gviz = dfg_visualizer.apply(dfg_algorithm.apply(log))
            dfg_visualizer.save(gviz, file_path)
            print(f"[INFO] Modello DFG salvato per {repo_name_temp} in '{file_path}'")
        except Exception as e:
            print(f"[ERROR] Impossibile salvare il DFG per {repo_name_temp}: {e}")

    return pd.DataFrame(df_features)

def main():
    # Caricamento di tutti e 24 i file puliti normalizzati CSV
    data = load_cleaned_csv()

    # Caricamento delle repo virali e non virali
    repo_list_virale = load_trending_repositories()
    repo_list_non_virale = load_non_trending_repositories(data, repo_list_virale)

    # Genera i modelli di processo e le feature
    df_features = generate_process_models_and_features(data, repo_list_virale, repo_list_non_virale)

    # Salva il dataset delle feature
    features_path = os.path.join('data', "process_model_features.csv")

    # Se esiste già un file con feature precedenti, lo carica
    if os.path.exists(features_path):
        df_existing = pd.read_csv(features_path)

        # Unisce il vecchio con il nuovo
        df_final = pd.concat([df_existing, df_features], ignore_index=True)
    else:
        df_final = df_features

    # Salva il nuovo dataset aggiornato
    df_final.to_csv(features_path, index=False)

    print(f"[INFO] Feature dataset aggiornato in '{features_path}' con {len(df_final)} repo totali.")

if __name__ == "__main__":
    main()
