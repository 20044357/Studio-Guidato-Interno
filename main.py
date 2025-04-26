import data_collection
import process_mining
import train_model

def main():
  print("\nFase 1: Raccolta dati:")
  data_collection.main()

  print("\nFase 2: Process Mining:")
  process_mining.main()

  #print("\nFase 3: Addestramento del modello:")
  #train_model.main()

if __name__ == '__main__':
    main()
