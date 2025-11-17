import json

def convert_test_results_to_results(input_file='test_results.json', output_file='results.json'):
    
    # Lire le fichier d'entrée
    with open(input_file, 'r') as f:
        test_results = json.load(f)
    
    # Convertir au nouveau format
    results = {}
    for item in test_results:
        # Extraire le numéro du nom de fichier (sans l'extension .jpg)
        file_number = item['filename'].replace('.jpg', '')
        results[file_number] = item['predicted']
    
    # Écrire le fichier de sortie
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"Conversion terminée: {len(results)} entrées converties")
    print(f"Fichier de sortie: {output_file}")

if __name__ == '__main__':
    convert_test_results_to_results()
