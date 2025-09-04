import json
from sentence_transformers import SentenceTransformer

# Cargar el modelo de embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Leer el archivo prueba1.json
with open('core/services/prueba.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

nueva_data = []

for item in data['informacion']:
    # Crear texto concatenado para generar embeddings
    texto_para_embedding = (
        f"Capitulo: {item['capitulo']}\n"
        f"Seccion: {item['seccion']}\n"
        f"Nombre de la Seccion: {item['nombre_seccion']}\n"
        f"Info: {item['info']}"
    )
    
    # Generar embedding
    embedding = model.encode(texto_para_embedding).tolist()
    
    # Construir nuevo objeto con claves claras
    nuevo_item = {
        "capitulo": item['capitulo'],
        "seccion": item['seccion'],
        "nombre_seccion": item['nombre_seccion'],
        "info": item['info'],
        "embedding": embedding
    }
    
    nueva_data.append(nuevo_item)

# Guardar el nuevo archivo con embeddings
with open('core/services/prueba1_embedded.json', 'w', encoding='utf-8') as f:
    json.dump(nueva_data, f, ensure_ascii=False, indent=4)

print("Archivo 'prueba1_embedded.json' generado correctamente con embeddings.")
