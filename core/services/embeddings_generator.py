import json
from sentence_transformers import SentenceTransformer

# Carga el modelo de embeddings
model = SentenceTransformer('BAAI/bge-m3')

# Ruta al archivo JSON de entrada
ruta_json = "core/services/manual.json"

# Función para leer y procesar los fragmentos
def procesar_fragmentos(ruta_json):
    with open(ruta_json, "r", encoding="utf-8") as f:
        datos = json.load(f)

    nuevos_datos = []

    # Aquí accedemos directamente a la lista de fragmentos
    for fragmento in datos.get("informacion", []):
        contenido = fragmento.get("contenido", [])
        texto_concatenado = ""

        for item in contenido:
            if isinstance(item, dict) and item.get("tipo") == "texto":
                texto_concatenado += item.get("texto", "") + " "

        texto_concatenado = texto_concatenado.strip()
        if not texto_concatenado:
            continue

        # Generar el embedding solo a partir del texto concatenado
        embedding = model.encode(texto_concatenado, normalize_embeddings=True).tolist()

        # Agregar el embedding al fragmento original
        fragmento["embedding"] = embedding
        nuevos_datos.append(fragmento)

    return nuevos_datos

# Ejecutar procesamiento
resultado = procesar_fragmentos(ruta_json)

# Guardar el resultado en un nuevo JSON
with open("core/services/manual_embeddings.json", "w", encoding="utf-8") as f:
    json.dump(resultado, f, indent=4, ensure_ascii=False)

print("Archivo generado correctamente.")
