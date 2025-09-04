import json
from sentence_transformers import SentenceTransformer

def generate_embeddings_from_json(json_path, model_name="intfloat/multilingual-e5-small"):
    """
    Lee un archivo JSON, extrae el texto y genera embeddings usando un modelo SentenceTransformer.

    Args:
        json_path (str): La ruta al archivo JSON.
        model_name (str): El nombre del modelo de SentenceTransformer a usar.

    Returns:
        list: Una lista de diccionarios, donde cada diccionario contiene
              la información original de la sección y su embedding.
    """

    with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    model = SentenceTransformer(model_name)

    sections_with_embeddings = []
    texts_to_embed = []
    original_sections = []

    for item in data["informacion"]:
        if "info" in item and isinstance(item["info"], str):
            texts_to_embed.append(item["info"])
            original_sections.append(item)
        else:
            print(f"Advertencia: Sección sin campo 'info' o no es una cadena: {item}")

    embeddings = model.encode(texts_to_embed, show_progress_bar=True)

    for i, embedding in enumerate(embeddings):
        section_data = original_sections[i]
        section_data["embedding"] = embedding.tolist() # Convertir a lista para JSON serializable
        sections_with_embeddings.append(section_data)

    return sections_with_embeddings

if __name__ == "__main__":
    json_path = "core\services\prueba.json"
    embedded_data = generate_embeddings_from_json(json_path)

    if embedded_data:
        # Opcional: Guardar los embeddings en un nuevo archivo JSON
        output_json_path = "core/services/prueba_embeddings.json"
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(embedded_data, f, ensure_ascii=False, indent=4)
        print(f"Embeddings guardados en: {output_json_path}")
