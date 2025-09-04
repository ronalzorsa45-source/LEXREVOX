import google.generativeai as genai
from PIL import Image
import os

# 1. Carga tu API Key desde una variable de entorno o directamente
genai.configure(api_key="AIzaSyAjphNoOPUjeaO8klCj5pM1QWqnj1kou7Q")

# 2. Carga el modelo Gemini 1.5 Flash
model = genai.GenerativeModel(model_name="gemini-2.0-flash")

# 3. Carga la imagen con PIL
image_path = "core/services/test/imagen_def.PNG"
image = Image.open(image_path)

# 4. Genera la respuesta
response = model.generate_content(
    [
        "Describe lo que ves en la imagen:",
        image
    ]
)

# 5. Muestra la respuesta
print(response.text)
