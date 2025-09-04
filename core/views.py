import markdown
from django.utils.safestring import mark_safe
from django.shortcuts import render, redirect
from core.services import gemini


def index_ask(request): 
    
    historial = request.session.get('historial', [])

    # inicializar historial de conversacion
    if 'historial' not in request.session:
        request.session['historial'] = []
    
    # obtiene el valor del input del template
    if request.method == "POST":
        valor = request.POST.get("campo")

        # verificacion de si presiono el boton de borrar
        if 'borrar' in request.POST:
            request.session['historial'] = []
            request.session.modified = True
            return redirect('vista')

        # Guarda el mensaje escrito por el usuario
        request.session['historial'].append({
            'rol' : 'usuario',
            'mensaje' : valor
        })
    

        # respuesta de gemini
        respuesta = gemini.generate_augmented_response(valor, historial)

         # --- Renderizamos Markdown a HTML ---
        respuesta_html = mark_safe(markdown.markdown(respuesta))

        # Guarda la respuesta de gemini
        request.session['historial'].append({
            'rol' : 'lexrevox',
            'mensaje' : respuesta_html
        })

        # Guarda los cambios en sesion
        request.session.modified = True

        return render(request, "core/index_ask.html", {'historial' : historial})
    
    else: # por si no se ha enviado nada
        return render(request, "core/index_ask.html")




    