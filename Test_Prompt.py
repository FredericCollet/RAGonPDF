import gradio as gr
import httpx
import json

OLLAMA_URL = "http://192.168.1.123:11434"  # PC_MAX
#OLLAMA_URL = "http://192.168.1.73:11434" # PC_HENRI

def stream_ollama(prompt,histo):
    stream_text = ""
    context="MSEA-1711 1 CONTRATO DE PROMESA DE COMPRAVENTA DE COSA FUTURA SUJETO A CONDICIÓN SUSPENSIVA, QUE CELEBRAN POR UNA PARTE " \
    "DE C.V., POR MEDIO DE SU REPRESENTANTE LEGAL, EL SEÑOR MANUEL ENRIQUE CANO DE ANDA, A QUIEN EN LO SUCESIVO SE LE IDENTIFICARA COMO LA “DESARROLLADORA”, MISMAS QUE SE SUJETAN AL TENOR DE LOS SIGUIENTES: ANTECEDENTES PRIMERO.- Que mediante Escritura Pública número 38,257, de fecha 21 de diciembre de 2018, otorgada ante la fe del Lic."\
    "Gustavo Escamilla Flores, titular de la Notaria Pública número 26, se celebró un Contrato de Compra Venta de Inmueble, entre Disenio y Construcciones Óptimas, S.A. de C.V. en su carácter de parte vendedora, y por la otra parte Desarrollos Inmobiliarios Proyectos 9, S.A.P.I."\
    "DESARROLLOS INMOBILIARIOS PROYECTOS 9, S.A.P.I DE C.V. REPRESENTADA EN ESTE ACTO POR EL SEÑOR MANUEL ENRIQUE CANO DE ANDA, COMPARECIENDO EN SU CARÁCTER DE COMERCIALIZADOR Y FIDEICOMITENTE Y FIDEICOMISARIO, DE UN FUTURO A QUIEN EN LO SUCESIVO SE LE DENOMINARÁ \"LA PROMITENTE VENDEDORA\"; Y POR LA OTRA PARTE LA SEÑORA MARIA SUSANA ESPARZA ALONSO, POR SUS PROPIOS DERECHOS, A QUIEN EN LO SUCESIVO SE LE DENOMINARÁ COMO “LA PROMITENTE COMPRADORA” , et voila.1711 7 SEXTA.- INCUMPLIMIENTO DEL CONTRATO: En caso de que se cumpla la primera de las condiciones suspensivas convenidas en este instrumento y cualquiera de las partes decidiera no seguir adelante con los compromisos contraídos mediante el presente contrato, la parte inocente podrá exigir el cumplimiento forzoso de este contrato o bien la rescisión del mismo, sin necesidad de declaración judicial alguna. 1711 3 g) Que la PROMITENTE VENDEDORA ha hecho del conocimiento a la PROMITENTE COMPRADORA que ha constituido o se puede llegar a constituir un gravamen en los" \
    " INMUEBLES sobre los cuales se realizará el DESARROLLO INMOBILIARIO, y en consecuencia sobre la parte proporcional de la UNIDAD PRIVATIVA, con el propósito de " \
    "garantizar el financiamiento que la PROMITENTE VENDEDORA solicitó o está solicitando ante diversas entidades financieras, de ser el caso, una copia de" \
    " la Escritura Pública en donde se haga constar la constitución del gravamen será puesto a disposición de la PROMITENTE COMPRADORA de conformidad con el artículo 73 Bis de la Ley Federal de Protección al Consumidor."
    message=prompt    
    #  📄 Concatène les résultats pour créer le contexte du prompt
    #context = "\n\n".join([f"Document: {res['filename']}\nTexte: {res['text']}" for res in search_results])
    prompt = f"""
    Eres un asistente inteligente. Responde a la siguiente pregunta utilizando únicamente la siguiente información:
    {context}

    Question : {message}
    Réponse :
    """
    LePrompt=prompt

    print(f"🔹 {LePrompt}") 

    try:
        with httpx.stream(
            "POST",
            f"{OLLAMA_URL}/api/generate",
            #json={"model": "mistral", "prompt": LePrompt},
            json={"model": "gemma3", "prompt": LePrompt},
            #json={"model": "llama2", "prompt": LePrompt},
            #timeout=60.0
            timeout=120.0
        ) as response:
            response.raise_for_status()
            yield "✅ Connexion établie. Génération en cours...\n"

            for line in response.iter_lines():
                if line:
                    print(f"🔹 {line}")  # ← utile pour debug
                    data = json.loads(line)
                    token = data.get("response", "")
                    stream_text += token
                    yield stream_text
                    if data.get("done", False):
                        break
    except Exception as e:
        yield f"❌ Erreur : {str(e)}"

demo = gr.ChatInterface(stream_ollama, title="Test (Prompt) Streaming Ollama")

demo.launch(share=False)