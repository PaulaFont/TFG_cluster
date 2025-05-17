import os
from vllm import LLM, SamplingParams
import torch

os.environ["CUDA_VISIBLE_DEVICES"]="2,3"


llm = LLM(model="mistralai/Mistral-7B-Instruct-v0.2", dtype="bfloat16", max_model_len=28208)
sampling_params = SamplingParams(temperature=0.1, max_tokens=300)

ocr_txt = """0 ; ye
Dirección General de Soguridad E

COMISARIA DE:LA PROVINCIA DE TOLEDO
GRUPO CIVIL:

OCAÑA > Pongo en su comcimiento que
WM ón el dia de hoy,a las 1530 horas
en la calle de largo Caballero de
esta localidad la camioneta núm60L
al servicio de la Brigada Internacio
nal conducida por el soldadodulvador
Calvin Garrido perteneciente al 1
Xegimiento de Transportes,con resi-
dencia en Corral de -»lmaguer,ha atro
pellado al Guardia de Asalto de la
116 compañia 29 Urupo,D. FRANCISCO
GOMEZ GAXIIDO, causandole lesiones
de pronostico grave,segun comunicado
del ilospital de “angre,en el cual ha
quedado hospitalizado.
Salud y epublica
Ocaña 25 Diciembre 1937
El Subcomisario Jefe

e cmo + Señor.

Instruccion de este partido.+.-Ucaña,

00 06
"""

prompt1_txt = f"""Estoy digitalizando documentos históricos, específicamente relacionados con la Guerra Civil Española, como consejos de guerra. Tras procesar las imágenes con OCR, el texto resultante puede contener errores, omisiones o problemas de formato.

Tu tarea es corregir y mejorar la transcripción sin añadir información que no esté presente en el original. Ajusta palabras incompletas o mal reconocidas en la medida de lo posible, asegurándote de que el resultado sea fiel al documento original. Respeta la estructura y el significado del texto.

Aquí está el texto extraído:

TEXTO:
{ocr_txt}"""

prompt2_txt = f"""Contexto:
Estoy digitalizando documentos históricos relacionados con la Guerra Civil Española, en particular consejos de guerra. Utilizo un sistema OCR para extraer el texto, pero este proceso puede generar errores como caracteres mal interpretados, palabras incompletas o estructuras desordenadas.

Tu Tarea:

1. Corrección: Identifica y corrige errores del OCR sin alterar el significado original.
2. Mejora de legibilidad: Ajusta la puntuación, la gramática y la coherencia sin modificar el contenido.
3. Fidelidad: No agregues información inventada. Si una palabra es ilegible o incompleta, intenta inferirla solo si el contexto lo permite. De lo contrario, márcala con "_".
4. Formato: Mantén la estructura del documento siempre que sea posible. Respeta los párrafos y la disposición del texto.

Ejemplo de Entrada y Salida Esperada

- Entrada (Texto OCR con errores):

"EN EL dia dehoy, el soldadoPdro Gnzalez fué acusdopor susuperior de abandono de sus puesto   duranteel combate. la sntcia fue dictda en el acto."  

- Salida esperada (Texto corregido):

"En el día de hoy, el soldado Pedro González fue acusado por su superior de abandono de su puesto   durante el combate. La sentencia fue dictada en el acto."  

Texto extraído por OCR:
{ocr_txt}
"""


prompt3_txt = f"Corrige el siguiente texto OCR sin añadir información extra:\n\n{ocr_txt}"


# Generate output 1
output = llm.generate(prompt1_txt, sampling_params)
print(output[0].outputs[0].text)

with open('out1_mistralai2.txt', 'w') as archivo:
    archivo.write(output[0].outputs[0].text)

torch.cuda.empty_cache()

# Generate output 2
output = llm.generate(prompt2_txt, sampling_params)
print(output[0].outputs[0].text)


with open('out2_mistralai2.txt', 'w') as archivo:
    archivo.write(output[0].outputs[0].text)

torch.cuda.empty_cache()

# Generate output 3
output = llm.generate(prompt3_txt, sampling_params)
print(output[0].outputs[0].text)


with open('out3_mistralai2.txt', 'w') as archivo:
    archivo.write(output[0].outputs[0].text)

torch.cuda.empty_cache()