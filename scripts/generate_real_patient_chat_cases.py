import json
import re
import unicodedata
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_PATH = BASE_DIR / "tests" / "real_patient_chat_intentions.json"

CATEGORY_ALIASES = {
    "cotizar_servicios": "cotizar_servicio",
    "horarios": "consultar_horario",
    "agendamiento": "agendar_cita",
    "ubicacion": "consultar_ubicacion",
    "asesor": "hablar_asesor",
}


def case(text, intent, intents, action, category):
    return {
        "texto": text,
        "intencion_esperada": intent,
        "intenciones_detectadas_esperadas": intents,
        "accion_esperada": action,
        "categoria_fuente": CATEGORY_ALIASES.get(category, category),
    }


CASES = [
    case("Hola buenas", "saludo", ["saludo"], "respuesta_simple", "saludo"),
    case("Buenas tardes", "saludo", ["saludo"], "respuesta_simple", "saludo"),
    case("Buenos días", "saludo", ["saludo"], "respuesta_simple", "saludo"),
    case("Hola", "saludo", ["saludo"], "respuesta_simple", "saludo"),
    case("Buen día", "saludo", ["saludo"], "respuesta_simple", "saludo"),
    case("Buenas noches", "saludo", ["saludo"], "respuesta_simple", "saludo"),
    case("Hhola", "saludo", ["saludo"], "respuesta_simple", "saludo"),
    case("buenas tardes", "saludo", ["saludo"], "respuesta_simple", "saludo"),
    case("holaaaaa", "saludo", ["saludo"], "respuesta_simple", "saludo"),

    case("Hola que precio tiene la ecografía venosa", "consulta_servicios", ["consulta_servicios", "cotizar_servicio"], "listar_opciones", "cotizar_servicios"),
    case("Que valor tiene una audiometría", "consulta_servicios", ["consulta_servicios", "cotizar_servicio"], "respuesta_directa", "cotizar_servicios"),
    case("Buenas tardes disculpe que valor tiene consulta con neumología y dispone también de espirometria y electrocardiograma?", "consulta_servicios", ["consulta_servicios", "cotizar_servicio", "consulta_especialidades"], "listar_opciones", "cotizar_servicios"),
    case("Buenas, que costo tiene una ecografia hepatobiliar?", "consulta_servicios", ["consulta_servicios", "cotizar_servicio"], "listar_opciones", "cotizar_servicios"),
    case("Necesito saber el costo de una ecografia hepatobiliar", "consulta_servicios", ["consulta_servicios", "cotizar_servicio"], "listar_opciones", "cotizar_servicios"),
    case("Cuánto cuesta un lavado de oído?", "consulta_servicios", ["consulta_servicios", "cotizar_servicio"], "respuesta_directa", "cotizar_servicios"),
    case("Que precio tiene la consulta con traumatología", "consulta_servicios", ["consulta_servicios", "consulta_especialidades", "cotizar_servicio"], "respuesta_directa", "cotizar_servicios"),
    case("Hola buen dia cual es el costo de la resonancia magnética", "consulta_servicios", ["consulta_servicios", "cotizar_servicio"], "listar_opciones", "cotizar_servicios"),
    case("Una consulta que costo tiene electromiografia", "consulta_servicios", ["consulta_servicios", "cotizar_servicio"], "listar_opciones", "cotizar_servicios"),
    case("Qué precio tienen las tomografías", "consulta_servicios", ["consulta_servicios", "cotizar_servicio"], "listar_opciones", "cotizar_servicios"),
    case("Buen día que costo tiene la consulta para neumólogia ??", "consulta_servicios", ["consulta_servicios", "consulta_especialidades", "cotizar_servicio"], "respuesta_directa", "cotizar_servicios"),
    case("Cuál es el costo de la primera consulta para embarazo y ecografía", "consulta_servicios", ["consulta_servicios", "cotizar_servicio"], "listar_opciones", "cotizar_servicios"),
    case("Quisiera saber cuánto cuesta la consulta de primer embarazo y ecografía", "consulta_servicios", ["consulta_servicios", "cotizar_servicio"], "listar_opciones", "cotizar_servicios"),
    case("Hola buenas noches Quisiera saber si hacen resonancia cerebral con contraste y en que precio?", "consulta_servicios", ["consulta_servicios", "cotizar_servicio"], "listar_opciones", "cotizar_servicios"),
    case("Buenas noches disculpe hacen ecografías de tórax?? Y si , si que costó tiene", "consulta_servicios", ["consulta_servicios", "cotizar_servicio"], "listar_opciones", "cotizar_servicios"),
    case("Buenas noches, que cuesta el lavado de oidos?", "consulta_servicios", ["consulta_servicios", "cotizar_servicio"], "respuesta_directa", "cotizar_servicios"),
    case("buenas tardes m puede ayudar con el costo d la consulta", "cotizar_servicio", ["cotizar_servicio"], "listar_opciones", "cotizar_servicios"),
    case("Cuál es el valor de una resonancia magnética de columna completa", "consulta_servicios", ["consulta_servicios", "cotizar_servicio"], "listar_opciones", "cotizar_servicios"),
    case("Buenos días que precio tiene la consulta y donde están ubicados disculpe gracias", "cotizar_servicio", ["cotizar_servicio", "consultar_ubicacion"], "consulta_ubicacion", "cotizar_servicios"),
    case("Muy buenas noches, querria saber si en esta sección se hace las pruebas de embarazo de sangre? cuanto seria el costo?", "consulta_servicios", ["consulta_servicios", "cotizar_servicio"], "respuesta_directa", "cotizar_servicios"),
    case("Eco transvaginal precio?", "consulta_servicios", ["consulta_servicios", "cotizar_servicio"], "respuesta_directa", "cotizar_servicios"),
    case("Valor de la consulta, muchas gracias.", "cotizar_servicio", ["cotizar_servicio"], "listar_opciones", "cotizar_servicios"),
    case("Cuanto salen las protesis", "consulta_servicios", ["consulta_servicios", "cotizar_servicio"], "listar_opciones", "cotizar_servicios"),
    case("Buenas noches que precio tiene la recanalizacion de trompas", "consulta_servicios", ["consulta_servicios", "cotizar_servicio"], "respuesta_directa", "cotizar_servicios"),
    case("para sacar un tercer molar q vale.", "consulta_servicios", ["consulta_servicios", "cotizar_servicio"], "listar_opciones", "cotizar_servicios"),
    case("costo de un tratamiento de conducto", "consulta_servicios", ["consulta_servicios", "cotizar_servicio"], "respuesta_directa", "cotizar_servicios"),
    case("Q precio tiene el calce de muela", "consulta_servicios", ["consulta_servicios", "cotizar_servicio"], "listar_opciones", "cotizar_servicios"),
    case("Precio de blancamiento dental", "consulta_servicios", ["consulta_servicios", "cotizar_servicio"], "respuesta_directa", "cotizar_servicios"),
    case("Los calces q precio tienen", "consulta_servicios", ["consulta_servicios", "cotizar_servicio"], "listar_opciones", "cotizar_servicios"),
    case("Buen dia , precio de extracion de muela o tercer molar", "consulta_servicios", ["consulta_servicios", "cotizar_servicio"], "listar_opciones", "cotizar_servicios"),
    case("Quisiera saber precio de una extracion de muela o tercer molar", "consulta_servicios", ["consulta_servicios", "cotizar_servicio"], "listar_opciones", "cotizar_servicios"),
    case("Donde están ubicados y cuanto vale la consulta con el gastroenterologo", "consulta_especialidades", ["consultar_ubicacion", "cotizar_servicio", "consulta_especialidades"], "respuesta_directa", "cotizar_servicios"),
    case("Buenas noches precio de un ecocardiogrma y una prueba de holter de arritmia", "consulta_servicios", ["consulta_servicios", "cotizar_servicio"], "listar_opciones", "cotizar_servicios"),

    case("Realizan resonancias magnéticas ?", "consulta_servicios", ["consulta_servicios"], "listar_opciones", "consulta_servicios"),
    case("Información de ecografía abdominal completa", "consulta_servicios", ["consulta_servicios"], "respuesta_directa", "consulta_servicios"),
    case("Buenas, tmb quitan caries?", "consulta_servicios", ["consulta_servicios"], "respuesta_directa", "consulta_servicios"),
    case("Hola quisiera saber si uds hacen operaciones", "consulta_servicios", ["consulta_servicios"], "listar_opciones", "consulta_servicios"),
    case("hiperplasia prostatica, calcificaciones en la prostata", "consulta_servicios", ["consulta_servicios"], "listar_opciones", "consulta_servicios"),
    case("Buenas tardes resonancia magnética", "consulta_servicios", ["consulta_servicios"], "listar_opciones", "consulta_servicios"),
    case("Buenos días, quisiera saber si hacen exámenes de audiometría", "consulta_servicios", ["consulta_servicios"], "respuesta_directa", "consulta_servicios"),
    case("Hola buenas que cuesta una prueba de embarazo", "consulta_servicios", ["consulta_servicios", "cotizar_servicio"], "respuesta_directa", "consulta_servicios"),
    case("Buenas tardes que precio tiene el eco de las 20 semanas, eco morfológico", "consulta_servicios", ["consulta_servicios", "cotizar_servicio"], "respuesta_directa", "consulta_servicios"),
    case("Precio de la ecografía de las 20 semanas de embarazo", "consulta_servicios", ["consulta_servicios", "cotizar_servicio"], "respuesta_directa", "consulta_servicios"),
    case("3er molares", "consulta_servicios", ["consulta_servicios"], "listar_opciones", "consulta_servicios"),
    case("Buen día realizan exámenes en las mamas", "consulta_servicios", ["consulta_servicios"], "listar_opciones", "consulta_servicios"),
    case("Tiene servicio para mamografía", "consulta_servicios", ["consulta_servicios"], "respuesta_directa", "consulta_servicios"),
    case("Realizan tomografia de craneo", "consulta_servicios", ["consulta_servicios"], "listar_opciones", "consulta_servicios"),
    case("Tomografia de craneo", "consulta_servicios", ["consulta_servicios"], "listar_opciones", "consulta_servicios"),
    case("Le pregunto si realizan tomografia de craneo", "consulta_servicios", ["consulta_servicios"], "listar_opciones", "consulta_servicios"),
    case("Buenos días que precio tiene el eco renal", "consulta_servicios", ["consulta_servicios", "cotizar_servicio"], "respuesta_directa", "consulta_servicios"),
    case("Siasenprotesis", "consulta_servicios", ["consulta_servicios"], "listar_opciones", "consulta_servicios"),
    case("Quiero saber sirealisanprotesis", "consulta_servicios", ["consulta_servicios"], "listar_opciones", "consulta_servicios"),
    case("uds mismo hacen la radiografía", "consulta_servicios", ["consulta_servicios"], "listar_opciones", "consulta_servicios"),
    case("Buenos dias en su centro realizan resonancia contrastada abdominal", "consulta_servicios", ["consulta_servicios"], "respuesta_directa", "consulta_servicios"),
    case("Buenas tardes hacen tratamiento con gas de la risa q valor tiene es para un niño de tres años", "consulta_servicios", ["consulta_servicios", "cotizar_servicio"], "respuesta_directa", "consulta_servicios"),
    case("Quiero aserme limpieza y tapar caries", "consulta_servicios", ["consulta_servicios"], "listar_opciones", "consulta_servicios"),
    case("Necesito sacarme las cordales ya tengo la panorámica", "consulta_servicios", ["consulta_servicios"], "listar_opciones", "consulta_servicios"),

    case("cardiología", "consulta_especialidades", ["consulta_especialidades"], "respuesta_directa", "consulta_especialidades"),
    case("tienen nutrición", "consulta_especialidades", ["consulta_especialidades"], "respuesta_directa", "consulta_especialidades"),
    case("Buen día cuenta con la especialidad de Cardiólogia?", "consulta_especialidades", ["consulta_especialidades"], "respuesta_directa", "consulta_especialidades"),
    case("Servicio de cardiología", "consulta_especialidades", ["consulta_especialidades"], "respuesta_directa", "consulta_especialidades"),
    case("Urología", "consulta_especialidades", ["consulta_especialidades"], "respuesta_directa", "consulta_especialidades"),
    case("Oftalmología", "consulta_especialidades", ["consulta_especialidades"], "respuesta_directa", "consulta_especialidades"),

    case("Hola atienden mañana o el sábado", "consultar_horario", ["consultar_horario"], "respuesta_simple", "horarios"),
    case("¿Cuál es su horario de atención?", "consultar_horario", ["consultar_horario"], "respuesta_simple", "horarios"),
    case("Atienden los días sábados", "consultar_horario", ["consultar_horario"], "respuesta_simple", "horarios"),
    case("Buenos días disculpe hoy atienden", "consultar_horario", ["consultar_horario"], "respuesta_simple", "horarios"),
    case("Quisiera saber si hoy atienden", "consultar_horario", ["consultar_horario"], "respuesta_simple", "horarios"),
    case("Horarios de atención", "consultar_horario", ["consultar_horario"], "respuesta_simple", "horarios"),
    case("favor indicar dirección y horario", "consultar_ubicacion", ["consultar_ubicacion", "consultar_horario"], "respuesta_simple", "horarios"),

    case("me gustaría agendar una cita de laboratorio, que incluya este paquete", "agendar_cita", ["agendar_cita"], "iniciar_agendamiento", "agendamiento"),
    case("Buenas necesito que me ayuden con una orden de tomografía", "consulta_servicios", ["consulta_servicios"], "listar_opciones", "agendamiento"),
    case("¿Dónde puedo agendar mi cita?", "agendar_cita", ["agendar_cita", "consultar_ubicacion"], "respuesta_simple", "agendamiento"),
    case("¿Puedo reservar una cita?", "agendar_cita", ["agendar_cita"], "iniciar_agendamiento", "agendamiento"),
    case("¿Como puedo agendar una cita?", "agendar_cita", ["agendar_cita"], "iniciar_agendamiento", "agendamiento"),
    case("Quiero reservar una cita.", "agendar_cita", ["agendar_cita"], "iniciar_agendamiento", "agendamiento"),
    case("Algún número para comunicarme o sacar cita", "agendar_cita", ["agendar_cita"], "iniciar_agendamiento", "agendamiento"),

    case("Una pregunta donde puedo entregar mi hoja de vida", "trabajo", ["trabajo"], "abrir_trabajo", "trabajo"),
    case("Muy buenas tardes me comunicaba con ustedes para saber si tienen un correo de recursos humanos que puedan brindarme", "trabajo", ["trabajo"], "abrir_trabajo", "trabajo"),
    case("Soy enfermera auxiliar", "trabajo", ["trabajo"], "abrir_trabajo", "trabajo"),
    case("Buenas tardes, disculpe están necesitando personal médico ? Algún correo al que pueda enviar un curriculum", "trabajo", ["trabajo"], "abrir_trabajo", "trabajo"),
    case("Hola buenas noches me ayuda con un correo para enviar mi CV", "trabajo", ["trabajo"], "abrir_trabajo", "trabajo"),
    case("Buenas tardes, saludos Le saluda Obst. Luis Chavez.\nEscribia con el fin de obtener informacion sobre el centro médico y si existe oferta laboral para servicios de consulta en area de obstetricia y ginecologia, diplomado en ecografia y colposcopia.", "trabajo", ["trabajo"], "abrir_trabajo", "trabajo"),
    case("buenas tardes me pueden ayudar con un mail para trabajo soy enfermera profesional", "trabajo", ["trabajo"], "abrir_trabajo", "trabajo"),
    case("Solicito comedidamente me ayude direccionando un correo electrónico para enviar mi hoja de vida", "trabajo", ["trabajo"], "abrir_trabajo", "trabajo"),
    case("Buen día disculpe tienen alguna vacante laboral", "trabajo", ["trabajo"], "abrir_trabajo", "trabajo"),
    case("Buenas noches con mi respeto usted no nesesita una doc odontologa", "trabajo", ["trabajo"], "abrir_trabajo", "trabajo"),
    case("buenas tardes me llamo yerferzon Urdaneta soy médico general me gustaría trabajar con ustedes", "trabajo", ["trabajo"], "abrir_trabajo", "trabajo"),
    case("buenas noches de pronto no estarán contratando aux de fisioterapia", "trabajo", ["trabajo"], "abrir_trabajo", "trabajo"),
    case("Mi nombre es Jennyfer Patiño Paredes\nHe trabajado como asistente dental desde hace años", "trabajo", ["trabajo"], "abrir_trabajo", "trabajo"),
    case("Buenos días, el día de ayer envié a su correo electrónico un mail con mi CV", "trabajo", ["trabajo"], "abrir_trabajo", "trabajo"),
    case("buenas tardes disculpe la molestia pero dentro de su personal requieren de terapeuta respiratorio?", "trabajo", ["trabajo"], "abrir_trabajo", "trabajo"),
    case("Soy medico general", "trabajo", ["trabajo"], "abrir_trabajo", "trabajo"),
    case("Deseo entragar ni curriculo\nTiene algun correo para poderlo enviar ?\nO me puedo hacercar a entregarlo?", "trabajo", ["trabajo"], "abrir_trabajo", "trabajo"),

    case("¿Cuál es su ubicación?", "consultar_ubicacion", ["consultar_ubicacion"], "respuesta_simple", "ubicacion"),
    case("Dónde están ubicados", "consultar_ubicacion", ["consultar_ubicacion"], "respuesta_simple", "ubicacion"),
    case("fonde estan ubicadod yo vivo en Milagro", "consultar_ubicacion", ["consultar_ubicacion"], "respuesta_simple", "ubicacion"),
    case("donde están ubicados.", "consultar_ubicacion", ["consultar_ubicacion"], "respuesta_simple", "ubicacion"),
    case("deme su ubicación", "consultar_ubicacion", ["consultar_ubicacion"], "respuesta_simple", "ubicacion"),
    case("donde ubicados", "consultar_ubicacion", ["consultar_ubicacion"], "respuesta_simple", "ubicacion"),
    case("Donde esta ubicado?", "consultar_ubicacion", ["consultar_ubicacion"], "respuesta_simple", "ubicacion"),
    case("Ola mucho gusto disculpe dirección exacta en Guayaquil", "consultar_ubicacion", ["consultar_ubicacion"], "respuesta_simple", "ubicacion"),
    case("disculpe dirección exacta en Guayaquil", "consultar_ubicacion", ["consultar_ubicacion"], "respuesta_simple", "ubicacion"),
    case("dirección exacta en Guayaquil", "consultar_ubicacion", ["consultar_ubicacion"], "respuesta_simple", "ubicacion"),

    case("Buenos días tiene atención a domicilio gracias", "hablar_asesor", ["hablar_asesor"], "derivar_asesor", "asesor"),
    case("Buenas noches para hacerme una ecografia es necesario sacar cita.", "consulta_servicios", ["consulta_servicios", "agendar_cita"], "listar_opciones", "asesor"),
    case("Quiero hacerme una ecografia es necesario sacar cita.", "consulta_servicios", ["consulta_servicios", "agendar_cita"], "listar_opciones", "asesor"),

    case("Me recomendaron hacerme una mamografía. puedo agendar el examen directamente? Qué horarios tienen disponibles y cuál sería el costo?", "consulta_servicios", ["consulta_servicios", "agendar_cita", "consultar_horario", "cotizar_servicio"], "respuesta_directa", "multi_intencion"),
    case("quisiera información sobre las áreas de pediatría y dermatologia", "consulta_especialidades", ["consulta_especialidades"], "listar_opciones", "multi_intencion"),
    case("Disculpe una última interrogante hay que agendar cita o solo acercarse?", "agendar_cita", ["agendar_cita"], "iniciar_agendamiento", "multi_intencion"),
]


COMMON_REPLACEMENTS = [
    ("Buenos días", "Bnos dias"),
    ("Buenas tardes", "Bnas tardes"),
    ("Buenas noches", "Bnas noches"),
    ("ecografía", "eco"),
    ("ecografia", "eco"),
    ("resonancia magnética", "RM"),
    ("precio", "precio"),
    ("ubicados", "ubicadps"),
    ("agendar", "ajendar"),
]


def strip_accents(text):
    normalized = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")


def collapse_spaces(text):
    return re.sub(r"\s+", " ", text).strip()


def variant_without_accents(text):
    value = strip_accents(text)
    return value if value != text else None


def variant_common_typo(text):
    for old, new in COMMON_REPLACEMENTS:
        if old in text:
            return text.replace(old, new, 1)
    return None


def variant_case(text):
    if len(text) < 80:
        return text.upper()
    return None


def variant_order(text):
    lower = strip_accents(text.lower())
    if "ubic" in lower and "precio" in lower:
        return "Dónde están ubicados y qué precio tiene la consulta"
    if "horario" in lower and "direccion" in lower:
        return "Horario y dirección por favor"
    return None


def build_cases():
    generated = []
    seen = set()

    def add(item, variant_type="base", source_text=None):
        text = collapse_spaces(item["texto"])
        key = text.lower()
        if key in seen:
            return
        seen.add(key)
        clone = dict(item)
        clone["texto"] = text
        clone["variant_type"] = variant_type
        if source_text:
            clone["source_text"] = source_text
        generated.append(clone)

    for item in CASES:
        add(item)
        variants = [
            ("sin_tildes", variant_without_accents(item["texto"])),
            ("errores_ortograficos", variant_common_typo(item["texto"])),
            ("mayusculas", variant_case(item["texto"])),
            ("cambio_orden", variant_order(item["texto"])),
        ]
        for variant_type, text in variants:
            if not text:
                continue
            variant = dict(item)
            variant["texto"] = text
            add(variant, variant_type, item["texto"])

    return generated


def main():
    cases = build_cases()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "description": "Casos reales de pacientes para validar /chat: intencion, intenciones_detectadas y accion.",
        "total_cases": len(cases),
        "cases": cases,
    }
    OUTPUT_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Generated {len(cases)} cases at {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
