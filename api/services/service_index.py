import re
import unicodedata


def normalizar_texto(texto):
    texto = str(texto or "").strip().lower()
    texto = unicodedata.normalize("NFD", texto)
    return "".join(
        caracter
        for caracter in texto
        if unicodedata.category(caracter) != "Mn"
    )


def slug_a_texto(slug):
    return " ".join(re.findall(r"[a-z0-9]+", normalizar_texto(slug)))


def obtener_sinonimos_para_servicio(servicio, sinonimos_catalogo=None):
    sinonimos_catalogo = sinonimos_catalogo or {}
    texto_servicio = normalizar_texto(
        " ".join(
            str(valor)
            for valor in (
                servicio.get("nombre"),
                servicio.get("titulo"),
                servicio.get("name"),
                servicio.get("slug"),
                servicio.get("area"),
                servicio.get("categoria"),
                servicio.get("excerpt"),
                servicio.get("description"),
            )
            if valor
        )
    )
    sinonimos = []

    for frase, termino_catalogo in sinonimos_catalogo.items():
        termino_normalizado = normalizar_texto(termino_catalogo)
        if termino_normalizado and termino_normalizado in texto_servicio:
            sinonimos.append(frase)

    return sinonimos


def obtener_terminos_medicos_controlados(servicio):
    nombre = normalizar_texto(servicio.get("nombre"))
    area = normalizar_texto(servicio.get("area"))
    terminos = []

    if area == "ecografias":
        terminos.extend(["ecografia", "eco", "ultrasonido"])

    if area == "ecografias" and (
        "abdominal" in nombre
        or "abdomen" in nombre
        or "abdomino" in nombre
    ):
        terminos.extend([
            "higado",
            "hepatobiliar",
            "hepatico",
            "vesicula",
            "biliar",
            "abdomen",
        ])

    if "tercer molar" in nombre:
        terminos.extend([
            "muela del juicio",
            "cordal",
            "cordales",
            "tercer molar",
            "3er molar",
        ])

    return terminos


def construir_texto_indexable(servicio, sinonimos_catalogo=None):
    partes = []
    nombre = servicio.get("nombre")
    titulo = servicio.get("titulo")
    name = servicio.get("name")
    area = servicio.get("area")
    categoria = servicio.get("categoria")
    slug_texto = slug_a_texto(servicio.get("slug"))
    excerpt = servicio.get("excerpt")
    description = servicio.get("description")

    if nombre:
        partes.append(str(nombre))
    if titulo and titulo != nombre:
        partes.append(str(titulo))
    if name and name not in {nombre, titulo}:
        partes.append(str(name))
    if area:
        partes.append(f"Area: {area}")
    if categoria and categoria != area:
        partes.append(f"Categoria: {categoria}")
    if slug_texto:
        partes.append(f"Slug: {slug_texto}")
    if excerpt:
        partes.append(str(excerpt))
    if description:
        partes.append(str(description))

    sinonimos = obtener_sinonimos_para_servicio(servicio, sinonimos_catalogo)
    if sinonimos:
        partes.append("Sinonimos: " + ", ".join(sorted(set(sinonimos))))

    terminos_controlados = obtener_terminos_medicos_controlados(servicio)
    if terminos_controlados:
        partes.append("Terminos medicos: " + ", ".join(sorted(set(terminos_controlados))))

    return ". ".join(partes)


def construir_documento_servicio(servicio, sinonimos_catalogo=None):
    metadata = {
        "id": servicio.get("id"),
        "nombre": servicio.get("nombre"),
        "area": servicio.get("area"),
        "precio": servicio.get("precio"),
        "precio_promocion": servicio.get("precio_promocion"),
        "presencial": servicio.get("presencial"),
        "virtual": servicio.get("virtual"),
        "slug": servicio.get("slug"),
    }

    return {
        "id": servicio.get("id"),
        "texto": construir_texto_indexable(servicio, sinonimos_catalogo),
        "metadata": metadata,
    }


def construir_documentos_servicios(servicios, sinonimos_catalogo=None):
    return [
        construir_documento_servicio(servicio, sinonimos_catalogo)
        for servicio in servicios
        if servicio.get("id") is not None and servicio.get("nombre")
    ]
