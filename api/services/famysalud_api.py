from pathlib import Path
import requests


BASE_DIR = Path(__file__).resolve().parent.parent.parent
CATALOG_URL = "https://app.famysaludec.com/chatbot/catalogo-servicios"


def obtener_catalogo(timeout=5):
    response = requests.get(CATALOG_URL, timeout=timeout)
    response.raise_for_status()
    return response.json()


def normalizar_servicio(servicio, area):
    nombre_area = area.get("title") or area.get("name")

    return {
        "id": servicio.get("id"),
        "nombre": servicio.get("title") or servicio.get("name"),
        "titulo": servicio.get("title"),
        "name": servicio.get("name"),
        "slug": servicio.get("slug"),
        "area": nombre_area,
        "area_id": area.get("id"),
        "categoria": servicio.get("category") or nombre_area,
        "category_id": servicio.get("category_id") or area.get("id"),
        "precio": servicio.get("price") or servicio.get("precio"),
        "precio_promocion": (
            servicio.get("sale_price")
            or servicio.get("promotion_price")
            or servicio.get("precio_promocion")
        ),
        "excerpt": servicio.get("excerpt") or "",
        "description": servicio.get("description") or "",
        "image": servicio.get("image"),
        "presencial": bool(
            servicio.get("is_presential")
            if "is_presential" in servicio
            else servicio.get("presencial")
        ),
        "virtual": bool(
            servicio.get("is_virtual")
            if "is_virtual" in servicio
            else servicio.get("virtual")
        ),
    }


def obtener_servicios_normalizados(catalogo=None):
    catalogo = catalogo or obtener_catalogo()
    servicios_normalizados = []

    areas = catalogo.get("areas", []) if isinstance(catalogo, dict) else []
    for area in areas:
        servicios = area.get("services", []) if isinstance(area, dict) else []
        for servicio in servicios:
            servicios_normalizados.append(normalizar_servicio(servicio, area))

    return {
        "updated_at": catalogo.get("updated_at") if isinstance(catalogo, dict) else None,
        "servicios": servicios_normalizados,
    }
