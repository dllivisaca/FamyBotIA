import asyncio
import os
import sys
import traceback
from datetime import datetime
from http import HTTPStatus

LOG_PATH = os.path.join(os.path.dirname(__file__), "passenger_debug.log")


def log_debug(message):
    timestamp = datetime.now().isoformat(timespec="seconds")
    with open(LOG_PATH, "a", encoding="utf-8") as log_file:
        log_file.write(f"[{timestamp}] {message}\n")
        log_file.flush()


try:
    log_debug("inicio passenger_wsgi")

    log_debug("antes CURRENT_DIR")
    CURRENT_DIR = os.path.dirname(__file__)
    log_debug(f"despues CURRENT_DIR: {CURRENT_DIR}")

    log_debug("antes sys.path insert")
    sys.path.insert(0, CURRENT_DIR)
    log_debug(f"despues sys.path insert: {sys.path[0]}")

    log_debug("antes import api.app")
    from api.app import app as fastapi_app
    log_debug("despues import api.app")

    def _header_bytes(environ):
        headers = []

        for key, value in environ.items():
            if key.startswith("HTTP_"):
                name = key[5:].replace("_", "-").lower().encode("latin-1")
                headers.append((name, str(value).encode("latin-1")))

        for key, name in (
            ("CONTENT_TYPE", b"content-type"),
            ("CONTENT_LENGTH", b"content-length"),
        ):
            value = environ.get(key)
            if value:
                headers.append((name, str(value).encode("latin-1")))

        return headers

    async def _call_fastapi(environ, body):
        status_code = 500
        response_headers = []
        response_body = []
        request_sent = False

        path = environ.get("PATH_INFO") or "/"
        query_string = (environ.get("QUERY_STRING") or "").encode("latin-1")
        server_name = environ.get("SERVER_NAME") or "localhost"
        server_port = int(environ.get("SERVER_PORT") or 80)
        remote_addr = environ.get("REMOTE_ADDR") or ""

        scope = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": environ.get("SERVER_PROTOCOL", "HTTP/1.1").split("/")[-1],
            "method": environ.get("REQUEST_METHOD", "GET"),
            "scheme": environ.get("wsgi.url_scheme", "http"),
            "path": path,
            "raw_path": path.encode("latin-1"),
            "query_string": query_string,
            "root_path": environ.get("SCRIPT_NAME", ""),
            "headers": _header_bytes(environ),
            "server": (server_name, server_port),
            "client": (remote_addr, 0),
        }

        async def receive():
            nonlocal request_sent
            if request_sent:
                return {"type": "http.disconnect"}
            request_sent = True
            return {"type": "http.request", "body": body, "more_body": False}

        async def send(message):
            nonlocal status_code, response_headers
            message_type = message["type"]

            if message_type == "http.response.start":
                status_code = message["status"]
                response_headers = [
                    (
                        name.decode("latin-1"),
                        value.decode("latin-1"),
                    )
                    for name, value in message.get("headers", [])
                ]
            elif message_type == "http.response.body":
                response_body.append(message.get("body", b""))

        await fastapi_app(scope, receive, send)
        return status_code, response_headers, b"".join(response_body)

    log_debug("antes crear application FastAPI directa")

    def application(environ, start_response):
        try:
            path = environ.get("PATH_INFO") or "/"
            method = environ.get("REQUEST_METHOD", "GET")
            log_debug(f"request FastAPI directa: {method} {path}")

            try:
                content_length = int(environ.get("CONTENT_LENGTH") or 0)
            except ValueError:
                content_length = 0

            body = environ["wsgi.input"].read(content_length) if content_length else b""
            status_code, headers, response_body = asyncio.run(_call_fastapi(environ, body))
            reason = HTTPStatus(status_code).phrase if status_code in HTTPStatus._value2member_map_ else "OK"
            start_response(f"{status_code} {reason}", headers)
            return [response_body]
        except Exception:
            log_debug("ERROR request FastAPI directa")
            log_debug(traceback.format_exc())
            raise

    log_debug("despues crear application FastAPI directa")
except Exception:
    log_debug("ERROR passenger_wsgi")
    log_debug(traceback.format_exc())
    raise
