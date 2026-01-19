"""Run the PenguiFlow Playground UI backend for this agent."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import Response
from fastapi.routing import APIRoute
import uvicorn
from penguiflow.cli import playground as playground_module
from penguiflow.cli.playground import create_playground_app


def main() -> None:
    load_dotenv()
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")
    project_root = Path(__file__).resolve().parents[2]
    app = create_playground_app(project_root=project_root)
    ui_dir = Path(playground_module.__file__).resolve().parent / "playground_ui" / "dist"
    index_path = ui_dir / "index.html"

    def _patch_index_html(html: str) -> str:
        if "pf-uuid-polyfill" in html:
            return html
        polyfill = (
            "<script id=\"pf-uuid-polyfill\">"
            "if (!globalThis.crypto) { globalThis.crypto = {}; }"
            "if (!globalThis.crypto.randomUUID) {"
            "globalThis.crypto.randomUUID = function () {"
            "var s = [], hex = '0123456789abcdef';"
            "for (var i = 0; i < 36; i++) { s[i] = hex[Math.floor(Math.random() * 16)]; }"
            "s[14] = '4';"
            "s[19] = hex[(parseInt(s[19], 16) & 0x3) | 0x8];"
            "s[8] = s[13] = s[18] = s[23] = '-';"
            "return s.join('');"
            "};"
            "}"
            "</script>"
        )
        marker = "<head>"
        if marker in html:
            return html.replace(marker, f"{marker}{polyfill}")
        return f"{polyfill}{html}"

    if index_path.exists():
        app.router.routes = [
            route
            for route in app.router.routes
            if not (
                isinstance(route, APIRoute)
                and route.path == "/"
                and route.methods
                and "GET" in route.methods
            )
        ]

        @app.get("/", include_in_schema=False)
        async def root_ui() -> Response:
            html = index_path.read_text(encoding="utf-8")
            return Response(_patch_index_html(html), media_type="text/html")

        @app.get("/vite.svg", include_in_schema=False)
        async def vite_icon() -> Response:
            svg = (
                "<svg xmlns='http://www.w3.org/2000/svg' width='64' height='64' viewBox='0 0 64 64'>"
                "<rect width='64' height='64' rx='12' fill='#111827'/>"
                "<path d='M18 44L32 12l14 32H18z' fill='#60A5FA'/>"
                "</svg>"
            )
            return Response(svg, media_type="image/svg+xml")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
