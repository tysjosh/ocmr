"""Entry-point script: start the OCM API_Service (Req 28.1).

Thin ``argparse`` wrapper around ``uvicorn`` that serves the FastAPI app
exposed at ``ocm.app.main:app``. ``uvicorn`` is imported **lazily** inside
:func:`main` so importing this module (e.g. for testing argument parsing)
never requires uvicorn to be installed.

Usage::

    python -m ocm.scripts.serve                       # 127.0.0.1:8000
    python -m ocm.scripts.serve --host 0.0.0.0 --port 8080
    python -m ocm.scripts.serve --reload              # dev auto-reload

Requirements: 28.1.
"""

from __future__ import annotations

import argparse
from typing import Optional, Sequence

#: Import string uvicorn uses to locate the ASGI app (supports --reload).
APP_IMPORT_STRING = "ocm.app.main:app"

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for the serve command."""
    parser = argparse.ArgumentParser(
        prog="ocm-serve",
        description="Start the Ontology-Constrained Memory API service.",
    )
    parser.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help=f"Host interface to bind (default: {DEFAULT_HOST}).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"Port to bind (default: {DEFAULT_PORT}).",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload on code changes (development only).",
    )
    parser.add_argument(
        "--log-level",
        default="info",
        help="uvicorn log level (default: info).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse arguments and report the configuration without binding a port.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Parse arguments and launch the API via uvicorn.

    With ``--dry-run`` the server is *not* started; the resolved configuration
    is printed instead (useful for smoke checks and CI). Returns a process exit
    code (0 on success).
    """
    args = build_parser().parse_args(argv)

    if args.dry_run:
        print(
            f"[dry-run] would serve {APP_IMPORT_STRING} on "
            f"http://{args.host}:{args.port} "
            f"(reload={args.reload}, log_level={args.log_level})"
        )
        return 0

    # Lazy import so the module is importable without uvicorn installed.
    import uvicorn

    print(f"Serving {APP_IMPORT_STRING} on http://{args.host}:{args.port}")
    uvicorn.run(
        APP_IMPORT_STRING,
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
