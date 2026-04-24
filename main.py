from __future__ import annotations

import sys

from stl_painter.logging_utils import configure_logging

configure_logging()

# Sub-commands that should be routed to the headless CLI (no GUI)
_CLI_SUBCOMMANDS = frozenset({
    "info",
    "paint-face",
    "paint-all",
    "flood-fill",
    "paint-group",
    "paint-region",
    "list-groups",
    "save",
    "export",
    "pipeline",
})

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in _CLI_SUBCOMMANDS:
        # Headless CLI path – no GUI or OpenGL needed
        from stl_painter.cli import main as cli_main
        cli_main(sys.argv[1:])
    else:
        # GUI path (default when invoked with no arguments)
        from stl_painter.self_test import build_parser, run_stl_self_test
        from stl_painter.ui import run
        args = build_parser().parse_args()
        if args.self_test_stl:
            run_stl_self_test(args.self_test_stl)
        else:
            run()
