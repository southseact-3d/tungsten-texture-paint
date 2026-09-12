from __future__ import annotations

import sys

from stl_painter.logging_utils import configure_logging, log_file_path

# Dump Python tracebacks to a file on segfaults / fatal errors so native
# crashes (e.g. inside opengl32.dll) still leave a diagnosable trail in both
# source and frozen runs.
try:
    import faulthandler
    from pathlib import Path

    _fault_log = Path(str(log_file_path())).with_name("stl_texture_painter.fault.log")
    _fault_file = open(_fault_log, "w", encoding="utf-8")  # noqa: PTH123
    faulthandler.enable(file=_fault_file)
except Exception:
    pass

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
