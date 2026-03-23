from stl_painter.self_test import build_parser, run_stl_self_test
from stl_painter.logging_utils import configure_logging
from stl_painter.ui import run

configure_logging()


if __name__ == "__main__":
    args = build_parser().parse_args()
    if args.self_test_stl:
        run_stl_self_test(args.self_test_stl)
    else:
        run()
