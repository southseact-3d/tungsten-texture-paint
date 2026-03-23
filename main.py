import logging

from stl_painter.self_test import build_parser, run_stl_self_test
from stl_painter.ui import run


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


if __name__ == "__main__":
    args = build_parser().parse_args()
    if args.self_test_stl:
        run_stl_self_test(args.self_test_stl)
    else:
        run()
