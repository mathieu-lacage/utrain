import os
import logging
import sys

FORMAT = "%(asctime)s:%(levelname)s:%(module)s.%(funcName)s:%(message)s"
DATEFMT = "%H:%M:%S"


def setup(debug: int, log_filename: str) -> None:
    log_level = os.getenv("UTRAIN_LOG_LEVEL")
    match log_level:
        case "DEBUG":
            debug = 3
        case "INFO":
            debug = 2
        case "WARNING":
            debug = 1
        case None:
            pass
        case _:
            try:
                debug = int(log_level)
            except ValueError:
                pass
    if debug == 0:
        return
    match debug:
        case 1:
            level = logging.WARNING
        case 2:
            level = logging.INFO
        case _:
            level = logging.DEBUG
    try:
        f = open(log_filename, "a", buffering=1)
    except Exception:
        f = sys.stdout
    logging.basicConfig(stream=f, level=level, format=FORMAT, datefmt=DATEFMT, force=True)
