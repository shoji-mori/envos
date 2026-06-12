import logging
import pathlib

loggers = {}
logger_level = logging.INFO
stream_level = logging.INFO
file_level = logging.INFO
debug_logger = 0


def set_logger(name, htype="stream", filepath=None):
    """
    This function will be called when envos is imported.
    By using this, one can add logger with its name manually.
    """

    global loggers

    if debug_logger:
        print("set logger: name = ", name)

    ## Return logger if it already exists in loggers dictionary
    if name in loggers:
        return loggers[name]

    ## Create logger if it does not exist in loggers dictionary
    logger = logging.getLogger(name) # Create logger
    logger.propagate = False # Prevent logging to propagate to the root logger
    loggers[name] = logger # Add logger to loggers dictionary

    return logger


def get_level(level_name):
    """ Return logging level from level name. """
    return logging.getLevelName(level_name.upper())


def add_stream_hdlr(logger, level=None):
    """ Add stream handler to logger. """
    global stream_level, StandardFormatter
    level = level if level is not None else stream_level
    hdlr = logging.StreamHandler()
    hdlr.setFormatter(StandardFormatter())
    hdlr.setLevel(level)
    logger.addHandler(hdlr)
    if debug_logger:
        print("logger after _add_stream_handler", logger)


def add_file_hdlr(logger, path, level=None, write_mode="w"):
    global file_level, StandardFormatter
    level = level if level is not None else file_level
    # from . import gpath
    # gpath.make_dirs(run=gpath.run_dir)
    # os.makedirs(os.path.dirname(gpath.logfile), exist_ok=True)

    _logfile = pathlib.Path(str(path))
    _logfile.parent.mkdir(exist_ok=True, parents=True)
    # gpath.logfile.parent.mkdir(exist_ok=True, parents=True)

    hdlr = logging.FileHandler(_logfile, write_mode, "utf-8")
    hdlr.setFormatter(StandardFormatter())
    hdlr.setLevel(level)
    logger.addHandler(hdlr)

    if debug_logger:
        print("logger after _add_file_handler", logger)


def change_rundir(new_rundir):
    global loggers
    new_rundir = pathlib.Path(new_rundir)
    for lg in loggers.values():
        for hdlr in list(lg.handlers):
            if type(hdlr) == logging.FileHandler:
                old_level = hdlr.level
                old_name = pathlib.Path(hdlr.baseFilename).name
                new_path = new_rundir / old_name
                hdlr.close()
                lg.removeHandler(hdlr)
                add_file_hdlr(lg, new_path, level=old_level, write_mode="a")


##

## What user wants to do:
##  - set logging level
##  - set logger only for result output
##  - controll log stdout/file/etc


## Set a file handler of all loggers
##  - turn on/off etire logs
##  - newly set logfile
#
# user_logger = set_logger(name="", type="file", filepath="**", )
# user_logger.info(" ***** ")
def update_logfile(name="envos", filename=None, filepath=None, level=None):
    unset_logfile(name)
    set_logfile(name, filename=filename, filepath=filepath, level=level)


def set_logfile(name="envos", filename=None, filepath=None, level=None):
    """
    add a file handler to all loggers in envos
    filename: located under run directory
    filepath: specify exact location of created file
    """
    global file_level, logger
    level = level if level is not None else file_level
    from . import gpath

    filepath = filepath if filepath is not None else gpath.logfile
    add_file_hdlr(loggers[name], filepath, level)


def unset_logfile(name):
    """
    Note that this function destroys all file handlers included in the logger.
    """
    for hdlr in list(loggers[name].handlers):
        if type(hdlr) == logging.FileHandler:
            hdlr.close()
            loggers[name].removeHandler(hdlr)


## Controll Level of Handlers
def set_level(name, level_name, target="all"):
    global loggers, stream_level, file_level

    level = get_level(level_name)
    htype = {
        "stream": logging.StreamHandler,
        "file": logging.FileHandler,
        "all": None,
    }[target]

    for hdlr in loggers[name].handlers:
        if (type(hdlr) == htype) or (target == "all"):
            if debug_logger:
                print(f"Now setting level to be {level_name}:{level} for {type(hdlr)}")
            hdlr.setLevel(level)

    if debug_logger:
        show_loggers()


def show_loggers():
    global loggers
    print("All loggers:")
    for n, logger in loggers.items():
        print("    ", n, logger, ":")
        for hdlr in logger.handlers:
            print("        ", hdlr)


class color:
    BLACK = "\033[30m"
    GRAY = "\033[96m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    PURPLE = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    RETURN = "\033[07m"
    ACCENT = "\033[01m"
    FLASH = "\033[05m"
    RED_FLASH = "\033[05;41m"
    END = "\033[0m"


class StandardFormatter(logging.Formatter):
    def __init__(self):
        super().__init__(fmt="[%(filename)s] %(levelname)s: %(message)s")

        self._fmtdict = {
            logging.DEBUG: "(Debug) %(message)s",
            logging.INFO: "%(message)s",
            logging.WARN: "Warning! -- %(message)s",
            logging.ERROR: "!!ERROR!! %(message)s",
        }

    def format(self, record):
        format_orig = self._style._fmt
        if record.levelno in self._fmtdict.keys():
            self._style._fmt = self._fmtdict[record.levelno]
        result = logging.Formatter.format(self, record)
        self._style._fmt = format_orig
        return result


class DebugFormatter(logging.Formatter):
    def __init__(self):
        super().__init__(fmt="[%(filename)s] %(levelname)s: %(message)s")

        self._fmtdict = {
            logging.DEBUG: "%(asctime)s [%(filename)s] (Debug) %(message)s",
            logging.INFO: "%(asctime)s [%(filename)s] %(message)s",
            logging.WARN: "%(asctime)s [%(filename)s] Warning! -- %(message)s",
            logging.ERROR: "%(asctime)s [%(filename)s] !!ERROR!! -- %(message)s",
        }

    def format(self, record):
        format_orig = self._style._fmt
        if record.levelno in self._fmtdict.keys():
            self._style._fmt = self._fmtdict[record.levelno]
        result = logging.Formatter.format(self, record)
        self._style._fmt = format_orig
        return result


#############################################
# Set envos logger
#############################################

# Create a logger
logger = logging.getLogger("envos")
logger.propagate = False
logger.setLevel(logger_level)

# Add a stream handler to the logger
add_stream_hdlr(logger)

# Enroll the logger to the logger dictionary
loggers = {"envos": logger}
