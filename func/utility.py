import os, sys
from pathlib import Path
import logging
from datetime import datetime
from sys import platform

USERNAME = os.environ.get('USERNAME')

def loggergen(logfld = None):
    logfld = Path(logfld)
    logfld.mkdir(exist_ok=True, parents=True)
    logtime = datetime.now().strftime('%m%d%Y_%H%M')
    logformat = "%(levelname)s:%(name)s:%(asctime)s:%(message)s"

    handlers = []
    handlers.append(logging.StreamHandler(sys.stdout))
    
    if not logfld is None: 
        handlers.append(logging.FileHandler(filename=logfld.joinpath(f'log_{logtime}.log')))

    logging.basicConfig(
        format = logformat,
        handlers = handlers,
        encoding ='utf-8', 
        level = logging.INFO)
    
    logger = logging.getLogger()
    
    return logger