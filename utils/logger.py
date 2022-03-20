import logging, os

class CustomFormatter(logging.Formatter):

    def format(self, record):
        if hasattr(record, 'func_name_override'):
            record.funcName = record.func_name_override
            record.lineno = -1
        return super(CustomFormatter, self).format(record)

class MyLogger:

    INFO = logging.INFO
    DEBUG = logging.DEBUG
    ERROR = logging.ERROR
    WARNING = logging.WARNING
    CRITICAL = logging.CRITICAL
    NOTSET = logging.NOTSET

    def __init__(self, name, level=logging.INFO, with_stream=True):

        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        formatter = CustomFormatter("[%(asctime)s]; %(levelname)s; Function: %(funcName)s; Line: %(lineno)d => %(message)s")

        if "logs" not in os.listdir("./api"):
            os.mkdir("./api/logs")
        file_handler = logging.FileHandler(f"./api/logs/{name}.log")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)

        if with_stream:
            stream = logging.StreamHandler()
            stream.setLevel(level)
            stream.setFormatter(CustomFormatter("Line: %(lineno)d; Function: %(funcName)s => %(message)s"))
            self.logger.addHandler(stream)


    def __call__(self):
        return self.logger