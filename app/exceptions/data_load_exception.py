class DataLoadException(Exception):

    def __init__(
        self,
        message="Unable to load data. Please try again.",
    ):
        self.message = message
        super().__init__(message)