class HistoryException(Exception):

    def __init__(
        self,
        message="Couldn't load your history right now. Please try again.",
    ):
        super().__init__(message)