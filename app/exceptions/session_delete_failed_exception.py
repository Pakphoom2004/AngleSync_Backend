class SessionDeleteFailedException(Exception):

    def __init__(
        self,
        message="Unable to delete this record. Please try again.",
    ):
        super().__init__(message)