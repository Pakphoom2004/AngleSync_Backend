class UnauthorizedAccessException(Exception):

    def __init__(
        self,
        message="Unable to process your request. Please try again.",
    ):
        super().__init__(message)