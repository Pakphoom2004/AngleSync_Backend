class UnauthorizedAccessException(Exception):

    def __init__(
        self,
        message="You do not have permission to view this page.",
    ):
        super().__init__(message)