class SessionNotFoundException(Exception):

    def __init__(
            self,
            message="No sessions found for the selected date.",
    ):
        super().__init__(message)
