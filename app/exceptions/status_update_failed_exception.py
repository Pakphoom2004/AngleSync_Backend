class StatusUpdateFailedException(Exception):

    def __init__(
        self,
        message="Unable to update user status. Please try again.",
    ):
        super().__init__(message)