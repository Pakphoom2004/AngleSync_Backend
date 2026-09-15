class SelfStatusUpdateNotAllowedException(Exception):

    def __init__(
        self,
        message="You cannot deactivate your own account.",
    ):
        self.message = message
        super().__init__(message)