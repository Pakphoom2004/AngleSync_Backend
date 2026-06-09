class ServiceException(Exception):

    def __init__(
        self,
        message="AI service failed to process the request.",
    ):
        self.message = message
        super().__init__(self.message)
