class ServiceException(Exception):

    def __init__(
        self,
        message="Gemini API failed to process the request.",
    ):
        self.message = message
        super().__init__(self.message)
