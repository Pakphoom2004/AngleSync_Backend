class ServiceException(Exception):

    def __init__(
        self,
        message="if the Gemini API fails to process the request.",
    ):
        self.message = message
        super().__init__(self.message)