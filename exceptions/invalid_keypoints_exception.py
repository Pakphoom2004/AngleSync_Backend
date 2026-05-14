class InvalidKeypointsException(Exception):

    def __init__(
        self,
        message="No body joints could be reliably identified for angle calculation."
    ):
        self.message = message
        super().__init__(self.message)
        