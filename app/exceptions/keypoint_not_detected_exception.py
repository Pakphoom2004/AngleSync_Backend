class KeypointNotDetectedException(Exception):

    def __init__(
        self,
        message="Detection failed. Please ensure that the person is visible."
    ):
        self.message = message
        super().__init__(self.message)
