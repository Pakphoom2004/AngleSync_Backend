class KeypointNotDetectedException(Exception):

    def __init__(
        self,
        message="Keypoint not found. Please ensure that the person is visible."
    ):
        self.message = message
        super().__init__(self.message)
