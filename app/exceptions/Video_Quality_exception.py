class VideoQualityException(Exception):
    """Exception raised when video quality is too poor for pose detection."""

    def __init__(
            self,
            message="Pose detection failed. Please ensure the video is clear, "
    ):
        self.message = message
        super().__init__(self.message)