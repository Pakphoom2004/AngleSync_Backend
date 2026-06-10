class PoseModuleNotFoundError(Exception):
    """Exception raised when pose estimation libraries are missing."""

    def __init__(self, message="Required pose module is not installed."):
        self.message = message
        super().__init__(self.message)