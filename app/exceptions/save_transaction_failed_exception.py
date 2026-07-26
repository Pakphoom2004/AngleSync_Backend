class SaveTransactionFailedException(Exception):

    def __init__(
            self,
            message="Unable to save result. Please try again.",
    ):
        super().__init__(message)