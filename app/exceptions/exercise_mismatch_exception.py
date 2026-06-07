class ExerciseMismatchException(Exception):

    def __init__(
        self,
        message="Uploaded video does not match the selected reference exercise.",
        similarity_score=0.0,
        details=None
    ):
        self.message = message
        self.similarity_score = similarity_score
        self.details = details or {}
        super().__init__(self.message)
