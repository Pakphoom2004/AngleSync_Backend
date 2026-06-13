import pytest

from app.services.video_validation import verify_file_type
from app.exceptions import ServiceException


class TestVerifyFileType:

    def test_mp4_does_not_raise(self):
        verify_file_type("exercise_vid.mp4")

    def test_mov_does_not_raise(self):
        verify_file_type("exercise_vid.mov")

    def test_unsupported_format_raises_service_exception(self):
        with pytest.raises(ServiceException) as exc_info:
            verify_file_type("exercise_vid.avi")

        assert "Invalid file format" in str(exc_info.value)
        assert "MP4 or MOV" in str(exc_info.value)