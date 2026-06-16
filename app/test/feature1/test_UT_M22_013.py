import pytest
from PIL import Image

from app.services.graph_service import generate_risk_graph


RISK_SCORES          = (10.2, 47.76, 85.4, 30.1, 55.6)
HIGHEST_RISK_INDEX   = 2


class TestGenerateRiskGraph:

    def test_returns_image_object_that_is_not_none(self):
        result = generate_risk_graph(RISK_SCORES, HIGHEST_RISK_INDEX)

        assert result is not None
        assert isinstance(result, Image.Image)

    def test_highest_risk_score_at_correct_index(self):
        highest_risk_score = RISK_SCORES[HIGHEST_RISK_INDEX]

        assert highest_risk_score == 85.4