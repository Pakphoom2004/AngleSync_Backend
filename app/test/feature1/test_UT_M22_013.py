from unittest.mock import patch

from PIL import Image

from app.services.graph_service import generate_risk_graph


RISK_SCORES = (10.2, 47.76, 85.4, 30.1, 55.6)
HIGHEST_RISK_INDEX = 2
FRAME_TIMES = None
FPS = 30


class TestGenerateRiskGraph:

    def test_returns_rendered_risk_graph_image(self):
        result = generate_risk_graph(
            RISK_SCORES,
            HIGHEST_RISK_INDEX,
            frame_times=FRAME_TIMES,
            fps=FPS
        )

        assert result is not None
        assert isinstance(result, Image.Image)
        assert result.format == "PNG"
        assert len(RISK_SCORES) == 5

    @patch("app.services.graph_service.plt.annotate")
    @patch("app.services.graph_service.plt.axvline")
    @patch("app.services.graph_service.plt.scatter")
    def test_highest_risk_frame_is_marked_on_graph(
        self,
        mock_scatter,
        mock_axvline,
        mock_annotate
    ):
        generate_risk_graph(
            RISK_SCORES,
            HIGHEST_RISK_INDEX,
            frame_times=FRAME_TIMES,
            fps=FPS
        )

        expected_time = round(HIGHEST_RISK_INDEX / FPS, 2)
        expected_score = RISK_SCORES[HIGHEST_RISK_INDEX]

        mock_scatter.assert_called_once()
        assert mock_scatter.call_args.args[0] == expected_time
        assert mock_scatter.call_args.args[1] == expected_score

        mock_axvline.assert_called_once()
        assert mock_axvline.call_args.args[0] == expected_time

        mock_annotate.assert_called_once()
        assert mock_annotate.call_args.args[0] == "Highest risk\n85.4%"