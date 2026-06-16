import pytest
import numpy as np

from app.services.overlay_service import render_skeleton_overlay


KP02 = (
    (0.4680759608745575,  0.13428252935409546),
    (0.0,                 0.0),
    (0.4652771055698395,  0.11990076303482056),
    (0.0,                 0.0),
    (0.43922296166419983, 0.11978987604379654),
    (0.42056581377983093, 0.2169395238161087),
    (0.4168631434440613,  0.22159205377101898),
    (0.41852816939353943, 0.3678935170173645),
    (0.4093763530254364,  0.3831363916397095),
    (0.0,                 0.0),
    (0.44086283445358276, 0.5217946171760559),
    (0.42167890071868896, 0.4936578869819641),
    (0.3994448781013489,  0.49802303314208984),
    (0.47539588809013367, 0.6768141388893127),
    (0.36448314785957336, 0.6863085627555847),
    (0.5105754137039185,  0.8785256743431091),
    (0.2795264422893524,  0.8489685654640198),
)


class TestRenderSkeletonOverlay:

    def test_returns_image_not_none_with_same_shape(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        result = render_skeleton_overlay(frame, KP02)

        assert result is not None
        assert result.shape == (480, 640, 3)

    def test_draws_green_lines_and_red_circles(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        result = render_skeleton_overlay(frame, KP02)

        green_color = [0, 255, 0]
        has_green = np.any(np.all(result == green_color, axis=2))
        assert has_green, "No green skeleton line (0, 255, 0) found in frame"

        red_color = [0, 0, 255]
        has_red = np.any(np.all(result == red_color, axis=2))
        assert has_red, "No red joint circle (0, 0, 255) found in frame"