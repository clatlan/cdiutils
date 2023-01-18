

CXI_TO_XU_TRANSITIONS = {
        "x+": "y+",
        "x-": "y-",
        "y+": "z+",
        "y-": "z-",
        "z+": "x+",
        "z-": "x-",
}


class Geometry:
    """
    A class to handle the geomtry of the experiment setup.
    """
    def __init__(
            self,
            sample_circles: list[str],
            detector_circles: list[str],
            detector_vertical_orientation: str="y-",
            detector_horizontal_orientation: str="x+",
            beam_direction: list=None
    ) -> None:
        self.sample_circles = sample_circles
        self.detector_circles = detector_circles
        self.detector_vertical_orientation = detector_vertical_orientation
        self.detector_horizontal_orientation = detector_horizontal_orientation
        if beam_direction is None:
            self.beam_direction = [1, 0, 0]
        else:
            self.beam_direction = beam_direction

    @classmethod
    def from_name(cls, beamline_name: str) -> None:
        """Create a Geometry instance using a beamline name."""

        if beamline_name in ("ID01", "ID01SPEC", "ID01BLISS"):
            return cls(
                sample_circles=["x-", "y-"],
                detector_circles=["y+", "x-"],
                detector_vertical_orientation="y-",
                detector_horizontal_orientation="x+",
                beam_direction=[1, 0, 0]
            )
        else:
            raise NotImplementedError(
                f"The beamline_name {beamline_name} is not known"
            )

    def cxi_to_xu(self) -> None:
        """
        Convert the CXI circle axes to the xrayutilities coordinates system
        """
        self.sample_circles = [
            CXI_TO_XU_TRANSITIONS[v] for v in self.sample_circles
        ]
        self.detector_circles = [
            CXI_TO_XU_TRANSITIONS[v] for v in self.detector_circles
        ]
        self.detector_vertical_orientation = CXI_TO_XU_TRANSITIONS[
            self.detector_vertical_orientation
        ]
        self.detector_horizontal_orientation = CXI_TO_XU_TRANSITIONS[
            self.detector_horizontal_orientation
        ]
