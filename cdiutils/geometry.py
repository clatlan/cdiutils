

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
    A class to handle the geometry of the experiment setup.
    The CXI convention is used here.
    """
    def __init__(
            self,
            sample_circles: list,
            detector_circles: list,
            detector_vertical_orientation: str = "y-",
            detector_horizontal_orientation: str = "x+",
            beam_direction: list = None
    ) -> None:
        self.sample_circles = sample_circles
        self.detector_circles = detector_circles
        self.detector_vertical_orientation = detector_vertical_orientation
        self.detector_horizontal_orientation = detector_horizontal_orientation
        if beam_direction is None:
            self.beam_direction = [1, 0, 0]
        else:
            self.beam_direction = beam_direction

        self.is_cxi = True

    @classmethod
    def from_setup(cls, beamline_setup: str) -> None:
        """Create a Geometry instance using a beamline name."""

        # Note that we use CXI convention here
        if beamline_setup in ("ID01", "ID01SPEC", "ID01BLISS"):
            return cls(
                sample_circles=["x-", "y-"],  # eta, phi
                detector_circles=["y-", "x-"],  # delta, nu
                detector_vertical_orientation="y-",
                detector_horizontal_orientation="x+",
                beam_direction=[1, 0, 0]
            )
        if beamline_setup == "P10":
            return cls(
                sample_circles=["x-", "y-"],  # om, phi
                detector_circles=["y+", "x-"],  # del, gam
                detector_vertical_orientation="y-",
                detector_horizontal_orientation="x+",
                beam_direction=[1, 0, 0]
            )
        if beamline_setup == "SIXS2022":
            return cls(
                sample_circles=["x-", "y+"],  # mu, omega
                detector_circles=["y+", "x-"],  # gamma, delta
                detector_vertical_orientation="y-",
                detector_horizontal_orientation="x+",
                beam_direction=[1, 0, 0]
            )
        raise NotImplementedError(
            f"The beamline_setup {beamline_setup} is not valid. Available:\n"
            "'ID01', 'ID01SPEC', 'ID01BLISS', 'P10' , 'SIXS2022'"
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
        self.is_cxi = False
