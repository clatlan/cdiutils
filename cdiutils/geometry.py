

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
            sample_circles: list = None,
            detector_circles: list = None,
            detector_vertical_orientation: str = "y-",
            detector_horizontal_orientation: str = "x+",
            beam_direction: list = None,
            name: str = None
    ) -> None:
        self.sample_circles = sample_circles
        self.detector_circles = detector_circles
        self.detector_vertical_orientation = detector_vertical_orientation
        self.detector_horizontal_orientation = detector_horizontal_orientation
        if beam_direction is None:
            self.beam_direction = [1, 0, 0]
        else:
            self.beam_direction = beam_direction

        self.name = name

        self.is_cxi = True

    @classmethod
    def from_setup(cls, beamline_setup: str) -> None:
        """Create a Geometry instance using a beamline name."""

        # Note that we use CXI convention here
        if beamline_setup.lower() in ("id01", "id01spec", "id01bliss"):
            return cls(
                sample_circles=["x-", "y-"],  # eta, phi
                detector_circles=["y-", "x-"],  # nu, delta
                detector_vertical_orientation="y-",
                detector_horizontal_orientation="x+",
                beam_direction=[1, 0, 0],
                name="ID01"
            )
        if "p10" in beamline_setup.lower():
            return cls(
                sample_circles=["x-", "y-"],  # om (or samth), phi
                detector_circles=["y+", "x-"],  # gam, del (or e2_t02)
                detector_vertical_orientation="y-",
                detector_horizontal_orientation="x+",
                beam_direction=[1, 0, 0],
                name="P10"
            )
        if beamline_setup.lower() == "sixs2022":
            return cls(
                sample_circles=["x-", "y+"],  # mu, omega
                detector_circles=["y+", "x-"],  # gamma, delta  NOT SURE OF THE COMMENT
                detector_vertical_orientation="y-",
                detector_horizontal_orientation="x+",
                beam_direction=[1, 0, 0],
                name="SIXS2022"
            )
        if beamline_setup.lower() == "nanomax":
            return cls(
                sample_circles=["x-", "y-"],  # gontheta, gonphi
                detector_circles=["y-", "x-"],  # gamma, delta
                detector_vertical_orientation="y+",
                detector_horizontal_orientation="x-",
                beam_direction=[1, 0, 0],
                name="NanoMAX"
            )
        if beamline_setup.lower() == "cristal":
            return cls(
                sample_circles=["x-", "y+"],  # omega, phi
                detector_circles=["y+", "x-"],  # gamma, delta  OK FOR omega/delta but not for the two others
                detector_vertical_orientation="y-",
                detector_horizontal_orientation="x+",
                beam_direction=[1, 0, 0],
                name="CRISTAL"
            )

        if beamline_setup.lower() == "id27":
            return cls(
                detector_vertical_orientation="y-",
                detector_horizontal_orientation="x-",
                beam_direction=[1, 0, 0],
                name="ID27"
            )
        raise NotImplementedError(
            f"The beamline_setup {beamline_setup} is not valid. Available:\n"
            "'ID01', 'ID01SPEC', 'ID01BLISS', 'P10', 'P10EH2', 'SIXS2022' "
            "and NanoMAX."
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

    def __repr__(self) -> str:
        return (
            f"{self.name} geometry:\n"
            f"{self.sample_circles = }\n"
            f"{self.detector_circles = }\n"
            f"{self.detector_vertical_orientation = }\n"
            f"{self.detector_horizontal_orientation = }\n"
            f"{self.beam_direction = }\n"
            f"{self.is_cxi = }\n"
        )
