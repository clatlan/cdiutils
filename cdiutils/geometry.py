

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
            detector_axis0_orientation: str = "y-",
            detector_axis1_orientation: str = "x+",
            beam_direction: list = None,
            name: str = None,
            is_cxi: bool = True
    ) -> None:
        self.sample_circles = sample_circles
        self.detector_circles = detector_circles
        self.detector_axis0_orientation = detector_axis0_orientation
        self.detector_axis1_orientation = detector_axis1_orientation
        if beam_direction is None:
            self.beam_direction = [1, 0, 0]
        else:
            self.beam_direction = beam_direction

        self.name = name

        self.is_cxi = is_cxi

    def to_dict(self) -> dict:
        """
        Return the attributes of the Geometry instance as a dictionary.
        """
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, data) -> "Geometry":
        """Create a Geometry instance from a dictionary."""
        return cls(**data)

    @classmethod
    def from_setup(cls, beamline_setup: str) -> None:
        """Create a Geometry instance using a beamline name."""

        # Note that we use CXI convention here
        if beamline_setup.lower() in ("id01", "id01spec", "id01bliss"):
            return cls(
                sample_circles=["x-", "y-"],  # eta, phi
                detector_circles=["y-", "x-"],  # nu, delta
                detector_axis0_orientation="y-",
                detector_axis1_orientation="x+",
                beam_direction=[1, 0, 0],
                name="ID01"
            )
        if "p10" in beamline_setup.lower():
            return cls(
                sample_circles=["x-", "y-"],  # om (or samth), phi
                detector_circles=["y+", "x-"],  # gam, del (or e2_t02)
                detector_axis0_orientation="y-",
                detector_axis1_orientation="x+",
                beam_direction=[1, 0, 0],
                name="P10"
            )
        if "sixs" in beamline_setup.lower():
            return cls(
                sample_circles=["x-", "y+"],  # mu, omega
                detector_circles=["y+", "x-"],  # gamma, delta  NOT SURE OF THE COMMENT
                detector_axis0_orientation="x-" if "2022" in beamline_setup.lower() else "y-",
                detector_axis1_orientation="y-" if "2022" in beamline_setup.lower() else "x+",
                beam_direction=[1, 0, 0],
                name="SIXS"
            )
        if beamline_setup.lower() == "nanomax":
            return cls(
                sample_circles=["x-", "y-"],  # gontheta, gonphi
                detector_circles=["y-", "x-"],  # gamma, delta
                detector_axis0_orientation="y-",
                detector_axis1_orientation="x-",
                beam_direction=[1, 0, 0],
                name="NanoMAX"
            )
        if beamline_setup.lower() == "cristal":
            return cls(
                sample_circles=["x-", "y+"],  # omega, phi
                detector_circles=["y+", "x-"],  # gamma, delta  OK FOR omega/delta but not for the two others
                detector_axis0_orientation="y-",
                detector_axis1_orientation="x+",
                beam_direction=[1, 0, 0],
                name="CRISTAL"
            )

        if beamline_setup.lower() == "id27":
            return cls(
                sample_circles=["x-", "y-"],  # In plane rotation only
                detector_circles=["y-", "x-"],  # There is no circle, these values are dummy
                detector_axis0_orientation="y-",
                detector_axis1_orientation="x-",
                beam_direction=[1, 0, 0],
                name="ID27"
            )
        raise NotImplementedError(
            f"The beamline_setup {beamline_setup} is not valid. Available:\n"
            "'ID01', 'ID01SPEC', 'ID27', 'P10', 'P10EH2', 'SIXS2022' "
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
        self.detector_axis0_orientation = CXI_TO_XU_TRANSITIONS[
            self.detector_axis0_orientation
        ]
        self.detector_axis1_orientation = CXI_TO_XU_TRANSITIONS[
            self.detector_axis1_orientation
        ]
        self.is_cxi = False

    def __repr__(self) -> str:
        return (
            f"{self.name} geometry:\n"
            f"{self.sample_circles = }\n"
            f"{self.detector_circles = }\n"
            f"{self.detector_axis0_orientation = }\n"
            f"{self.detector_axis1_orientation = }\n"
            f"{self.beam_direction = }\n"
            f"{self.is_cxi = }\n"
        )
