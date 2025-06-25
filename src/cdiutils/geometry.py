import numpy as np
import warnings

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
            sample_surface_normal: list = None,
            name: str = None,
            is_cxi: bool = True
    ) -> None:
        """
        Initialise the Geometry instance with the given parameters.

        Args:
            sample_circles (list, optional): list of sample circle
                orientations. Defaults to None.
            detector_circles (list, optional): list of detector circle
                orientations. Defaults to None.
            detector_axis0_orientation (str, optional): orientation of
                the detector axis 0. Defaults to "y-".
            detector_axis1_orientation (str, optional): orientation of
                the detector axis 1. Defaults to "x+".
            beam_direction (list, optional): direction of the beam.
                Defaults to None.
            sample_surface_normal (list, optional): normal vector of the
                sample surface. Defaults to None.
            name (str, optional): name of the geometry instance.
                Defaults to None.
            is_cxi (bool, optional): flag indicating if the geometry is
                in CXI format. Defaults to True.
        """

        self.sample_circles = sample_circles
        if self.sample_circles is None:
            self.sample_circles = ["x+", "y-"]

        self.detector_circles = detector_circles
        if self.detector_circles is None:
            self.detector_circles = ["y-", "x-"]

        self.detector_axis0_orientation = detector_axis0_orientation
        if self.detector_axis0_orientation is None:
            self.detector_axis0_orientation = "y-"

        self.detector_axis1_orientation = detector_axis1_orientation
        if self.detector_axis1_orientation is None:
            self.detector_axis1_orientation = "x+"

        self.beam_direction = beam_direction
        if self.beam_direction is None:
            self.beam_direction = [1, 0, 0]

        self.sample_surface_normal = sample_surface_normal
        if self.sample_surface_normal is None:
            # default normal pointing in the ycxi direction,
            # corresponding to a horizontal sample
            self.sample_surface_normal = [0, 1, 0]

        self.name = name

        self.is_cxi = is_cxi

    def to_dict(self) -> dict:
        """Return the attributes of the Geometry instance as a dictionary."""
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, data: dict) -> "Geometry":
        """
        Factory method to create a Geometry instance from a dictionary.
        """
        return cls(**data)

    @classmethod
    def from_setup(
            cls,
            beamline: str | None = None,
            beamline_setup: str | None = None,
            sample_orientation: str | None = None,
            sample_surface_normal: list | None = None
    ) -> "Geometry":
        """
        Factory method to create a Geometry instance using a beamline
        name.

        Args:
            beamline (str): the name of the beamline, supported are:
                "ID01", "P10", "SIXS", "NanoMAX", "CRISTAL", "ID27"
            beamline_setup (str | None, optional): DEPRACATED, use
                'beamline' instead. Will be removed in a future
                version.
            sample_orientation (str | None, optional): the orientation
                of the sample surface, either "horizontal", "h", "vertical" or
                "v". Defaults to None.
            sample_surface_normal (list | None, optional): the normal
                vector of the sample surface. This overrides the
                sample_orientation as the sample_surface_normal fully
                controls the sample orientation. Defaults to None.

        Raises:
            NotImplementedError: if the beamline is not supported.

        Returns:
            Geometry: a Geometry instance with the appropriate
            parameters set according to the beamline.
        """
        # handle backward compatibility
        if beamline is None:
            if beamline_setup is None:
                raise ValueError(
                    "The beamline name must be provided."
                )
            beamline = beamline_setup
            warnings.warn(
                "The 'beamline_setup' parameter is deprecated and will be "
                "removed in a future version. Use 'beamline' instead.",
                DeprecationWarning,
                stacklevel=2
            )

        geometry = None

        # Note that we use CXI convention here
        if beamline.lower() in ("id01", "id01spec", "id01bliss"):
            # by default, sample orientation is horizontal, pointing up
            geometry = cls(
                sample_circles=["x-", "y-"],  # eta, phi
                detector_circles=["y-", "x-"],  # nu, delta
                detector_axis0_orientation="y-",
                detector_axis1_orientation="x+",
                beam_direction=[1, 0, 0],
                surface_normal_direction=[0, 1, 0],  # default sample facing up
                name="ID01"
            )
            # default orientation for ID01 when sample is vertical
            if sample_orientation.lower() in ("vertical", "v"):
                geometry.sample_surface_normal = [0, 0, -1]

        if "p10" in beamline.lower():
            geometry = cls(
                sample_circles=["x-", "y-"],  # om (or samth), phi
                detector_circles=["y+", "x-"],  # gam, del (or e2_t02)
                detector_axis0_orientation="y-",
                detector_axis1_orientation="x-",
                beam_direction=[1, 0, 0],
                surface_normal_direction=[0, 1, 0],  # default sample facing up
                name="P10"
            )
        if "sixs" in beamline.lower():
            geometry = cls(
                sample_circles=["x-", "y+"],  # mu, omega
                detector_circles=["y+", "x-"],  # gamma, delta  NOT SURE
                detector_axis0_orientation=(
                    "x-" if "2022" in beamline.lower() else "y-"
                ),
                detector_axis1_orientation=(
                    "y-" if "2022" in beamline.lower() else "x+"
                ),
                beam_direction=[1, 0, 0],
                surface_normal_direction=[0, 1, 0],  # default sample facing up
                name="SIXS"
            )
        if beamline.lower() == "nanomax":
            geometry = cls(
                sample_circles=["x-", "y-"],  # gontheta, gonphi
                detector_circles=["y-", "x-"],  # gamma, delta
                detector_axis0_orientation="y-",
                detector_axis1_orientation="x-",
                beam_direction=[1, 0, 0],
                surface_normal_direction=[0, 1, 0],  # default sample facing up
                name="NanoMAX"
            )
        if beamline.lower() == "cristal":
            # OK FOR omega/delta but not for the two others
            geometry = cls(
                sample_circles=["x-", "y+"],  # omega, phi
                detector_circles=["y+", "x-"],  # gamma, delta
                detector_axis0_orientation="y-",
                detector_axis1_orientation="x+",
                beam_direction=[1, 0, 0],
                surface_normal_direction=[0, 1, 0],  # default sample facing up
                name="CRISTAL"
            )

        if beamline.lower() == "id27":
            geometry = cls(
                sample_circles=["x-", "y-"],  # In plane rotation only
                detector_circles=["y-", "x-"],  # no circle, values dummy
                detector_axis0_orientation="y-",
                detector_axis1_orientation="x-",
                beam_direction=[1, 0, 0],
                surface_normal_direction=[0, 1, 0],  # default sample facing up
                name="ID27"
            )
        if geometry is None:
            raise NotImplementedError(
                f"The beamline name {beamline} is not valid. Available:\n"
                "'ID01', 'ID01SPEC', 'ID27', 'P10', 'P10EH2', 'SIXS2022' "
                "and NanoMAX."
            )

        # if the sample orientation is provided, override any default
        # value, this fully controls the sample orientation.
        if sample_surface_normal is not None:
            geometry.sample_surface_normal = sample_surface_normal

        return geometry

    @property
    def sample_orientation(self) -> str | None:
        """
        Returns the sample mounting orientation: 'horizontal' or 'vertical'.

        The sample is considered:
        - 'horizontal' when its surface normal is along the -Ycxi or
        Ycxi, i.e. pointing up/down.
        - 'vertical' when its surface normal is along the -Xcxi or Xcxi.
        """
        normal = np.array(self.sample_surface_normal)
        normal = normal / np.linalg.norm(normal)

        # set the index of interest, the index along which the normal
        # should be pointing up if the geometry type is "horizontal".
        # In CXI or XU conventions, this is the y-axis or z-axis, respectively.
        index_of_interest = 1  # y-axis in CXI convention
        if not self.is_cxi:
            index_of_interest = 2  # z-axis in XU convention

        # checking if normal is primarily along y-axis (CXI convention)
        if np.argmax(np.abs(normal)) == index_of_interest:
            return "horizontal"
        else:
            return "vertical"

    @sample_orientation.setter
    def sample_orientation(self, orientation: str) -> None:
        """
        Set the sample orientation by updating the sample_surface_normal.

        Args:
            orientation (str): the orientation of the sample surface,
                either 'horizontal', 'h' (normal along y-axis in CXI
                convention), 'vertical' or 'v' (normal along x-axis in
                CXI convention).

        Raises:
            ValueError: if the orientation is not
            'horizontal', 'h', 'vertical' or 'v'.
        """
        orientation = orientation.lower()
        if orientation in ("horizontal", "h"):
            self.sample_surface_normal = [0, 1, 0]  # y-axis in CXI
        elif orientation in ("vertical", "v"):
            self.sample_surface_normal = [0, 0, 1]  # x-axis in CXI
        else:
            raise ValueError(
                "Orientation must be either 'horizontal' or 'vertical'"
                " (or their abbreviations 'h' or 'v')."
            )

    @property
    def surface_normal_direction(self) -> str:
        """
        Returns the direction of the sample surface normal: 'up',
        'down', 'outboard', or 'inboard'.

        For horizontal samples:
        - 'up': normal points along +y in CXI (+z in XU)
        - 'down': normal points along -y in CXI (-z in XU)

        For vertical samples:
        - 'outboard': normal points along +x in CXI (+y in XU)
        - 'inboard': normal points along -x in CXI (-y in XU)
        """
        normal = np.array(self.sample_surface_normal)
        normal = normal / np.linalg.norm(normal)

        # determine orientation first
        orientation = self.sample_orientation

        # then determine direction based on the sign of the
        # corresponding component
        if orientation == "horizontal":
            index = 1 if self.is_cxi else 2  # y-axis in CXI, z-axis in XU
            return "up" if normal[index] > 0 else "down"
        else:  # vertical
            index = 2 if self.is_cxi else 1  # x-axis in CXI, y-axis in XU
            return "outboard" if normal[index] > 0 else "inboard"

    def cxi_to_xu(self) -> None:
        """
        Convert the CXI circle axes to the xrayutilities coordinates
        system. Modifies this Geometry instance in place.
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

        self.sample_surface_normal = self.swap_convention(
            self.sample_surface_normal
        )
        # The following is not necessary because we always consider
        # the beam along Zcxi axis, which is physically-speaking
        # equivalent to the beam along Xxu axis in XU convention which
        # are both encoded as [1, 0, 0] in either convention. See
        # Geometry.swap_convention() for more details.
        # Here,  we keep it for consistency.
        self.beam_direction = self.swap_convention(self.beam_direction)

        self.is_cxi = False

    @staticmethod
    def swap_convention(
            data: np.ndarray | list | tuple
    ) -> np.ndarray | list | tuple:
        """
        Swap the CXI and XU conventions for the given data.
        This method effectively swaps the last two axes of the input
        data.

        This operation is used to convert between CXI and XU conventions:
        - In CXI convention, arrays are stored in order (Zcxi, Ycxi, Xcxi)
        - In XU convention, arrays are stored in order (Xxu, Yxu, Zxu)
        And, physically, we have:
            - Zcxi = Xxu, pointing along the beam direction, away from
            the light source.
            - Ycxi = Zxu, vertical direction, pointing up.
            - Xcxi = Yxu, outboard direction in the Synchrotron frame,
            vertical plane, perpendicular to the beam direction.
        Both conventions are right-handed.
        What comes out of this description is that no swapping is
        needed for the beam direction, which is always along the first
        axis (axis 0) in both conventions.

        Args:
            data (np.ndarray | list | tuple): the input data to swap.

        Returns:
            np.ndarray: The data with swapped conventions.
        """
        if isinstance(data, (tuple, list)):
            axis0, axis1, axis2 = data
            return type(data)((axis0, axis2, axis1))
        if isinstance(data, np.ndarray):
            if data.shape == (3, ):
                return np.array([data[0], data[2], data[1]])
            return np.swapaxes(data, axis1=1, axis2=2)
        else:
            raise TypeError(
                "data should be a 3D np.ndarray, a list of 3 values, a tuple "
                "of 3 values or a np.ndarray of 3 values."
            )

        return data

    def __repr__(self) -> str:
        return (
            f"{self.name} geometry:\n"
            f"{self.sample_circles=}\n"
            f"{self.detector_circles=}\n"
            f"{self.detector_axis0_orientation=}\n"
            f"{self.detector_axis1_orientation=}\n"
            f"{self.beam_direction=}\n"
            f"{self.sample_surface_normal=}\n"
            f"{self.sample_orientation=}\n"
            f"{self.is_cxi=}\n"
        )
