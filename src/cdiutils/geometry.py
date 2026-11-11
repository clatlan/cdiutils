import warnings

import numpy as np

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
    Beamline diffractometer geometry configuration.

    Defines coordinate system conventions, motor circle orientations,
    and detector pixel directions for BCDI experiments. Critical for
    accurate coordinate transformations between detector, laboratory,
    and reciprocal space frames.

    The CXI convention is used by default:
        - Xcxi: outboard direction (perpendicular to beam)
        - Ycxi: vertical (pointing up)
        - Zcxi: along beam (away from source)
        - data storage follows (Zcxi, Ycxi, Xcxi) ordering.

    The XU (xrayutilities) convention can also be used:
        - Xxu: along beam (away from source)
        - Yxu: outboard direction (perpendicular to beam)
        - Zxu: vertical (pointing up)
        - data storage follows (Xxu, Yxu, Zxu) ordering.

    Attributes:
        sample_circles: Sample rotation axes as list of two CXI
            directions (e.g., ``["x-", "y-"]`` for eta, phi at ID01).
        detector_circles: Detector rotation axes as list of two CXI
            directions.
        detector_axis0_orientation: CXI direction for detector's
            first pixel axis (rows).
        detector_axis1_orientation: CXI direction for detector's
            second pixel axis (columns).
        beam_direction: Beam propagation direction as 3-element list
            in CXI frame. Always ``[1, 0, 0]`` (along Zcxi).
        sample_surface_normal: Normal vector to sample surface.
            ``[0, 1, 0]`` for horizontal mounting (surface up),
            ``[0, 0, -1]`` for vertical (ID01 style).
        name: Beamline identifier (e.g., "ID01", "P10").
        is_cxi: If True, geometry uses CXI convention. If False,
            uses xrayutilities (XU) convention.

    See Also:
        :class:`~cdiutils.converter.SpaceConverter`: Uses Geometry
        for coordinate transformations.
        :doc:`/user_guide/coordinate_systems`: Detailed explanation
        of CXI/XU conventions.
    """

    def __init__(
        self,
        sample_circles: list | None = None,
        detector_circles: list | None = None,
        detector_axis0_orientation: str | None = None,
        detector_axis1_orientation: str | None = None,
        beam_direction: list | None = None,
        sample_surface_normal: list | None = None,
        name: str | None = None,
        is_cxi: bool = True,
    ) -> None:
        """
        Initialise geometry configuration.

        Args:
            sample_circles: List of sample rotation axes as CXI
                directions. Defaults to ``["x-", "y-"]``.
            detector_circles: List of detector rotation axes as CXI
                directions. Defaults to ``["y-", "x-"]``.
            detector_axis0_orientation: CXI direction for detector
                pixel axis 0. Defaults to ``"y-"``.
            detector_axis1_orientation: CXI direction for detector
                pixel axis 1. Defaults to ``"x+"``.
            beam_direction: Beam propagation vector in CXI frame.
                Defaults to ``[1, 0, 0]``.
            sample_surface_normal: Normal vector to sample surface.
                Defaults to ``[0, 1, 0]`` (horizontal mounting).
            name: Beamline name. Defaults to None.
            is_cxi: If True, uses CXI convention. If False, uses XU
                convention. Defaults to True.
        """

        self.sample_circles = sample_circles
        if self.sample_circles is None:
            self.sample_circles = ["x-", "y-"]

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
        """
        Serialise geometry configuration to dictionary.

        Returns:
            Dictionary containing all geometry attributes.
        """
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, data: dict) -> "Geometry":
        """
        Create geometry instance from dictionary.

        Args:
            data: Dictionary with geometry parameters.

        Returns:
            New Geometry instance.
        """
        return cls(**data)

    @classmethod
    def from_setup(
        cls,
        beamline: str | None = None,
        beamline_setup: str | None = None,
        sample_orientation: str | None = None,
        sample_surface_normal: list | None = None,
    ) -> "Geometry":
        """
        Create geometry from beamline name.

        Factory method providing pre-configured geometries for
        supported beamlines. Automatically sets correct motor
        orientations and detector pixel directions.

        Args:
            beamline: Beamline identifier (case-insensitive).
                Supported values:

                - ``"ID01"``, ``"ID01SPEC"``, ``"ID01BLISS"``: ESRF
                  ID01
                - ``"P10"``, ``"P10EH2"``: PETRA III P10
                - ``"SIXS2019"``, ``"SIXS2022"``: SOLEIL SIXS (specify
                  year)
                - ``"NanoMAX"``: MAX IV NanoMAX
                - ``"CRISTAL"``: SOLEIL CRISTAL
                - ``"ID27"``: ESRF ID27

            beamline_setup: Deprecated. Use ``beamline`` instead.
            sample_orientation: Sample mounting style:

                - ``"horizontal"`` or ``"h"``: surface normal
                  pointing up (default)
                - ``"vertical"`` or ``"v"``: surface normal along
                  beam (ID01 style)

            sample_surface_normal: Explicit surface normal vector.
                Overrides ``sample_orientation`` if provided.

        Returns:
            Configured Geometry instance for the beamline.

        Raises:
            ValueError: If ``beamline`` not provided.
            NotImplementedError: If beamline not supported.

        Examples:
            >>> geometry = Geometry.from_setup(beamline="id01")
            >>> geometry.sample_orientation
            'horizontal'

            >>> geometry = Geometry.from_setup(
            ...     beamline="id01",
            ...     sample_orientation="vertical"
            ... )
        """
        # handle backward compatibility
        if beamline is None:
            if beamline_setup is None:
                raise ValueError("The beamline name must be provided.")
            beamline = beamline_setup
            warnings.warn(
                "The 'beamline_setup' parameter is deprecated and will be "
                "removed in a future version. Use 'beamline' instead.",
                DeprecationWarning,
                stacklevel=2,
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
                sample_surface_normal=[0, 1, 0],  # default sample facing up
                name="ID01",
            )
            # default orientation for ID01 when sample is vertical
            if (
                sample_orientation is not None
                and sample_orientation.lower() in ("vertical", "v")
            ):
                geometry.sample_surface_normal = [0, 0, -1]

        if "p10" in beamline.lower():
            geometry = cls(
                sample_circles=["x-", "y-"],  # om (or samth), phi
                detector_circles=["y+", "x-"],  # gam, del (or e2_t02)
                detector_axis0_orientation="y-",
                detector_axis1_orientation="x-",
                beam_direction=[1, 0, 0],
                sample_surface_normal=[0, 1, 0],  # default sample facing up
                name="P10",
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
                sample_surface_normal=[0, 1, 0],  # default sample facing up
                name="SIXS",
            )
        if beamline.lower() == "nanomax":
            geometry = cls(
                sample_circles=["x-", "y-"],  # gontheta, gonphi
                detector_circles=["y-", "x-"],  # gamma, delta
                detector_axis0_orientation="y-",
                detector_axis1_orientation="x-",
                beam_direction=[1, 0, 0],
                sample_surface_normal=[0, 1, 0],  # default sample facing up
                name="NanoMAX",
            )
        if beamline.lower() == "cristal":
            # OK FOR omega/delta but not for the two others
            geometry = cls(
                sample_circles=["x-", "y+"],  # omega, phi
                detector_circles=["y+", "x-"],  # gamma, delta
                detector_axis0_orientation="y-",
                detector_axis1_orientation="x+",
                beam_direction=[1, 0, 0],
                sample_surface_normal=[0, 1, 0],  # default sample facing up
                name="CRISTAL",
            )

        if beamline.lower() == "id27":
            geometry = cls(
                sample_circles=["x-", "y-"],  # In plane rotation only
                detector_circles=["y-", "x-"],  # no circle, values dummy
                detector_axis0_orientation="y-",
                detector_axis1_orientation="x-",
                beam_direction=[1, 0, 0],
                sample_surface_normal=[0, 1, 0],  # default sample facing up
                name="ID27",
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

    def cxi_to_xu(self) -> None:
        """
        Convert CXI convention to XU convention in-place.

        Transforms all coordinate system attributes (sample circles,
        detector circles, orientations) from CXI to xrayutilities
        convention. Sets ``is_cxi`` to False after conversion.

        Note:
            This is automatically called by SpaceConverter during
            initialisation if needed. Users rarely call this directly.
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
        data: np.ndarray | list | tuple,
    ) -> np.ndarray | list | tuple:
        """
        Swap CXI and XU coordinate conventions.

        Swaps the last two axes to convert between conventions:

        - CXI ordering: (Zcxi, Ycxi, Xcxi)
        - XU ordering: (Xxu, Yxu, Zxu)

        Physical correspondence:

        - Zcxi = Xxu: beam direction (away from source)
        - Ycxi = Zxu: vertical (pointing up)
        - Xcxi = Yxu: outboard (perpendicular to beam)

        Both are right-handed. The beam direction (axis 0) requires
        no swapping as [1,0,0] is identical in both conventions.

        Args:
            data: Array, list, or tuple to swap. Can be a 3-element
                vector or a 3D array.

        Returns:
            Data with swapped convention. Type matches input.

        Raises:
            TypeError: If data is not a 3D array, list, or tuple.

        Example:
            >>> # convert amplitude array from CXI to XU
            >>> amplitude_xu = Geometry.swap_convention(amplitude_cxi)
            >>> # convert vector [Zcxi, Ycxi, Xcxi] to [Xxu, Yxu, Zxu]
            >>> vec_xu = Geometry.swap_convention([1.0, 2.0, 3.0])
            >>> vec_xu
            [1.0, 3.0, 2.0]
        """
        if isinstance(data, (tuple, list)):
            axis0, axis1, axis2 = data
            return type(data)((axis0, axis2, axis1))
        if isinstance(data, np.ndarray):
            if data.shape == (3,):
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
