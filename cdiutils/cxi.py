"""
A submodule for cxi file handling. It provides

Raises:
    ValueError: _description_

Returns:
    _type_: _description_
    """

import h5py
import numpy as np
import re


from cdiutils import __version__


__cxi_version__ = 150


GROUP_ATTRIBUTES = {
    "image": {"default": "data", "nx_class": "NXdata"},
    "geometry": {"default": "name", "nx_class": "NXgeometry"},
    "source": {"default": "energy", "nx_class": "NXsource"},
    "process": {"default": "comment", "nx_class": "NXprocess"},
    "detector": {"default": "description", "nx_class": "NXdetector"},
    "sample": {"default": "sample_name", "nx_class": "NXsample"},
    "parameters": {"default": None, "nx_class": "NXparameters"},
    "result": {"default": None, "nx_class": "NXresult"},
}


class CXIFile:
    IMAGE_MEMBERS = (
        "title", "data", "data_error", "data_space", "data_type", "detector_",
        "dimensionality", "image_center", "image_size", "is_fft_shifter",
        "mask", "process_", "reciprocal_coordinates", "source_"
    )


    def __init__(self, file_path: str, mode: str = "a"):
        self.file_path = file_path
        self.mode = mode
        self.file = None

        # Tracks sub-group counters for each entry
        self._entry_counters = {}
        self._current_entry = None

    @property
    def entry_counters(self) -> None:
        return self._entry_counters

    @property
    def current_entry(self) -> None:
        return self._current_entry

    def open(self):
        """Open the CXI file."""
        if self.file is None:
            self.file = h5py.File(self.file_path, self.mode)
        return self

    def close(self):
        """Close the CXI file."""
        if self.file:
            self.file.close()
            self.file = None

    def __enter__(self):
        """Enter the runtime context related to this object."""
        return self.open()

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit the runtime context related to this object."""
        self.close()

    def set_entry(self, index: int = None) -> str:
        """
        Create or switch to a specific entry group (e.g., 'entry_1').
        """
        if index is None:
            # Get the next available index
            index = 1
            while f"entry_{index}" in self.file:
                index += 1
        entry_name = f"entry_{index}"

        if entry_name not in self.file:  # double check
            self.file.create_group(entry_name)
            self._entry_counters[entry_name] = {}  # Initialise counters

        self._current_entry = entry_name  # Set the current entry context
        return entry_name

    def _get_next_index(self, entry: str, group_type: str) -> int:
        """
        Get the next index for a specific group type (e.g., 'image')
        within an entry.
        """
        if entry not in self._entry_counters:
            self._entry_counters[entry] = {}

        if group_type not in self._entry_counters[entry]:
            return 1
        else:
            return self._entry_counters[entry][group_type] + 1

    def _increment_index(self, entry: str, group_type: str) -> int:
        # if group_type not in self._entry_counters[entry]:
        #     self._entry_counters[entry][group_type] = 1
        # else:
        #     self._entry_counters[entry][group_type] += 1
        self._entry_counters[entry][group_type] = self._get_next_index(
            entry, group_type
        )
        return self._entry_counters[entry][group_type]

    def create_cxi_group(
            self,
            group_type: str,
            index: int = None,
            attrs: dict = None,
            **kwargs,
    ) -> str:
        """
        Create a CXI-compliant group with optional NeXus class.

        Args:
            group_type (str): The type of group (e.g., 'image',
                'process').
            index (int, optional): explicit index. If None, the next
                available index is used. Defaults to None.
            attrs: Additional attributes for the group.
            **kwargs: the data to save in the CXI group.

        Returns:
            str: The full path of the created group.
        """
        if not self._current_entry:
            self.set_entry()  # Ensure at least 'entry_1' exists

        # Determine the next available index if not specified
        if index is None:
            index = self._get_next_index(self._current_entry, group_type)

        # Determine default values from GROUP_ATTRIBUTES
        if group_type not in GROUP_ATTRIBUTES:
            raise ValueError(
                f"Unknown group_type ({group_type}), must be in "
                f"{GROUP_ATTRIBUTES.keys()}"
            )
        default = GROUP_ATTRIBUTES.get(group_type).get("default")
        nx_class = GROUP_ATTRIBUTES.get(group_type).get("nx_class")

        group_name = f"{group_type}_{index}"
        path = f"{self._current_entry}/{group_name}"

        increment = self._create_group(path, nx_class, attrs)
        if increment:
            self._increment_index(self._current_entry, group_type)

        if default:
            self.file[path].attrs["default"] = default

        self.create_cxi_dataset(path, data=kwargs)
        return path

    def _create_group(
            self,
            path: str,
            nx_class: str = None,
            attrs: dict = None
    ) -> bool:
        """
        Private method to handle the creation of groups in the context
        of H5 files, not in the context of CXI.

        Args:
            path (str): the path to create the group at
            nx_class (str, optional): NeXus class for the group.
                Defaults to None.
            attrs (dict, optional): Additional attributes for the group.

        returns: True if the group was created else False.
        """
        if path not in self.file:
            group = self.file.require_group(path)
            if nx_class:
                group.attrs["NX_class"] = nx_class
            if attrs:
                group.attrs.update(attrs)
            return True
        return False

    def create_cxi_dataset(
            self,
            path: str,
            data,
            dtype=None,
            nx_class: str = None,
            **attrs
    ) -> h5py.Dataset | h5py.Group:
        """
        Create a CXI-compliant dataset with optional NeXus class.

        Args:
            path (str): The path to the dataset.
            data: The data to store in the dataset (can be a dict).
            dtype (data-type, optional): The data type for the dataset.
                Defaults to None.
            nx_class (str, optional): The NeXus class for the dataset,
                if applicable. Defaults to None.

        Returns:
            h5py.Dataset: the dataset or group instance created.
        """
        # Handle nested dictionary by creating a group and populating it
        # recursively.
        if isinstance(data, dict):
            group = self._create_group(path, nx_class, **attrs)
            for key, value in data.items():
                # Create a nested group or dataset depending on the
                # value type.
                self.create_cxi_dataset(f"{path}/{key}", value)
            return group

        # Otherwise, simply create a standard dataset.
        data = np.nan if data is None else data
        dataset = self.file.create_dataset(path, data=data, dtype=dtype)
        if nx_class:
            dataset.attrs["NX_class"] = nx_class
        dataset.attrs.update(attrs)
        return dataset

    def softlink(self, path: str, target: str, raise_on_error: bool = False) -> None:
        """
        Create a soft link at the specified path pointing to an existing
        target path.

        Args:
            path (str): the path where the soft link will be created.
            target (str): the target path that the soft link points to.

        Raises:
            ValueError: if the target path does not exist in self.file.
        """
        if not target.startswith("/"):
            target = "/" + target
        if target in self.file:
            self.file[path] = h5py.SoftLink(target)
        elif raise_on_error:
            raise ValueError(f"The target path '{target}' does not exist.")
        else:
            print(f"Warning: The target path '{target}' does not exist.")

    def stamp(self):
        """
        Add metadata to the CXI file, recording information about the
        software and file creation details.
        """
        # Store software information
        self.file.attrs["creator"] = "CdiUtils"
        self.file.attrs["version"] = __version__
        self.create_cxi_dataset("creator", "CdiUtils")
        self.create_cxi_dataset("version", __version__)

        # Store file path, CXI version, and timestamp
        self.create_cxi_dataset("file_path", data=self.file_path)
        self.create_cxi_dataset("cxi_version", data=__cxi_version__)
        self.create_cxi_dataset(
            "time",
            data=np.bytes_(np.datetime64("now").astype(str))
        )

    def create_cxi_image(
            self,
            data: np.ndarray,
            link_data: bool = True,
            **members
    ) -> str:
        """
        Create a minimal CXI image entry with associated metadata and
        soft links.

        Args:
            path (str): the path to the image entry.
            data (np.ndarray): the image data.
            link_data (bool, optional): whether to link to a data_N
                group. Defaults to True.
            **members: additional members to add to the image group.
                Keys ending in a digit will be indexed accordingly.

        Returns:
            str: The full path of the created group.
        """
        # Minimal CXI image entry.
        path = self.create_cxi_group(
            "image", nx_class="NXdata", default="data"
        )
        self.create_cxi_dataset(f"{path}/data", data)
        self.create_cxi_dataset(f"{path}/image_size", data.shape)

        for k, v in members.items():
            # Match the member base and any trailing digit
            match = re.match(r"(.*?)(\d+)?$", k)
            member_base, index = match.groups()

            # Check if the base member is allowed by CXI convention
            if member_base in self.IMAGE_MEMBERS:
                # Construct the full member name, adding the index if present
                if index:
                    self.softlink(f"{path}/{member_base}{index}", v)
                else:
                    self.create_cxi_dataset(f"{path}/{k}", v)
            else:
                print(
                    f"Warning: '{k}' is not allowed in CXI image convention."
                )
        # if this is the first image, it should be default attribute
        # of the parent entry_
        if self._entry_counters[self._current_entry]["image"] == 1:
            self.file[self._current_entry].attrs["default"] = "image_1"

        # link the image to a data entry
        if link_data:
            index = self._get_next_index(self._current_entry, "data")
            self.softlink(f"{self._current_entry}/data_{index}", path)
            self._increment_index(self._current_entry, "data")

        return path
