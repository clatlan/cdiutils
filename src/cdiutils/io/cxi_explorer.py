"""
Module for exploring and visualising CXI files with interactive
browser functionality.
"""

from pathlib import Path

import h5py
import ipywidgets
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display

from cdiutils.io.cxi import CXIFile
from cdiutils.plot.formatting import add_colorbar
from cdiutils.plot.slice import plot_volume_slices


class CXIExplorer:
    """
    A class for inspecting and exploring the content of a CXI file.

    This explorer provides various ways to inspect CXI file structure,
    including:
    - Tree view of the file hierarchy
    - Detailed information about datasets and attributes
    - Visualisation of array data
    """

    max_number_of_values_printed = 100
    tree_max_string_length = 15
    tree_max_array_size = 5

    def __init__(self, cxi_file: str | CXIFile) -> None:
        """
        Initialise the CXI explorer with either a path to a CXI file or
        a CXIFile object.

        Args:
            cxi_file (str | CXIFile): Either a string path to a CXI file
            or a CXIFile object
        """
        # handle both string paths and CXIFile objects
        if isinstance(cxi_file, str):
            self.file_path = cxi_file
            self.cxi = CXIFile(cxi_file, mode="r")
            self.cxi.open()
            self._owner = True  # we opened it, so we should close it
        elif isinstance(cxi_file, CXIFile):
            self.file_path = cxi_file.file_path
            self.cxi = cxi_file
            self._owner = False  # we didn't open it, so shouldn't close it
        else:
            raise TypeError(
                "cxi_file must be a string path or a CXIFile object"
            )

        # check if file is open
        if self.cxi.file is None:
            self.cxi.open()

        self.tab = 0

        # build path lists including regular paths and soft links
        self.paths, self.soft_links = self._build_path_lists()

    def close(self):
        """Close the CXI file if it was opened by this explorer."""
        if self._owner and self.cxi.file is not None:
            self.cxi.close()
            self._owner = False

    def __del__(self):
        """Clean up resources when the explorer is deleted."""
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._owner and hasattr(self, "cxi"):
            self.cxi.close()

    def _build_path_lists(self) -> tuple[list, dict]:
        """
        Build lists of all paths in the file, including regular items
        and soft links.

        Returns:
            tuple: (paths, soft_links) where:
                - paths is a list of all paths
                - soft_links is a dict mapping link names to their
                target paths
        """
        paths = []
        soft_links = {}

        def collect_paths(name, obj):
            paths.append(name)

        def collect_links(name, obj):
            if isinstance(obj, h5py.SoftLink):
                soft_links[name] = obj.path

        # collect all regular items
        self.cxi.file.visititems(collect_paths)

        # collect all soft links
        self.cxi.file.visititems_links(collect_links)

        # add soft links to the paths list if not already there
        for link_name in soft_links:
            if link_name not in paths:
                paths.append(link_name)

        paths.sort()

        return paths, soft_links

    def _is_softlink(self, path: str) -> tuple:
        """
        Check if a path is a soft link.

        Args:
            path (str): Path to check

        Returns:
            tuple: (bool, str): if path is a softlink and the path it
                points to.
        """
        try:
            link_info = self.cxi.file.get(path, getlink=True)
            if isinstance(link_info, h5py.SoftLink):
                return True, link_info.path
            return False, None
        except (KeyError, AttributeError, RuntimeError):
            return False, None

    def tree(
        self, max_depth: int = None, show_attributes: bool = False
    ) -> None:
        """
        Print a tree view of the CXI file structure similar to the Linux
        'tree' command.

        Args:
            max_depth (int): Maximum depth to traverse (None for
                unlimited). Defaults to None.
            show_attributes (bool): whether to show attributes. Defaults
            to False.
        """
        print(f"CXI File: {Path(self.file_path).name}")

        def print_group(name, node, prefix="", is_last=True, depth=0):
            # Determine the base name (last part of the path)
            basename = name.split("/")[-1] if name != "/" else "/"

            # Check if this is a soft link
            is_link, target = self._is_softlink(name)
            if is_link:
                print(
                    f"{prefix}{'└── ' if is_last else '├── '}{basename} -> {target}"  # noqa: E501
                )
                return

            # Print current item with appropriate branch symbols
            if isinstance(node, h5py.Group):
                # Add attributes if present
                attr_str = ""
                if len(node.attrs) > 0 and show_attributes:
                    attr_keys = list(node.attrs.keys())
                    if len(attr_keys) <= 2:  # Show all if few
                        attrs = [f"{k}={node.attrs[k]}" for k in attr_keys]
                        attr_str = f" ({', '.join(attrs)})"
                    else:  # Otherwise just show count
                        attr_str = f" ({len(node.attrs)} attributes)"

                print(
                    f"{prefix}{'└── ' if is_last else '├── '}{basename}{attr_str}"  # noqa: E501
                )

                # Prepare for children
                keys = list(node.keys())
                child_prefix = prefix + ("    " if is_last else "│   ")

                # Check if we've reached the max depth for child nodes
                if max_depth is not None and depth == max_depth:
                    # Only count direct children at this level
                    if keys:  # Only show if there are actually children
                        print(
                            f"{child_prefix}└── {len(keys)} more entrie(s)..."
                        )  # noqa: E501
                    return

                # Process all children
                for i, key in enumerate(keys):
                    child_name = f"{name}/{key}" if name != "/" else f"/{key}"
                    try:
                        child_node = node[key]
                        print_group(
                            child_name,
                            child_node,
                            child_prefix,
                            i == len(keys) - 1,
                            depth + 1,
                        )
                    except Exception as e:
                        # Handle any issues accessing the child
                        print(
                            f"{child_prefix}{'└── ' if i == len(keys) - 1 else '├── '}{key} (Error: {str(e)})"  # noqa: E501
                        )

            elif isinstance(node, h5py.Dataset):
                # Format dataset info
                shape_str = f"{node.shape}" if node.shape else "(scalar)"
                type_str = f"{node.dtype}"

                # Truncate type string if it's too long
                if len(type_str) > self.tree_max_string_length:
                    type_str = (
                        type_str[: self.tree_max_string_length - 3] + "..."
                    )

                # Add short data preview for small datasets
                data_preview = ""
                if node.size <= self.tree_max_array_size:  # small datasets
                    value = self.cxi[name]
                    if (
                        isinstance(value, np.ndarray)
                        and value.size <= self.tree_max_array_size
                    ):
                        data_preview = f" = {value}"
                    elif np.isscalar(value) or isinstance(value, (str, bytes)):
                        if isinstance(value, (bytes, np.bytes_)):
                            try:
                                data_preview = f" = '{value.decode('utf-8')}'"
                            except UnicodeDecodeError:
                                data_preview = f" = {value}"
                            else:
                                data_preview = f" = {value}"

                # Add attributes indicator
                attr_str = ""
                if len(node.attrs) > 0:
                    attr_str = f" ({len(node.attrs)} attributes)"

                print(
                    f"{prefix}{'└── ' if is_last else '├── '}{basename} {shape_str} {type_str}{data_preview}{attr_str}"  # noqa: E501
                )

        # Start recursion from root
        print_group("/", self.cxi.file, "")

    def explore(self):
        """
        Create an interactive widget to explore the CXI file.
        """
        # create dropdown for selecting paths
        path_dropdown = ipywidgets.Dropdown(
            options=self.paths,
            description="Path:",
            style={"description_width": "initial"},
            layout=ipywidgets.Layout(width="80%"),
        )

        # create output widget to display content
        output = ipywidgets.Output()

        def show_item(change):
            path = change["new"]
            with output:
                output.clear_output()
                self.show(path)

        # Connect the callback to the dropdown widget
        path_dropdown.observe(show_item, names="value")

        # Set initial selection if possible
        if self.paths:
            with output:
                show_item({"new": self.paths[0]})

        # Display the ipywidgets
        display(ipywidgets.VBox([path_dropdown, output]))

    def tabbed_print(self, text: str, **kwargs) -> None:
        """
        Print text with optional tabbing.

        Args:
            text (str): Text to print.
        """
        print("\t" * self.tab + text, **kwargs)

    def _show_h5_dataset(
        self,
        dataset: h5py.Dataset,
        show_attributes: bool = False,
        plot: bool = True,
    ) -> None:
        """
        Show the content of a dataset in the CXI file.

        Args:
            dataset (h5py.Dataset): the h5 dataset to show.
            show_attributes (bool, optional): whether to show
                attributes. Defaults to False.
            plot (bool, optional): whether to plot. Defaults to True.
        """

        self.tabbed_print(f"Path: {dataset.name}")
        self.tabbed_print("Type: Dataset")
        data = dataset[()]

        if len(dataset.attrs) > 0 and show_attributes:
            self.tabbed_print(f"Attributes: {dict(dataset.attrs)}")

        # Handle different data types for visualisation

        # For string data
        if isinstance(data, (str, bytes, np.bytes_)):
            if isinstance(data, (bytes, np.bytes_)):
                try:
                    self.tabbed_print(f"Value: {data.decode('utf-8')}")
                except UnicodeDecodeError:
                    self.tabbed_print(f"Value: {data} (binary data)")
            else:
                self.tabbed_print(f"Value: {data}")

        # For scalar data
        elif np.isscalar(data) or data.size == 1:
            self.tabbed_print(f"Value: {data}")

        # For array data
        elif isinstance(data, np.ndarray):
            self.tabbed_print(f"Shape: {data.shape}")
            self.tabbed_print(f"Dtype: {data.dtype}")

            # For small arrays, show the values
            if data.size <= self.max_number_of_values_printed:
                self.tabbed_print(f"Values: {data}")

            if plot and data.size > 3:
                self.tabbed_print(
                    f"Data summary: min={np.nanmin(data):.3f}, "
                    f"max={np.nanmax(data):.3f}, "
                    f"mean={np.nanmean(data):.3f}"
                )
                # For 1D arrays of reasonable size, plot them
                if data.ndim == 1:
                    fig, ax = plt.subplots(1, 1, layout="tight")
                    ax.plot(data)
                    fig.suptitle(f"Plot of {dataset.name}")

                # For 2D arrays, display as images
                elif data.ndim == 2:
                    fig, ax = plt.subplots(1, 1, layout="tight")
                    ax.imshow(data)
                    add_colorbar(ax, label="Value")
                    fig.suptitle(f"Image of {dataset.name}")

                # For 3D arrays, show middle slice
                elif data.ndim == 3:
                    plot_volume_slices(
                        data,
                        title=f"3D Volume of {dataset.name}",
                        origin="lower",
                    )

                plt.show()

    def _show_h5_softlink(
        self,
        path: str,
        target_path: str,
    ) -> None:
        """
        Show the content of a soft link in the CXI file.

        Args:
            path (str): Path to the soft link.
            target_path (str): Path to the linked item.
        """
        self.tabbed_print(f"Path: {path}")
        self.tabbed_print(f"Type: Soft Link → {target_path}")

    def _show_h5_group(
        self,
        group: h5py.Group,
        show_attributes: bool = False,
    ) -> None:
        """
        Show the content of a group in the CXI file.

        Args:
            group (h5py.Group): the h5 group to show.
            show_attributes (bool, optional): whether to show
                attributes. Defaults to False.
        """
        path = group.name
        self.tabbed_print(f"Path: {path}")
        if "title" in group:
            self.tabbed_print(f"Title: {self.cxi[path + '/title']}")
        if "description" in group:
            self.tabbed_print(
                f"Description: {self.cxi[path + '/description']}"
            )
        self.tabbed_print(f"Type: Group with {len(group)} items.")
        if len(group.attrs) > 0 and show_attributes:
            self.tabbed_print(f"Attributes: {dict(group.attrs)}")
        self.tabbed_print("Content:")

        self.tab += 1
        # Print content
        for key, item in group.items():
            print()
            # Check if the item is a soft link
            is_softlink, target_path = self._is_softlink(item.name)
            if is_softlink:
                self._show_h5_softlink(item.name, target_path)
            elif isinstance(item, h5py.Group):
                shape_info = f" {item.shape}" if hasattr(item, "shape") else ""
                self.tabbed_print(
                    f"Path: {item.name}\n\tType: Group{shape_info}"
                )
            elif isinstance(item, h5py.Dataset):
                self._show_h5_dataset(item, show_attributes, plot=True)
        self.tab -= 1

    def show(
        self, path: str, show_attributes: bool = False, tab: int = 0
    ) -> None:
        """
        Visualise a specific dataset from the CXI file.

        Notes: This function only prints the content of the subset
        without going any deeper into the tree, except for soft links,
        for which it will print the target content and its content. If
        the target content of a group or dataset is a soft link, it will
        not print the target content.

        Args:
            path: Path to the dataset within the CXI file.
            show_attributes: Whether to show attributes.
            tab: Initial tab level.
        """
        node = self.cxi.get_node(path)

        self.tab = tab

        # Check if this is a soft link first
        is_link, target_path = self._is_softlink(path)
        if is_link:
            self._show_h5_softlink(path, target_path)
            print("Target Content:\n")
            self.tab += 1
            self.show(target_path, show_attributes, tab=self.tab)

        elif isinstance(node, h5py.Group):
            self._show_h5_group(node, show_attributes)
        elif isinstance(node, h5py.Dataset):
            self._show_h5_dataset(node, show_attributes)

    def search(self, pattern: str, search_attrs: bool = True) -> None:
        """
        Search for datasets, groups, or soft links that match the given
        pattern.

        Args:
            pattern (str): pattern to search for in names
            search_attrs (bool): whether to also search in attribute
                values. Defaults to True.
        """
        results = []

        # First check for matches in path names (including soft links)
        for path in self.paths:
            if pattern.lower() in path.lower():
                # Check if it's a soft link
                if path in self.soft_links:
                    results.append(
                        (
                            path,
                            f"soft link name match (→ {self.soft_links[path]})",  # noqa: E501
                        )
                    )
                else:
                    results.append((path, "name match"))

        # Check soft link targets
        for link_name, target_path in self.soft_links.items():
            if pattern.lower() in target_path.lower():
                results.append(
                    (link_name, f"soft link target match (→ {target_path})")
                )

        # Check attributes if requested
        if search_attrs:

            def attr_visitor(name, obj):
                if hasattr(obj, "attrs"):
                    for attr_name, attr_value in obj.attrs.items():
                        attr_str = str(attr_value).lower()
                        if pattern.lower() in attr_str:
                            results.append(
                                (
                                    name,
                                    f"attribute match: {attr_name}={attr_value}",  # noqa: E501
                                )
                            )

            self.cxi.file.visititems(attr_visitor)

        # Display results
        if not results:
            print(f"No matches found for '{pattern}'")
        else:
            print(f"Found {len(results)} matches for '{pattern}':")
            for path, match_type in results:
                # Determine node type (Group, Dataset, or Soft Link)
                if path in self.soft_links:
                    node_type = "Soft Link"
                else:
                    try:
                        node_type = (
                            "Group"
                            if isinstance(self.cxi.file[path], h5py.Group)
                            else "Dataset"
                        )
                    except KeyError:
                        # This should not happen, but just in case
                        node_type = "Unknown"

                print(f"- {node_type}: {path} ({match_type})")

    def summarise(self):
        """Provide a summary of the CXI file content."""
        # Get file info
        file_size = Path(self.file_path).stat().st_size / (1024 * 1024)  # MB

        # Count groups, datasets, and total data size
        group_count = 0
        dataset_count = 0
        total_data_size = 0

        def count_visitor(name, obj):
            nonlocal group_count, dataset_count, total_data_size

            if isinstance(obj, h5py.Group):
                group_count += 1
            elif isinstance(obj, h5py.Dataset):
                dataset_count += 1
                total_data_size += obj.size * obj.dtype.itemsize

        self.cxi.file.visititems(count_visitor)

        # Display summary
        print(f"CXI File Summary: {Path(self.file_path).name}")
        print(f"File size: {file_size:.2f} MB")
        print(f"Groups: {group_count}")
        print(f"Datasets: {dataset_count}")
        print(f"Total data size: {total_data_size / (1024 * 1024):.2f} MB")

        # Check for entry groups
        entries = [
            key for key in self.cxi.file.keys() if key.startswith("entry_")
        ]
        print(f"\nEntries: {len(entries)}")

        for entry in entries:
            print(f"\n{entry}:")
            group = self.cxi.file[entry]

            # Print default if available
            if "default" in group.attrs:
                print(f"  Default: {group.attrs['default']}")

            # List main group types and counts
            group_types = {}
            for key in group.keys():
                prefix = key.split("_")[0] if "_" in key else key
                group_types[prefix] = group_types.get(prefix, 0) + 1

            for prefix, count in group_types.items():
                print(f"  {prefix}: {count}")


# Add a convenience function to CXIFile
def get_explorer(self):
    """
    Create and return a CXIExplorer for this CXI file.

    Returns:
        CXIExplorer: An explorer instance for this file
    """
    return CXIExplorer(self)


# Add the explore method to CXIFile
CXIFile.get_explorer = get_explorer
