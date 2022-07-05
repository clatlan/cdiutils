from string import Template


def make_scan_file(
        output_scan_file,
        scan_file_template_path,
        scan,
        sample_name,
        working_directory,
):
    dump_directory = "/".join((working_directory, sample_name, f"S{scan}"))
    with open(scan_file_template_path, "r") as f:
        source = Template(f.read())
    scan_file = source.substitute(
        {
            "scan": scan,
            "save_dir": dump_directory,
            "reconstruction_file": dump_directory + "/modes.h5",
            "sample_name": sample_name,
        }
    )

    with open(output_scan_file, "w") as f:
        f.write(scan_file)

    