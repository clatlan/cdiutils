#!/usr/bin/env python3

import os
import sys
import yaml

from cdiutils.processing.pipeline import make_scan_parameter_file

if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise ValueError(
            f"Usage: {sys.argv[0]} path_to_scan_lists.yml "
            "path_to_scan_file_template.yml"
        )

    OUTPUT_DIR = os.path.dirname(sys.argv[2])

    with open(sys.argv[1], "r", encoding="utf8") as file:
        scan_dict = yaml.load(file,  Loader=yaml.FullLoader)

    if len(sys.argv) >= 4:
        scans = [int(s) for s in sys.argv[3:]]
    else:
        scans = scan_dict.keys()
    for s in scans:
        sample_name = scan_dict[s]["sample_name"] 
        dump_dir = (
             f"{os.getcwd()}/results/{sample_name}/S{s}/"
        )
        print(f"Making scan parameter file of {s}_{sample_name}")

        make_scan_parameter_file(
            output_parameter_file_path=(
                f"{OUTPUT_DIR}/scan_{s}_{sample_name}.yml"
            ),
            parameter_file_template_path=sys.argv[2],
            updated_parameters={
                "scan": s,
                "sample_name": sample_name,
                "dump_dir": dump_dir,
                "data": "$data",
                "mask": "$mask"
            }
        )
