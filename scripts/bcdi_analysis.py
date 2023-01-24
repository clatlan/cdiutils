#!/usr/bin/env python3

import sys

from cdiutils.processing.pipeline import BcdiPipeline

usage_text = (
    f"Usage: {sys.argv[0]} [-a, -prp, -phr, -pop, -fpop, -md, --cdiutils, "
    "--bcdi] path_to_scan_file.yml"
)


if __name__ == "__main__":
    prp = False
    phr = False
    pop = False
    fpop = False
    md = False

    if len(sys.argv) <= 1:
        print(usage_text)
        sys.exit(1)

    backend = "cdiutils"
    for arg in sys.argv[1:]:
        if arg.endswith(".yml") or arg.endswith(".yaml"):
            scan_file = arg
        elif arg == '-a':
            prp, phr, fpop = True, True, True
        elif arg == '-prp':
            prp = True
        elif arg == '-phr':
            phr = True
        elif arg == '-pop':
            pop = True
        elif arg == '-fpop':
            fpop = True
        elif arg == '-md':
            md = True
        elif arg == '--cdiutils':
            backend = "cdiutils"
        elif arg == '--bcdi':
            backend = "bcdi"
        else:
            print("Unknown arguments")
            print(usage_text)
            sys.exit(1)

    bcdi_pipeline = BcdiPipeline(scan_file, backend=backend)

    if prp:
        bcdi_pipeline.preprocess(backend=backend)

    if phr:
        bcdi_pipeline.phase_retrieval(
            # machine="lid01pwr9"
            machine="p9",
            remove_last_results=True
        )
    if fpop:
        bcdi_pipeline.find_best_candidates(nb_to_keep=5)
        bcdi_pipeline.mode_decomposition()
        bcdi_pipeline.postprocess(backend=backend)
        bcdi_pipeline.save_parameter_file()

    if md:
        bcdi_pipeline.mode_decomposition()

    if pop:
        bcdi_pipeline.postprocess(backend=backend)
        bcdi_pipeline.save_parameter_file()