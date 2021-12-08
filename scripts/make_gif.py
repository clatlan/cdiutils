import imageio

if __name__ == '__main__':
    import argparse

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--files", required=False, type=str, nargs="+",
                    help="files to read")
    args = vars(ap.parse_args())

    scan_digits = [182, 183, 184, 185]

    if args["files"] is None:
        file_template = (
            "/data/id01/inhouse/clatlan/exchange/facet-dependent-images/"
            "cross_section_quivers/strain-disp/same_support/"
            "seismic-turbo_strain-disp_S{}.png"
        )
        files = [file_template.format(i, i) for i in scan_digits]
    else:
        files = args["files"]

    images = []
    for f in files:
        images.append(imageio.imread(f))

    imageio.mimsave(
        "/data/id01/inhouse/clatlan/exchange/facet-dependent-images/"
        "cross_section_quivers/strain-disp/gifs/"
        "seismic-turbo_strain-disp.gif",
        images)
