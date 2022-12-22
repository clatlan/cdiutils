from matplotlib.cm import get_cmap
import numpy as np
import colorcet

from matplotlib.colors import ListedColormap, LinearSegmentedColormap


def save_json_cmap(colormap_name: str, output_path: str) -> None:
    cmap = get_cmap(colormap_name)
    path = 0.03125
    path = 0.015
    array = np.arange(0, 1+path, path)

    with open(output_path, "w") as file:
        file.write(
            '''
            [
                {
                    "ColorSpace" : "Lab",
                    "Creator" : "Matplotlib",
                    "DefaultMap" : true,
            '''
        )
        file.write(
            f'"Name" : "{colormap_name}",'
        )
        file.write(
            '''
                        "NanColor" :
                        [
                            0,
                            0,
                            0
                        ],
                        "RGBPoints" :
                            [
        '''.format(s=colormap_name)
        )
        for i, e in enumerate(array):
            file.write(
                f"{e} ,\n"
                f"{cmap(e)[0]},\n"
                f"{cmap(e)[1]},\n"
                f"{cmap(e)[2]}"
            )
            if i != array.shape[0]-1:
                file.write(",\n")

        file.write(
            """
            ]
        }
    ]
            """
        )


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 3:
        print(
            f"usage: {sys.argv[0]}, cmap_name, output_path"
        )
        exit(1)
    save_json_cmap(sys.argv[1], sys.argv[2])
    