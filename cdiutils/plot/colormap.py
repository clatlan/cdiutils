from matplotlib.cm import get_cmap
import numpy as np
import colorcet

from matplotlib.colors import LinearSegmentedColormap


RED_TO_TEAL = LinearSegmentedColormap.from_list(

    "red_to_teal",
    [
        '#f84650', '#fb4358', '#fb455b', '#fc475d', '#fc495e', '#fc4b60',
        '#fd4d62', '#fd4f64', '#fd5165', '#fd5367', '#fe5568', '#fe576a',
        '#fe596b', '#fe5b6d', '#fe5d6e', '#ff5f70', '#ff6171', '#ff6373',
        '#ff6574', '#ff6776', '#ff6977', '#ff6a79', '#ff6c7a', '#ff6e7c',
        '#ff707d', '#ff727e', '#ff7380', '#ff7581', '#ff7783', '#ff7984',
        '#ff7a85', '#ff7c87', '#ff7e88', '#ff7f8a', '#ff818b', '#ff838c',
        '#ff848e', '#ff868f', '#ff8890', '#ff8992', '#ff8b93', '#ff8d95',
        '#fe8e96', '#fe9097', '#fe9199', '#fe939a', '#fe959b', '#fe969d',
        '#fd989e', '#fd99a0', '#fd9ba1', '#fd9da2', '#fd9ea4', '#fca0a5',
        '#fca1a6', '#fca3a8', '#fca4a9', '#fba6ab', '#fba7ac', '#fba9ad',
        '#fbabaf', '#faacb0', '#faaeb1', '#faafb3', '#f9b1b4', '#f9b2b6',
        '#f9b4b7', '#f8b5b8', '#f8b7ba', '#f8b8bb', '#f7babc', '#f7bbbe',
        '#f6bdbf', '#f6bec1', '#f6c0c2', '#f5c1c3', '#f5c3c5', '#f4c4c6',
        '#f4c6c7', '#f3c7c9', '#f3c9ca', '#f2cacc', '#f2cccd', '#f1cdce',
        '#f1cfd0', '#f0d0d1', '#efd2d3', '#efd3d4', '#eed4d5', '#eed6d7',
        '#edd7d8', '#ecd9d9', '#ecdadb', '#ebdcdc', '#eaddde', '#eadfdf',
        '#e9e0e0', '#e8e2e2', '#e8e3e3', '#e7e5e5', '#e3e5e5', '#e1e5e4',
        '#dfe4e4', '#dce4e3', '#dae3e2', '#d7e3e1', '#d5e2e0', '#d3e1e0',
        '#d0e1df', '#cee0de', '#ccdfdd', '#c9dfdd', '#c7dedc', '#c5dddb',
        '#c3ddda', '#c0dcda', '#bedbd9', '#bcdbd8', '#badad7', '#b8d9d7',
        '#b6d8d6', '#b4d8d5', '#b2d7d5', '#afd6d4', '#add6d3', '#abd5d2',
        '#a9d4d2', '#a7d3d1', '#a5d3d0', '#a3d2d0', '#a1d1cf', '#9fd0ce',
        '#9dcfcd', '#9bcfcd', '#99cecc', '#97cdcb', '#95cccb', '#93ccca',
        '#91cbc9', '#90cac9', '#8ec9c8', '#8cc8c7', '#8ac7c7', '#88c7c6',
        '#86c6c5', '#84c5c5', '#82c4c4', '#80c3c3', '#7ec3c2', '#7dc2c2',
        '#7bc1c1', '#79c0c1', '#77bfc0', '#75bebf', '#73bebf', '#71bdbe',
        '#70bcbd', '#6ebbbd', '#6cbabc', '#6ab9bb', '#68b8bb', '#66b8ba',
        '#64b7b9', '#62b6b9', '#61b5b8', '#5fb4b7', '#5db3b7', '#5bb3b6',
        '#59b2b5', '#57b1b5', '#55b0b4', '#53afb4', '#51aeb3', '#4fadb2',
        '#4dacb2', '#4bacb1', '#49abb0', '#47aab0', '#45a9af', '#43a8af',
        '#41a7ae', '#3fa6ad', '#3da5ad', '#3aa5ac', '#38a4ab', '#36a3ab',
        '#33a2aa', '#31a1aa', '#2ea0a9', '#2c9fa8', '#299ea8', '#269ea7',
        '#239da6', '#209ca6', '#1c9ba5', '#189aa5', '#1499a4', '#0e98a3',
        '#0797a3', '#0097a2'
    ]
)


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
        '''
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
