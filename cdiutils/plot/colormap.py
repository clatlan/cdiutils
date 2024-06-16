import matplotlib.pyplot as plt
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


TURBO_FIRST_HALF = LinearSegmentedColormap.from_list(
    "turbo_first_half",
    [
        '#30123b', '#31133d', '#31143e', '#321540', '#321641', '#331743',
        '#331744', '#341846', '#341947', '#351a49', '#351b4b', '#361c4c',
        '#361d4e', '#371e4f', '#371f51', '#382052', '#382154', '#382256',
        '#392357', '#392359', '#3a245a', '#3a255c', '#3b265d', '#3b275f',
        '#3b2861', '#3c2962', '#3c2a64', '#3d2b65', '#3d2c67', '#3d2d68',
        '#3e2e6a', '#3e2f6b', '#3e306d', '#3f316f', '#3f3270', '#3f3372',
        '#403473', '#403575', '#403676', '#413778', '#413879', '#41397b',
        '#413a7c', '#423b7e', '#423c80', '#423d81', '#433e83', '#433f84',
        '#434086', '#434187', '#444289', '#44438a', '#44448c', '#44458d',
        '#44468f', '#454790', '#454892', '#454993', '#454a95', '#454c96',
        '#464d98', '#464e99', '#464f9a', '#46509c', '#46519d', '#46529f',
        '#4753a0', '#4754a2', '#4755a3', '#4756a4', '#4757a6', '#4759a7',
        '#475aa9', '#475baa', '#475cab', '#475dad', '#485eae', '#485faf',
        '#4860b1', '#4862b2', '#4863b3', '#4864b5', '#4865b6', '#4866b7',
        '#4867b9', '#4868ba', '#486abb', '#486bbc', '#486cbe', '#486dbf',
        '#486ec0', '#486fc1', '#4871c3', '#4872c4', '#4873c5', '#4774c6',
        '#4775c7', '#4777c8', '#4778ca', '#4779cb', '#477acc', '#477bcd',
        '#477dce', '#467ecf', '#467fd0', '#4680d1', '#4681d2', '#4683d3',
        '#4684d4', '#4585d5', '#4586d6', '#4588d7', '#4589d8', '#448ad9',
        '#448bda', '#448ddb', '#438edc', '#438fdd', '#4391de', '#4292de',
        '#4293df', '#4194e0', '#4196e1', '#4097e2', '#4098e2', '#409ae3',
        '#3f9be4', '#3e9ce4', '#3e9ee5', '#3d9fe6', '#3da0e6', '#3ca2e7',
        '#3ba3e7', '#3aa4e8', '#3aa6e9', '#39a7e9', '#38a8ea', '#37aaea',
        '#36abea', '#35adeb', '#34aeeb', '#33afeb', '#32b1ec', '#30b2ec',
        '#2fb4ec', '#2eb5ed', '#2cb6ed', '#2ab8ed', '#2ab9ed', '#2dbbea',
        '#30bce8', '#33bee6', '#36bfe4', '#38c1e1', '#3bc2df', '#3dc4dd',
        '#3fc5da', '#41c7d8', '#43c8d6', '#45c9d3', '#47cbd1', '#49ccce',
        '#4bcecc', '#4dcfc9', '#4ed1c7', '#50d2c4', '#52d4c2', '#54d5bf',
        '#55d6bc', '#57d8ba', '#59d9b7', '#5adbb4', '#5cdcb1', '#5eddaf',
        '#60dfac', '#61e0a9', '#63e1a6', '#65e3a3', '#67e4a0', '#69e59d',
        '#6be799', '#6de896', '#6fe993', '#71eb8f', '#73ec8c', '#75ed88',
        '#77ee84', '#7af080', '#7cf17c', '#7ff278', '#82f374', '#85f46f',
        '#88f56a', '#8bf765', '#8ef860', '#92f95a', '#96fa54', '#9afa4d',
        '#9efb45', '#a3fc3c'
    ]
)


TURBO_SECOND_HALF = LinearSegmentedColormap.from_list(
    "turbo_second_half",
    ['#a3fc3c', '#a7fa3c', '#abf83c', '#aff63c', '#b3f43b', '#b6f33b',
     '#b9f13b', '#bcef3b', '#bfed3a', '#c2eb3a', '#c4ea3a', '#c6e83a',
     '#c9e639', '#cbe439', '#cde339', '#cfe138', '#d1df38', '#d3dd38',
     '#d4dc37', '#d6da37', '#d8d837', '#d9d736', '#dbd536', '#dcd336',
     '#ddd135', '#dfd035', '#e0ce35', '#e1cc34', '#e3cb34', '#e4c933',
     '#e5c733', '#e6c633', '#e7c432', '#e8c232', '#e9c132', '#eabf31',
     '#ebbd31', '#ebbc30', '#ecba30', '#edb830', '#eeb72f', '#eeb52f',
     '#efb32e', '#f0b22e', '#f0b02d', '#f1ae2d', '#f2ad2d', '#f2ab2c',
     '#f3aa2c', '#f3a82b', '#f4a62b', '#f4a52b', '#f5a32a', '#f5a12a',
     '#f6a029', '#f69e29', '#f79c28', '#f79a28', '#f79928', '#f89727',
     '#f89527', '#f89426', '#f99226', '#f99025', '#f98f25', '#f98d25',
     '#fa8b24', '#fa8a24', '#fa8823', '#fa8623', '#fa8423', '#fb8222',
     '#fb8122', '#fb7f21', '#fb7e21', '#fa7c20', '#f97b20', '#f97a1f',
     '#f8791f', '#f7781e', '#f7761e', '#f6751d', '#f5741d', '#f4731c',
     '#f4721c', '#f3711c', '#f26f1b', '#f16e1b', '#f16d1a', '#f06c1a',
     '#ef6b19', '#ee6a19', '#ed6919', '#ed6818', '#ec6618', '#eb6517',
     '#ea6417', '#e96317', '#e86216', '#e76116', '#e76016', '#e65f15',
     '#e55e15', '#e45d15', '#e35b14', '#e25a14', '#e15913', '#e05813',
     '#df5713', '#de5613', '#de5512', '#dd5412', '#dc5312', '#db5211',
     '#da5111', '#d95011', '#d84f10', '#d74e10', '#d64d10', '#d54c10',
     '#d44b0f', '#d34a0f', '#d2490f', '#d1480f', '#d0470e', '#cf460e',
     '#ce450e', '#cd440e', '#cc430d', '#cb420d', '#ca410d', '#c9400d',
     '#c83f0c', '#c73e0c', '#c53d0c', '#c43c0c', '#c33b0c', '#c23a0b',
     '#c1390b', '#c0380b', '#bf370b', '#be370b', '#bd360b', '#bc350a',
     '#bb340a', '#ba330a', '#b8320a', '#b7310a', '#b6300a', '#b52f09',
     '#b42e09', '#b32d09', '#b22d09', '#b12c09', '#af2b09', '#ae2a09',
     '#ad2909', '#ac2808', '#ab2708', '#aa2608', '#a92508', '#a72508',
     '#a62408', '#a52308', '#a42208', '#a32108', '#a22008', '#a01f08',
     '#9f1f07', '#9e1e07', '#9d1d07', '#9c1c07', '#9b1b07', '#991a07',
     '#981907', '#971907', '#961807', '#951707', '#931607', '#921506',
     '#911406', '#901306', '#8f1306', '#8d1206', '#8c1106', '#8b1006',
     '#8a0f06', '#890e05', '#870d05', '#860c05', '#850b05', '#840b05',
     '#830a04', '#810904', '#800804', '#7f0704', '#7e0604', '#7c0503',
     '#7b0503', '#7a0403']
)


def save_json_cmap(colormap_name: str, output_path: str) -> None:
    cmap = plt.get_cmap(colormap_name)
    pace = 0.015
    array = np.arange(0, 1+pace, pace)

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
        sys.exit()
    save_json_cmap(sys.argv[1], sys.argv[2])
