import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import scipy.signal
import xrayutilities as xu
from numpy.polynomial import polynomial as P
from scipy.ndimage import center_of_mass

from cdiutils.bcdi.pipeline import pretty_print


def detector_calibration(
    detector_calibration_frames: np.ndarray,
    delta: float,
    nu: float,
    energy: float,
    sdd_estimate: float=None,
    show=True,
    verbose=True,
):

    x_com = []
    y_com = []
    for i in range(detector_calibration_frames.shape[0]):
        com = center_of_mass(detector_calibration_frames[i])
        x_com.append(com[0])
        y_com.append(com[1])
    
    # get the sample to detector distance
    # for that determine how much the the com has moved when the
    # detector has rotated by 1 degree. We may find this value with
    # delta or nu. Here, we do both and calculate the average. The
    # leading coefficient of the function x_com = f(delta) gives
    # how much the x_com has moved when delta has changed by one degree.

    x_com_shift = P.polyfit(delta, x_com, 1)[1]
    y_com_shift = P.polyfit(nu, y_com, 1)[1]

    pix0_x = P.polyfit(delta, x_com, 1)[0]  # pixel 0, reference of the
    # direct beam
    pix0_y = P.polyfit(nu, y_com, 1)[0]

    angle1, angle2 = nu, delta
    if sdd_estimate is None:
        sdd_estimate = (
            (1 / 2)
            * (1 / np.tan(np.pi / 180))
            * (x_com_shift + y_com_shift)
            * 55e-6
        )

    if verbose:
        print("[INFO] First estimate of sdd: {}\n".format(sdd_estimate))
    pretty_print(
        "[INFO] Processing to detector calibration using area_detector_calib"
    )
    parameter_list, eps = xu.analysis.sample_align.area_detector_calib(
        angle1,
        angle2,
        detector_calibration_frames,
        ["z-", "y-"],
        "x+",
        start=(55e-6, 55e-6, sdd_estimate, 0, 0, 0, 0),
        fix=(True, True, False, False, False, False, True),
        wl=xu.en2lam(energy),
    )

    parameters = {
        "cch1": parameter_list[0],
        "cch2": parameter_list[1],
        "pwidth1": parameter_list[2],
        "pwidth2": parameter_list[3],
        "distance": parameter_list[4],
        "tiltazimuth": parameter_list[5],
        "tilt": parameter_list[6],
        "detrot": parameter_list[7],
        "outerangle_offset": parameter_list[8],
    }

    if verbose:
        pretty_print("Computed parameters")
        for k, v in parameters.items():
            print(
                f"{k} = {v}"
            )


    
    if show:
        fig1, ax1 = plt.subplots(1, 1, figsize=(6, 4))
        fig2, axes2 = plt.subplots(1, 2)
        ax1.imshow(np.log10(detector_calibration_frames.sum(axis=0)))
        axes2[0].plot(delta, x_com)
        axes2[0].set_xlabel("delta")
        axes2[0].set_ylabel("COM in x")

        axes2[1].plot(nu, y_com)
        axes2[1].set_xlabel("nu")
        axes2[1].set_ylabel("COM in y")
        fig1.tight_layout()
        fig2.tight_layout()
    
    return parameters
        

    

def det_calib(
    edf_file_template: str,
    calib_scan: str,
    specfile,
    nav=[1, 1],
    roi=[0, 516, 0, 516],
    energy=13000 - 6,
    qconv=None,
    median_filtering=False,
    show=True,
    verbose=True,
):


    frames_id = specfile[calib_scan + ".1/measurement/mpx4inr"][...]
    frames_nb = len(frames_id)
    frames = np.empty((frames_nb, roi[1], roi[3]))
    x_com = np.empty(frames_nb)
    y_com = np.empty(frames_nb)

    positioners = specfile[calib_scan + ".1/instrument/positioners"]
    eta = positioners["eta"][...]
    delta = positioners["del"][...]
    phi = positioners["phi"][...]
    nu = positioners["nu"][...]

    for i, id in enumerate(frames_id):
        edf_data = xu.io.EDFFile(edf_file_template.format(id=int(id))).data
        ccdraw = xu.blockAverage2D(edf_data, nav[0], nav[1], roi=roi)

        if median_filtering:
            frames[i, ...] = scipy.signal.medfilt2d(ccdraw, [3, 3])
        else:
            frames[i, ...] = ccdraw
        x_com[i], y_com[i] = center_of_mass(ccdraw)

    if show:
        fig1, ax1 = plt.subplots(1, 1, figsize=(6, 4))
        fig2, axes2 = plt.subplots(1, 2)
        ax1.imshow(np.log10(frames.sum(axis=0)))
        axes2[0].plot(delta, x_com)
        axes2[0].set_xlabel("delta")
        axes2[0].set_ylabel("COM in x")

        axes2[1].plot(nu, y_com)
        axes2[1].set_xlabel("nu")
        axes2[1].set_ylabel("COM in y")
        fig2.tight_layout()

    # get the sample to detector distance
    # for that determine how much the the com has moved when the
    # detector has rotated by 1 degree. We may find this value with
    # delta or nu. Here, we do both and calculate the average. The
    # leading coefficient of the function x_com = f(delta) gives
    # how much the x_com has moved when delta has changed by one degree.

    x_com_shift = P.polyfit(delta, x_com, 1)[1]
    y_com_shift = P.polyfit(nu, y_com, 1)[1]

    pix0_x = P.polyfit(delta, x_com, 1)[0]  # pixel 0, reference of the
    # direct beam
    pix0_y = P.polyfit(nu, y_com, 1)[0]

    angle1, angle2 = nu, delta
    sdd = (
        (1 / 2)
        * (1 / np.tan(np.pi / 180))
        * (x_com_shift + y_com_shift)
        * 55e-6
    )

    if verbose:
        print("First estimate of sdd: {}\n".format(sdd))
    print(
        "###################################################################\n"
        "[INFO] Processing to detector calibration using area_detector_calib\n"
        "###################################################################"
        "\n\n"
    )
    parameters, eps = xu.analysis.sample_align.area_detector_calib(
        angle1,
        angle2,
        frames,
        ["z-", "y-"],
        "x+",
        start=(55e-6, 55e-6, sdd, 45, 0, -0.7, 0),
        fix=(True, True, False, False, False, False, True),
        wl=xu.en2lam(energy),
    )

    if verbose:
        print(
            "\n\n####################\n"
            "Computed parameters\n"
            "####################\n"
            "cch1 = {}\n"
            "cch2 = {}\n"
            "pwidth1 = {}\n"
            "pwidth2 = {}\n"
            "distance = {}\n"
            "tiltazimuth = {}\n"
            "tilt = {}\n"
            "detrot = {}\n"
            "outerangle_offset = {}\n".format(
                parameters[0],
                parameters[1],
                parameters[2],
                parameters[3],
                parameters[4],
                parameters[5],
                parameters[6],
                parameters[7],
                parameters[8],
            )
        )

    if qconv is None:
        # By default use the ID01 simplified goniometer
        # sample: eta, phi detector nu,del
        # convention for coordinate system: x downstream; z upwards;
        # y to the "outside" (righthanded)
        qconv = xu.experiment.QConversion(
            ["y-", "z-"], ["z-", "y-"], [1, 0, 0]
        )
    hxrd = xu.HXRD([1, 0, 0], [0, 0, 1], en=energy, qconv=qconv)
    hxrd.Ang2Q.init_area(
        "z-",
        "y+",
        cch1=parameters[0] - roi[0],
        cch2=parameters[1] - roi[2],
        Nch1=roi[1] - roi[0],
        Nch2=roi[3] - roi[2],
        pwidth1=5.5000e-05,
        pwidth2=5.5000e-05,
        distance=parameters[4],
        detrot=parameters[7],
        tiltazimuth=parameters[5],
        tilt=parameters[6],
    )

    if show:
        area = hxrd.Ang2Q.area(eta, phi, nu, delta, delta=(0, 0, 0, 0))
        nx, ny, nz = frames.shape
        gridder = xu.Gridder3D(nx, ny, nz)
        gridder(area[0], area[1], area[2], frames)
        qx, qy, qz = gridder.xaxis, gridder.yaxis, gridder.zaxis
        intensity = gridder.data
        # Clip intensity values between
        intensity = xu.maplog(intensity, 6, 0)
        matrix_com = center_of_mass(intensity)
        qcom = (
            qx[round(matrix_com[0])],
            qy[round(matrix_com[1])],
            qz[round(matrix_com[2])],
        )

        if verbose:
            print(
                "Center of mass of intensity in matrix coordinates: {}\n"
                "Center of mass of intensity in reciprocal space coordinates: "
                "{} (angstroms)".format(matrix_com, qcom)
            )

        fig3, axes3 = plt.subplots(1, 3, figsize=(12, 5))
        contour1 = axes3[0].contourf(
            qx, qy, intensity.sum(axis=2).T, 150, cmap="turbo"
        )
        axes3[0].set_xlabel(r"$Q_X (\AA^{-1})$")
        axes3[0].set_ylabel(r"$Q_Y (\AA^{-1})$")

        contour2 = axes3[1].contourf(
            qx, qz, intensity.sum(axis=1).T, 150, cmap="turbo"
        )
        axes3[1].set_xlabel(r"$Q_X (\AA^{-1})$")
        axes3[1].set_ylabel(r"$Q_Z (\AA^{-1})$")

        contour3 = axes3[2].contourf(
            qy, qz, intensity.sum(axis=0).T, 150, cmap="turbo"
        )
        axes3[2].set_xlabel(r"$Q_Y (\AA^{-1})$")
        axes3[2].set_ylabel(r"$Q_Z (\AA^{-1})$")

        for c, ax in zip([contour1, contour2, contour3], axes3.ravel()):
            cbar = fig3.colorbar(
                c, ax=ax, orientation="horizontal", pad=0.18, aspect=15
            )
            tick_locators = mpl.ticker.MaxNLocator(nbins=5)
            cbar.locator = tick_locators
            cbar.update_ticks()

        fig3.tight_layout()

    return hxrd, parameters