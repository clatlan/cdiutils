import h5py
import silx.io
import scipy.signal
import numpy as np
from numpy.polynomial import polynomial as P
import xrayutilities as xu
import matplotlib.pyplot as plt
from scipy.ndimage.measurements import center_of_mass


# TODO: Make a function that handles the detector calibration 

if __name__ == '__main__':
    file = "/data/id01/inhouse/data/IHR/ihhc3586/id01/spec/" \
        "2021_01_21_151706_platinum.spec"


    sf = silx.io.open(file)
    detector_calibration_scan = "449"
    mpx4inr = sf[detector_calibration_scan + ".1/measurement/mpx4inr"][...]

    positioners = sf[detector_calibration_scan + ".1/instrument/positioners"]

    eta = positioners["eta"][...]
    delta = positioners["del"][...]
    phi = positioners["phi"][...]
    nu = positioners["nu"][...]
    mu = positioners["mu"][...]
    outerangle_offset = 0



    frame_id = mpx4inr
    edf_file_template = "/data/id01/inhouse/data/IHR/ihhc3586/id01/detector/" \
    	+ "2021_01_21_151602_platinum/data_mpx4_{id:0>5d}.edf.gz"

    roi = [0, 516, 0, 516]
    nav = [1, 1]

    frames_nb = len(frame_id)
    frames = np.empty((frames_nb, roi[1], roi[3]))
    x_com = np.empty(frames_nb)
    y_com = np.empty(frames_nb)

    for i, id in enumerate(frame_id):
    	edf_data = xu.io.EDFFile(edf_file_template.format(id=int(id))).data
    	ccdraw =  xu.blockAverage2D(edf_data, nav[0], nav[1], roi=roi)
    	frames[i] = scipy.signal.medfilt2d(ccdraw, [3, 3])

    	x_com[i], y_com[i] = center_of_mass(ccdraw)

    detector_figure = plt.figure(figsize=(5,5))
    ax1 = detector_figure.add_subplot()
    ax1.imshow(np.log10(frames.sum(axis=0)))

    com_figure, ax = plt.subplots(1, 2, figsize=(8,3))
    ax[0].plot(delta, x_com)
    ax[0].set_xlabel("delta")
    ax[0].set_ylabel("COM in x")

    ax[1].plot(nu, y_com)
    ax[1].set_xlabel("nu")
    ax[1].set_ylabel("COM in y")
    com_figure.tight_layout()

    plt.show()

    # get the sample to detector distance
    # for that determine how much the the com has moved when the detector
    # has rotated by 1 degree. We may find this value with delta or nu. Here, we do
    # both and calculate the average. The leading coefficient of the function
    # x_com = f(delta) gives how much the x_com has moved when delta has changed by
    # one degree.

    x_com_shift = P.polyfit(delta, x_com, 1)[1]
    y_com_shift = P.polyfit(nu, y_com, 1)[1]

    pix0_x = P.polyfit(delta, x_com, 1)[0] # pixel 0, reference of the direct beam
    pix0_y = P.polyfit(nu, y_com, 1)[0]

    sdd = (1/2) * (1 / np.tan(np.pi / 180)) * (x_com_shift + y_com_shift) * 55e-6

    param, eps = xu.analysis.sample_align.area_detector_calib(
        angle1, angle2,ccdimages, ['z-', 'y-'],'x+',
        start=(55e-6, 55e-6, ssd, 45, 0, -0.7, 0),
        fix=(True, True, False, False, False, False, True),
        wl=xu.en2lam(en))





    # qx, qy, qz = hxrd.Ang2Q.area(eta, phi, nu, delta,
    # 							 delta=(0, 0, outerangle_offset, 0))
