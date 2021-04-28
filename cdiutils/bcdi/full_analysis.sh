#!/bin/sh

function make_pynx_config() {

	support_threshold=($3)

	if test -f $1/pynx-cdi-inputs.txt; then
    	rm $1/pynx-cdi-inputs.txt 
	fi
	data=($1/*_pynx_norm*.npz)
	data="${data##*/}"
	mask=($1/*maskpynx*.npz)
	mask="${mask##*/}"

	cat <<EOT>> $1/pynx-cdi-inputs.txt 
# parameters
data = "$data"
mask = "$mask"

data2cxi = True 
auto_center_resize = False

#support_type=
#support_size=
support = "$2"
support_threshold =  ${support_threshold[0]}, ${support_threshold[1]}  # pick a random number between these two numbers
support_method = "$4"
support_only_shrink = False
support_update_period = 20
support_smooth_width_begin = 2
support_smooth_width_end = 1
support_post_expand = 1,-2,1

#algorithm=(Sup*ER**20)**5 * PSF**100*RAAR**10 * (Sup*RAAR**20)**10 * PSF**100*ER**10 * (Sup*ER**10)**5 * PSF**100*RAAR**10 * (Sup*RAAR**20)**10 * (Sup*ER**10)**5 * (Sup*RAAR**20)**10
psf = $5 #True #False #
nb_raar = 1000
nb_hio = 400
nb_er = 300
nb_ml = 0

nb_run = $6
nb_run_keep = $7

zero_mask = $8 # masked pixels will start from imposed 0 and then let free
# max_size = 256 
crop_output= 0 # set to 0 to avoid cropping the output in the .cxi
# if N > 0 it will crop to support range + N pixels

positivity = False
beta = 0.9
detwin = False
rebin = 1,1,1

# Generic parameters
detector_distance = $9 # in m 
pixel_size_detector = 55e-6   # eiger 75e-6, maxipix 55e-6
wavelength = 1.5e-10
verbose = 100
output_format = 'cxi'  # 'npz'
live_plot = False
EOT
}



function make_slurm_config() {

	if test -f $1/pynx-id01cdi.slurm; then
    	rm $1/pynx-id01cdi.slurm 
	fi

	cat <<EOT>> $1/pynx-id01cdi.slurm
#!/bin/bash -l
#SBATCH --nodes=2
#SBATCH --ntasks=4
#SBATCH --partition=p9gpu
#SBATCH --gres=gpu:2
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --output=%x-%j.out
  
env | grep CUDA
scontrol --details show jobs $SLURM_JOBID |grep RES
 
## echo `date`  | mail -s " SLURM JOB $SLURM_JOBID has started" your.mail@example.fr
 
cd $1
 
mpiexec -n \$SLURM_NTASKS /sware/exp/pynx/devel.p9/bin/pynx-id01cdi.py pynx-cdi-inputs.txt
 
## echo `date`  | mail -s " SLURM has finished job: $SLURM_JOBID !! " your.mail@example.fr

EOT
}


# Required paths
WORKING_DIRECTORY=/data/id01/inhouse/clatlan/experiments/ihhc3586/analysis
OUTPUT_DIRECTORY=$WORKING_DIRECTORY/results
BCDI_ENV_PATH=/data/id01/inhouse/clatlan/.envs/drxutils2
DATA_ROOT_FOLDER=/data/id01/inhouse/data/IHR/ihhc3586/id01/detector/2021_01_21_151602_platinum
SPECFILE_PATH=/data/id01/inhouse/data/IHR/ihhc3586/id01/spec/2021_01_21_151706_platinum.spec

# Parameters related to preprocessing
INTERACT=False
DEBUG=False
BEAMLINE=ID01	
DETECTOR=Maxipix
BINNING="1, 1, 1"
TEMPLATE_DATA_FILE=data_mpx4_%05d.edf.gz
MASK=""

# Parameters related to the phase retrieval
SUPPORT=auto
SUPPORT_THRESHOLD=(0.30 0.40)
SUPPORT_THRESHOLD_METHOD=rms # max
PSF=True
NB_RUNS=45
NB_RUNS_KEEP=15
ZERO_MASK=False
FILTER=skip

# Parameters related to the mode decomposition
NB_RUNS_TO_DECOMPOSE=5  #"all"
CRITERION=std

# Parameters related to the postprocessing
SDD=0.92
ENERGY=12994
FLIP_RECONSTRUCTION=False


# You might want to skip some of the processing parts
PREPROCESSING=false
PHASE_RETRIEVAL=false
MODE_DECOMPOSITION=true
POSTPROCESSING=true

# Choose what gpu machine to use for phase retrieval
GPU_MACHINE="slurm" # "lid01gpu1"

	
# Scans to be processed

# Deionised water P1
# SCANS=(251 252 253 254 257 260 263 266 293 298 300 302)
# SCANS=(253)

# # HCl04
# # cycle 1
# SCANS=(322 325 328 332)
# # cycle 2
# SCANS=(340 344 348 352)

# # H2S04
# # cycle 1
# SCANS=(366 370 378)

# # cycle 5
# SCANS=(426 430 434 438)

# # H3PO4, cycle 1
# SCANS=(460 464 468 472)	

# # Particle with defect, HCLO4, HCL04, H2S04, H2S04
# SCANS=(336 356 382 402)

SCANS=(366 370 430 434 464 468 472)


# BACK to deionised water
# SCANS=(11 13 14 21 22 25 26 27)


# Allows X11 forwarding if INTERACT or DEBUG are set to True. Otherwise, no
# X11 forwarding.
X11=-X
if [ "$INTERACT" = "False" ] && [ "$DEBUG" = "False" ]; then
	X11=-x
fi


if $PREPROCESSING; then
	
	ssh $X11 atlan@rnice9  << EOF

		echo
		echo "/===================================================\ "
		echo [INFO] Connected to rnice9
		echo "\===================================================/"
		echo
		
		$(typeset -p SCANS)
		$(typeset -f make_pynx_config)
		$(typeset -f make_slurm_config)


		conda activate $BCDI_ENV_PATH
		cd $WORKING_DIRECTORY

		for scan in "\${SCANS[@]}";
		do
			echo [INFO] Preprocessing scan "\$scan"

			args=(
				  --scan \$scan
				  --interact $INTERACT
				  --debug $DEBUG
				  --data-root-folder $DATA_ROOT_FOLDER
				  --specfile-path $SPECFILE_PATH
				  --beamline $BEAMLINE
				  --detector $DETECTOR
				  --template-data-file $TEMPLATE_DATA_FILE
				  --output-dir $OUTPUT_DIRECTORY
				  # --mask $MASK
				  --binning "$BINNING"
				  --filter $FILTER
				  )
			python preprocess_bcdi.py "\${args[@]}"
				
			make_pynx_config results/S"\$scan"/pynxraw $SUPPORT "${SUPPORT_THRESHOLD[@]}" $SUPPORT_THRESHOLD_METHOD $PSF $NB_RUNS $NB_RUNS_KEEP $ZERO_MASK $SDD

			make_slurm_config $WORKING_DIRECTORY/results/S"\$scan"/pynxraw

			echo [INFO] Files saved in $WORKING_DIRECTORY/results/S\$scan/pynxraw
			echo
		done

		conda deactivate

		exit

EOF
fi


if $PHASE_RETRIEVAL; then
    if [ $GPU_MACHINE = "slurm" ]; then

        ssh atlan@slurm-access << EOF


            echo
            echo "/===================================================\ "
            echo [INFO] Connected to slurm
            echo "\===================================================/"
            echo

            $(typeset -p SCANS)

            for scan in "\${SCANS[@]}";
            do
                echo [INFO] Phase retrieval on scan \$scan

                cd $WORKING_DIRECTORY/results/S"\$scan"/pynxraw

                sbatch pynx-id01cdi.slurm

                echo [INFO] Job submitted
                echo
            done

            squeue -u $USER

            exit

EOF
    elif [ $GPU_MACHINE = "lid01gpu1" ]; then
        ssh atlan@lid01gpu1 << EOF

		echo
		echo "/===================================================\ "
		echo [INFO] Connected to lid01gpu1
		echo "\===================================================/"
		echo

		$(typeset -p SCANS)

		source /sware/exp/pynx/devel.debian9/bin/activate

		for scan in "\${SCANS[@]}";
		do
        
        done
        
        deactivate

        
EOF
	fi
fi


if $MODE_DECOMPOSITION; then

	if [ "$NB_RUNS_TO_DECOMPOSE" != "all" ]; then
		ssh atlan@rnice9 << EOF

		echo
		echo "/===================================================\ "
		echo [INFO] Connected to rnice9
		echo "\===================================================/"
		echo

		$(typeset -p SCANS)

		conda activate $BCDI_ENV_PATH
		cd $WORKING_DIRECTORY

		for scan in "\${SCANS[@]}";
		do
			echo [INFO] finding the $NB_RUNS_TO_DECOMPOSE best results of scan \$scan
			python find_best_candidates.py -s \$scan -n $NB_RUNS_TO_DECOMPOSE -c $CRITERION
			echo
		done

		conda deactivate

		exit

EOF
	fi

	ssh atlan@lid01gpu1 << EOF

		echo
		echo "/===================================================\ "
		echo [INFO] Connected to lid01gpu1
		echo "\===================================================/"
		echo

		$(typeset -p SCANS)

		source /sware/exp/pynx/devel.debian9/bin/activate
		# source /sware/exp/pynx/devel.p9/bin/activate

		for scan in "\${SCANS[@]}";
		do
			echo [INFO] Mode decomposition on scan \$scan
			cd $WORKING_DIRECTORY/results/S"\$scan"/pynxraw

			if [ "$NB_RUNS_TO_DECOMPOSE" = "all" ]; then

				pynx-cdi-analysis.py *Run*.cxi modes=1
			else
				pynx-cdi-analysis.py candidate_*.cxi modes=1
			fi
			echo
		done

		exit

EOF
fi


if $POSTPROCESSING; then
	ssh -x atlan@rnice9 << EOF

		echo
		echo "/===================================================\ " 
		echo [INFO] Connected to rnice9
		echo "\===================================================/"
		echo

		$(typeset -p SCANS)

		conda activate $BCDI_ENV_PATH
		cd $WORKING_DIRECTORY

		for scan in "\${SCANS[@]}";
		do
			echo [INFO] Postprocessing on scan \$scan
			echo [INFO] First, retrieving motor positions...


			# Fetching out-of-plane, grazing, in-plane, rocking
			# angles and angle step from .spec file.

			OUTOFPLANE_INCIDENCE_INPLANE_ROCKINGANGLE_ANGLESTEP="\$(python - << END

beamline = "$BEAMLINE"

if beamline == "ID01":

	import silx.io

	specfile = silx.io.open("$SPECFILE_PATH")
	positioners = specfile["\$scan" + ".1/instrument/positioners"]

	outofplane_angle = positioners["del"][...] # delta, outofplane detector angle
	incidence_angle = positioners["eta"][...] # eta, incidence sample angle
	inplane_angle = positioners["nu"][...] # nu, inplane detector angle
	azimuth_angle = positioners["phi"][...] # phi, azimuth sample angle

	specfile.close()

	if incidence_angle.shape != ():
		angle_step = (incidence_angle[-1] - incidence_angle[0]) / incidence_angle.shape[0]
		incidence_angle = (incidence_angle[-1] + incidence_angle[0]) / 2
		rocking_angle = "outofplane"

	elif azimuth_angle.shape != ():
		angle_step = (azimuth_angle[-1] - azimuth_angle[0]) / azimuth_angle.shape[0]
		rocking_angle = "inplane"

elif beamline == "SIXS_2019":

	beamline_geometry = "$BEAMLINE_GEOMETRY"

	if beamline_geometry == "MED_V":
	
		import bcdi.preprocessing.ReadNxs3 as nxsRead3

		scan_file_name ="$DATA_ROOT_FOLDER/$TEMPLATE_DATA_FILE"%int("\$scan")
		data = nxsRead3.DataSet(scan_file_name)

		outofplane_angle = data.gamma[0] # gamma, outofplane detector angle
		incidence_angle = data.mu # mu, incidence sample angle
		inplane_angle = data.delta[0] # delta, inplane detector angle
		azimuth_angle = data.omega # omega, azimuth sample angle

		if incidence_angle[0] != incidence_angle[1]:
			rocking_angle = "outofplane"
			angle_step = (incidence_angle[-1] - incidence_angle[0]) / incidence_angle.shape[0]
			incidence_angle = (incidence_angle[-1] + incidence_angle[0]) / 2


		elif azimuth_angle[0] != azimuth_angle[1]:
			rocking_angle = "inplane"
			angle_step = (azimuth_angle[-1] - azimuth_angle[0]) / azimuth_angle.shape[0]
			# azimuth_angle = (azimuth_angle[-1] + azimuth_angle[0]) / 2



print(outofplane_angle, incidence_angle, inplane_angle, rocking_angle, angle_step)

END
)"

			array=(\$OUTOFPLANE_INCIDENCE_INPLANE_ROCKINGANGLE_ANGLESTEP)
			# OUTOFPLANE=\${array[0]}
			# INCIDENCE=\${array[1]}
			# INPLANE=\${array[2]}
			# ROCKINGANGLE=\${array[3]}
			# ANGLESTEP=\${array[4]}

			OUTOFPLANE=\${array[-5]}
			INCIDENCE=\${array[-4]}
			INPLANE=\${array[-3]}
			ROCKINGANGLE=\${array[-2]}
			ANGLESTEP=\${array[-1]}

			echo [INFO] Positions found:
			echo Out of plane angle  = \$OUTOFPLANE°
			echo Incidence angle  = \$INCIDENCE°
			echo Inplane angle = \$INPLANE°
			echo Rocking angle is \$ROCKINGANGLE
			echo Angle step = \$ANGLESTEP

			args=(
	  			  --scan \$scan
	  			  --outofplane-angle \$OUTOFPLANE
	  			  --incidence-angle \$INCIDENCE
	  			  --inplane-angle \$INPLANE 
	  			  --angle-step \$ANGLESTEP
	  			  -sdd $SDD
	  			  --energy $ENERGY
	  			  --modes $OUTPUT_DIRECTORY/S\$scan/pynxraw/modes.h5
	  			  --flip $FLIP_RECONSTRUCTION
	  			  --rocking-angle \$ROCKINGANGLE
	  			  --debug False
	  			  --save-dir $OUTPUT_DIRECTORY/S\$scan
	  			  --beamline $BEAMLINE
				  )

			echo [INFO] Running fast_strain.py ...

			python fast_strain.py "\${args[@]}"	   
			echo
		done

		conda deactivate

		exit

EOF
fi
