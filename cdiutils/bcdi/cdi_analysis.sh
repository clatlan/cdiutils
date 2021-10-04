#!/bin/bash

# You might want to skip some of the processing parts
PREPROCESSING=false
PHASE_RETRIEVAL=false
MODE_DECOMPOSITION=true
POSTPROCESSING=true
ERASE_PREVIOUS_ANALYSIS=false

# Required paths
WORKING_DIRECTORY=/data/id01/inhouse/clatlan/experiments/hc4050/analysis
OUTPUT_DIRECTORY=$WORKING_DIRECTORY/results
BCDI_ENV_PATH=/data/id01/inhouse/clatlan/.envs/cdiutils
DATA_ROOT_FOLDER=/data/id01/inhouse/data/IHR/hc4050_a/id01/test/BCDI_2021_07_26_165851/mpx/
SPECFILE_PATH=/data/id01/inhouse/data/IHR/hc4050_a/id01/test/BCDI_2021_07_26_165851/spec/BCDI_2021_07_26_165851.spec
UTILS_FOLDER=/data/id01/inhouse/clatlan/pythonies/cdiutils/cdiutils/bcdi


# Parameters related to preprocessing
TEMPLATE_DATA_FILE=data_mpx4_%05d.edf.gz
SAMPLE_NAME=2nd_specfile/C9/350C/H2
SAMPLE_NAME=3rd_specfile/C17/350C/H2


INTERACT=False
DEBUG=False
BEAMLINE=ID01
DETECTOR=Maxipix
BINNING="1, 1, 1"
MASK=""
ORTHOGONALIZE=False
if [ "$ORTHOGONALIZE" = "True" ]; then
	RESULT_FOLDER=pynx

else
	RESULT_FOLDER=pynxraw
fi


# Parameters related to the phase retrieval
SUPPORT=auto
SUPPORT_THRESHOLD="0.20, 0.25"
SUPPORT_THRESHOLD_METHOD=rms # max
PSF=pseudo-voigt,0.5,0.1,10  #gaussian,0.5,20 #pseudo-voigt,0.5,0.1,10 #gaussian,0.5,20 #lorentzian,0.5,20 #default is pseudo-voigt,1,0.05,20
NB_RUNS=15
NB_RUNS_KEEP=15
ZERO_MASK=False
FILTER=skip


# Parameters related to the mode decomposition
NB_RUNS_TO_DECOMPOSE=5  #"all"
CRITERION=std


# Parameters related to the pre and postprocessing
SDD=0.83
ENERGY=12994
WAVELENGTH=$(python -c "print(1.2398516685506e-06 / $ENERGY)")
ROCKINGANGLE="outofplane"
FLIP_RECONSTRUCTION=true
ISOSURFACE_THRESHOLD=0.5
VOXEL_SIZE=10 # Voxel size in nm

# Choose what gpu machine to use for phase retrieval
MAIN_MACHINE="workstation" #"rnice9"
GPU_MACHINE="p9" # "lid01gpu1"


# Make de dump file directory template, this is where the files created
# during the process will be dumped
DUMP_FILE_DIR_TEMPLATE=$OUTPUT_DIRECTORY/$SAMPLE_NAME/S%d/$RESULT_FOLDER


# Scans to be processed
SCANS=(122)


# Allows X11 forwarding if INTERACT or DEBUG are set to True. Otherwise, no
# X11 forwarding.
X11=-X
if [ "$INTERACT" = "False" ] && [ "$DEBUG" = "False" ]; then
	X11=-x
fi


function make_pynx_config() {

	support_threshold="$3"

	if test -f $1/pynx-cdi-inputs.txt; then
    	rm $1/pynx-cdi-inputs.txt
	fi
	data=($1/*_pynx_*.npz)
	data="${data##*/}"
	mask=($1/*maskpynx*.npz)
	mask="${mask##*/}"

	cat <<EOT>> $1/pynx-cdi-inputs.txt
# parameters
data="$data"
mask="$mask"

data2cxi=True
auto_center_resize=False

#support_type=
#support_size=
support="$2"
support_threshold=${support_threshold[0]}  # pick a random number between these two numbers
support_threshold_method="$4"
support_only_shrink=False
support_update_period=50
support_smooth_width_begin=2
support_smooth_width_end=1
support_post_expand=1,-2,2 #1,-2, 2

#algorithm=(Sup*ER**20)**5 * PSF**100*RAAR**10 * (Sup*RAAR**20)**10 * PSF**100*ER**10 * (Sup*ER**10)**5 * PSF**100*RAAR**10 * (Sup*RAAR**20)**10 * (Sup*ER**10)**5 * (Sup*RAAR**20)**10
psf=$5 #True #False #
nb_raar=1000
nb_hio=100
nb_er=300
nb_ml=0

nb_run=$6
nb_run_keep=$7

zero_mask=$8 # masked pixels will start from imposed 0 and then let free
# max_size=256
crop_output=1 # set to 0 to avoid cropping the output in the .cxi
# if N > 0 it will crop to support range + N pixels

positivity=False
beta=0.9
detwin=False
rebin=1,1,1

# Generic parameters
detector_distance=$9 # in m
pixel_size_detector=55e-6   # eiger 75e-6, maxipix 55e-6
wavelength=${10}
verbose=100
output_format='cxi'  # 'npz'
live_plot=False
save_plot=False
mpi=run
EOT
}



function make_slurm_config() {

	if test -f $1/pynx-id01cdi.slurm; then
    	rm $1/pynx-id01cdi.slurm
	fi

	cat <<EOT>> $1/pynx-id01cdi.slurm
#!/bin/bash -l
#SBATCH --partition=p9gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --ntasks=4
#SBATCH --gres=gpu:2
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --output=%x-%j.out

env | grep CUDA
scontrol --details show jobs \$SLURM_JOBID |grep RES

cd $1

mpiexec -n \$SLURM_NTASKS /sware/exp/pynx/devel.p9/bin/pynx-id01cdi.py pynx-cdi-inputs.txt

EOT
}


if $ERASE_PREVIOUS_ANALYSIS;then
	echo First removing old data
	for scan in ${SCANS[@]};
	do
	    DUMP_FILE_DIR="$( printf "$DUMP_FILE_DIR_TEMPLATE" $scan)"
		rm -r $DUMP_FILE_DIR
	done
fi


if $PREPROCESSING; then


	for scan in ${SCANS[@]};
	do
		echo [INFO] Preprocessing scan "$scan"

        DUMP_FILE_DIR="$( printf "$DUMP_FILE_DIR_TEMPLATE" $scan)"
		args=(
			  --scan $scan
			  --interact $INTERACT
			  --debug $DEBUG
			  --data-root-folder $DATA_ROOT_FOLDER
			  --specfile-path $SPECFILE_PATH
			  --beamline $BEAMLINE
			  --detector $DETECTOR
			  --template-data-file $TEMPLATE_DATA_FILE
			  --sample-name $SAMPLE_NAME
			  --output-dir $DUMP_FILE_DIR ##$OUTPUT_DIRECTORY
			  # --mask $MASK
			  # --binning "($BINNING)"
			  --filter $FILTER
			  --orthogonalize $ORTHOGONALIZE
			  -sdd $SDD
	  		  --energy $ENERGY
	  		  --rocking-angle $ROCKINGANGLE
			  )

		conda activate $BCDI_ENV_PATH
		fast_bcdi_preprocess_BCDI.py "${args[@]}"
		conda deactivate

		make_pynx_config $DUMP_FILE_DIR $SUPPORT \
		"${SUPPORT_THRESHOLD[@]}" $SUPPORT_THRESHOLD_METHOD $PSF $NB_RUNS \
		$NB_RUNS_KEEP $ZERO_MASK $SDD $USE_RAWDATA $WAVELENGTH

		make_slurm_config $DUMP_FILE_DIR

		echo [INFO] Files saved in $DUMP_FILE_DIR
		echo
	done

fi


if $PHASE_RETRIEVAL; then
	if ! $PREPROCESSING; then
		for scan in ${SCANS[@]};
		do
			make_pynx_config $DUMP_FILE_DIR $SUPPORT \
			"${SUPPORT_THRESHOLD[@]}" $SUPPORT_THRESHOLD_METHOD $PSF $NB_RUNS \
			$NB_RUNS_KEEP $ZERO_MASK $SDD $USE_RAWDATA $WAVELENGTH
		done
	fi

    if [ $GPU_MACHINE = "p9" ]; then

        ssh atlan@slurm-access << EOF


            echo
            echo "/===================================================\ "
            echo [INFO] Connected to slurm
            echo "\===================================================/"
            echo

            $(typeset -p SCANS)

            for scan in \${SCANS[@]};
            do
                echo [INFO] Phase retrieval on scan \$scan

				## cd $OUTPUT_DIRECTORY/S"\$scan"/$RESULT_FOLDER
				cd $WORKING_DIRECTORY/results/$SAMPLE_NAME/S"\$scan"/pynxraw

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

		for scan in \${SCANS[@]};
			do
			    DUMP_FILE_DIR="$(printf "$DUMP_FILE_DIR_TEMPLATE" $scan)"
			    ## cd $OUTPUT_DIRECTORY/S"\$scan"/$RESULT_FOLDER
			    cd $DUMP_FILE_DIR
		        pynx-id01cdi.py pynx-cdi-inputs.txt
		    done

        deactivate

EOF
	fi
fi


if $MODE_DECOMPOSITION; then

	if [ "$NB_RUNS_TO_DECOMPOSE" != "all" ]; then


		conda activate $BCDI_ENV_PATH
		for scan in ${SCANS[@]};
		do
			echo
			echo [INFO] finding the $NB_RUNS_TO_DECOMPOSE best results \
				of scan $scan

			DUMP_FILE_DIR="$(printf "$DUMP_FILE_DIR_TEMPLATE" $scan)"

			args=(
				-s $DUMP_FILE_DIR ##$OUTPUT_DIRECTORY/S$scan/$RESULT_FOLDER
				-n $NB_RUNS_TO_DECOMPOSE
				-c $CRITERION
				)
			python $UTILS_FOLDER/find_best_candidates.py ${args[@]}

		done
		conda deactivate

	fi

	ssh atlan@lid01gpu1 << EOF

		echo
		echo "/===================================================\ "
		echo [INFO] Connected to lid01gpu1
		echo "\===================================================/"
		echo

		$(typeset -p SCANS)

		source /sware/exp/pynx/devel.debian9/bin/activate

		for scan in \${SCANS[@]};
		do
			echo [INFO] Mode decomposition on scan \$scan

			DUMP_FILE_DIR="$(printf "$DUMP_FILE_DIR_TEMPLATE" $scan)"
			cd $DUMP_FILE_DIR

			## cd $OUTPUT_DIRECTORY/S"\$scan"/$RESULT_FOLDER

			if [ "$NB_RUNS_TO_DECOMPOSE" = "all" ]; then

				pynx-cdi-analysis.py *Run*.cxi modes=1 modes_output=modes.h5
			else
				pynx-cdi-analysis.py candidate_*.cxi modes=1 modes_output=modes.h5
			fi
			echo
		done

		deactivate

		exit

EOF
fi


if $POSTPROCESSING; then
	conda activate $BCDI_ENV_PATH
	for scan in ${SCANS[@]};
	do
		echo [INFO] Postprocessing on scan $scan
		echo [INFO] First, retrieving motor positions...

		if [ "$BEAMLINE" = "P10" ]; then
			SPECFILE_PATH="$( printf "%s%s_%05d/%s_%05d.fio" $DATA_ROOT_FOLDER $SAMPLE_NAME $scan $SAMPLE_NAME $scan)"
			echo $SPECFILE_PATH
		fi


		command=$(python $UTILS_FOLDER/get_positions.py \
			--specfile-path $SPECFILE_PATH --scan $scan --beamline $BEAMLINE)
		array=($command)
		# echo array is "${array[@]}"
		# echo subscript ${array[-5]}


		# Values are retrieved in the reverse order so that it will work
		# even if the python code block print out something before the last
		# line
		OUTOFPLANE=${array[-5]}
		INCIDENCE=${array[-4]}
		INPLANE=${array[-3]}
		ROCKINGANGLE=${array[-2]}
		ANGLESTEP="${array[-1]}"

		echo [INFO] Positions found:
		echo Out of plane angle  = $OUTOFPLANE°
		echo Incidence angle  = $INCIDENCE°
		echo Inplane angle = $INPLANE°
		echo Rocking angle is $ROCKINGANGLE
		echo Angle step = $ANGLESTEP

		DUMP_FILE_DIR="$(printf "$DUMP_FILE_DIR_TEMPLATE" $scan)"

		args=(
			  --data-root-folder $DATA_ROOT_FOLDER
	  		  --scan $scan
	  		  --outofplane-angle $OUTOFPLANE
	  		  --incidence-angle $INCIDENCE
	  		  --inplane-angle $INPLANE
	  		  --angle-step $ANGLESTEP
	  		  -sdd $SDD
	  		  --energy $ENERGY
	  		  --modes $DUMP_FILE_DIR/modes.h5
	  		  --flip $FLIP_RECONSTRUCTION
	  		  --rocking-angle $ROCKINGANGLE
	  		  --debug False
	  		  --save-dir $DUMP_FILE_DIR ##$OUTPUT_DIRECTORY/S$scan/$RESULT_FOLDER
	  		  --beamline $BEAMLINE
	  		  --is-orthogonalized $ORTHOGONALIZE
	  		  --specfile-path $SPECFILE_PATH
	  		  --isosurface-threshold $ISOSURFACE_THRESHOLD
	  		  --voxel-size $VOXEL_SIZE
			  )

		echo [INFO] Running fast_strain.py ...
		python /data/id01/inhouse/clatlan/.envs/cdiutils/bin/fast_bcdi_strain.py ${args[@]}

	done
	conda deactivate
fi
