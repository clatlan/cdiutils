# username that will be used to connect to the different remote hosts 
USERNAME = "atlan"

# where to find the key file that it is needed to connect to remote hosts
# whithout password
KEYFILENAME = "/users/atlan/.ssh/id_rsa"

# where to find the pynx script for phase retrieval on slurm p9 gpus
PYNX_SLURM_FILE_TEMPLATE = (
    "/data/id01/inhouse/clatlan/pythonies/cdiutils/cdiutils/bcdi/"
    "pynx-id01cdi_template.slurm"
)

# the number of reconstructions to keep after running pnyx phase
# retrieval
NB_OF_PYNX_RECONSTRUCTIONS_TO_KEEP = 3