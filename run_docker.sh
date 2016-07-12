

# Arguments
# ---------
# - out_dir="/path/to/run_1"                        : Save data to this folder
# - model_dir="/path/to/model"                      : Load model stored here
# - image_dir="/path/to/images"                     : Run model on each image in this folder
# - blobs="conv1,pool1,norm1,conv2,pool2,norm2,\    : Optional (for AlexNet) argument - list of layers to save
#        conv3,conv4,conv5,pool5,fc6,fc7,fc8,prob"
#
#
# Example call
# ------------
# sh run_docker.sh /Users/eric/code/save-deep/docker1 \
#                  activations.pk \
#                  /Users/eric/code/save-deep/models/alexnet \
#                  /Users/eric/code/save-deep/images \
#                  "fc6,fc7,fc8,prob"
#

out_dir=$1
out_file=$2
model_dir=$3
image_dir=$4

# If no blobs argument, use all alexnet blobs 
if [[ ! $5 ]]; then
    blobs="conv1,pool1,norm1,conv2,pool2,norm2,conv3,conv4,conv5,pool5,fc6,fc7,fc8,prob"
else
    blobs=$5
fi

# Make source directory, and link models/images to this
mkdir -p $out_dir

# Download docker image `floydhub/dl-docker:cpu`
docker pull floydhub/dl-docker:cpu

# Run docker image `savedeep`:
#   - mount output, model, and image directories
#   - clone `ebigelow/save-deep` github repo to image
#   - run `save.py` with specified blobs
docker run -i --name savedeep -p 8888:8888 -p 6006:6006 \
    -v $out_dir:/root/sdvo  -v $model_dir:/root/sdvm -v $image_dir:/root/sdvi \
    floydhub/dl-docker:cpu bash << COMMANDS
cd /opt
git clone https://github.com/ebigelow/save-deep.git
chown -R $(id -u):$(id -u) /root/sdvm
chown -R $(id -u):$(id -u) /root/sdvi
chown -R $(id -u):$(id -u) /root/sdvo
cd /opt/save-deep
python save.py -m /root/sdvm -i /root/sdvi -o /root/sdvo/activations.pk -b $blobs
COMMANDS

# Clean up docker image
docker rm savedeep
