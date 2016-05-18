

## Arguments
##  `source_dir` is a temporary directory to be mounted to docker container
#
# source_dir="/path/to/run_1"
# model_dir="/path/to/model"
# image_dir="/path/to/images"
# outfile="alexnet.pk"
# blobs="conv1,pool1,norm1,conv2,pool2,norm2,\
#        conv3,conv4,conv5,pool5,fc6,fc7,fc8,prob"

out_dir=$1
model_dir=$2
image_dir=$3

# If no blobs argument, use all alexnet blobs 
if [[ ! $variable ]]; then
    blobs="conv1,pool1,norm1,conv2,pool2,norm2,conv3,conv4,conv5,pool5,fc6,fc7,fc8,prob"
else
    blobs=$4    
fi

# echo $blobs | tee $out_dir"/blobs.txt"

# Make source directory, and link models/images to this
mkdir -p $out_dir
# ln -s $model_dir $out_dir"/model"
# ln -s $image_dir $out_dir"/images"


# Run docker image ebigelow/savedeep
docker run -i --name savedeep \
    -v $out_dir:/opt/sdvo -v $model_dir:/opt/sdvm -v $image_dir:/opt/sdvi \
    tleyden5iwx/caffe-cpu-master << COMMANDS
cd /opt
git clone https://github.com/ebigelow/save-deep.git
chown -R $(id -u):$(id -u) /opt/save-deep
cd /opt/save-deep
python save.py -m /opt/sdvm -i /opt/sdvi -o /opt/sdvo/data.pk -b $blobs
COMMANDS


# Delete the temporary stuff we set up in the output directory
# rm $out_dir"/model"
# rm $out_dir"/images"


# Remove docker image
# docker rm savedeep



# # Local variables
# dvol=$source_dir"/volume"               # Save data volume stuff here
# dcont="savedeep/sdvc:v1"                # Data volume container
# dinst="sdvc"                            # Data volume container instance

# rcont="ebigelow/save-deep"
# rinst="savedeep"



# mkdir -p $dvol
# # TODO TODO!
# cat > $dvol"/Dockerfile"
# "
# # Dockerfile that modifies oraclelinux:6.6 to create a data volume container
# FROM oraclelinux:6.6
# RUN mkdir -p "$source_dir"
# VOLUME "$source_dir"
# ENTRYPOINT /usr/bin/tail -f /dev/null
# "

# # Build data volume
# docker build --tag=$dcont --file=$dvol
# docker run -d --name $dinst $dcont tail -f /dev/null

# # Run with data volume mounted
# docker run -d --volumes-from $dinst --name $rinst -P $rcont 



