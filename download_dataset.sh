#!/bin/bash
set -e

image_docker=physionet
mne_folder=dataset/MNE-eegbci-data/files/eegmmidb/1.0.0

docker build -t ${image_docker} - <<EOF
FROM alpine:3.22

RUN adduser -D -u $(id -u) -g $(id -g) user
RUN apk add --no-cache aws-cli

USER user
WORKDIR /home/user/

CMD ["aws", "s3", "sync", "--no-sign-request", "s3://physionet-open/eegmmidb/1.0.0/", "mount/"]
EOF

mkdir -p ${mne_folder}
docker run --rm --user $(id -u):$(id -g) --mount type=bind,src=$(pwd)/${mne_folder},target=/home/user/mount/ ${image_docker}
docker image rm ${image_docker}