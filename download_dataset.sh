#!/bin/bash
set -e

image_docker=physionet
dataset_docker=dataset

docker build -t ${image_docker} - <<EOF
FROM alpine:latest

RUN apk add --no-cache aws-cli

CMD ["aws", "s3", "sync", "--no-sign-request", "s3://physionet-open/eegmmidb/1.0.0/", "/mnt/local/${dataset_docker}"]
EOF

docker run -it --rm --mount type=bind,src=.,target=/mnt/local physionet
docker image rm ${image_docker}