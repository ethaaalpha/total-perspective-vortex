#!/bin/bash
docker run -it --rm \
    --mount type=bind,src=./,dst=/app \
    --workdir /app \
    -e DISPLAY=:0 \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    python:3.11.13-bookworm \
    bash