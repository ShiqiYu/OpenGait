source_folder="/home/$(whoami)/Git/OpenGait" # Change these to your path
sustech1k_folder="/media/$(whoami)/Files/Data/SUSTech1K" 


xhost local:
docker run -it --rm \
    -e SDL_VIDEODRIVER=x11 \
    -e DISPLAY=$DISPLAY \
    --env='DISPLAY' \
    --gpus all \
    --ipc host \
    --privileged \
    --network host \
    -p 8080:8081 \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v /$source_folder/:/opengait/ \
    -v /$sustech1k_folder/:/SUSTech1K/ \
    opengait:cuda11