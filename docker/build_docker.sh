docker build .\
    --file docker/cu116.Dockerfile \
    --tag opengait:cuda11 \
    --build-arg HOST_USER_GROUP_ARG=$(id -g $USER)