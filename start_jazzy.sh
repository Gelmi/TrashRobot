sudo docker run -it \
  --env="DISPLAY" \
  --env="QT_X11_NO_MITSHM=1" \
  --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
  -u $(id -u):$(id -g) \
  --volume="$HOME/Docker/Jazzy/dev_ws:/root/dev_ws" \
  --env="XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR" \
  --device=/dev/dri:/dev/dri \
  --name ubuntu-jazzy \
  --hostname ros2-jazzy \
  --net=host \
  --ipc=host \
  osrf/ros:jazzy-desktop-full
