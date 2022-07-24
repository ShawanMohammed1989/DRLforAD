ARG GPU

FROM rayproject/ray-ml:latest"$GPU"

RUN apt-get update
# Serve X Windows to localhost
RUN apt-get install -y x11-xserver-utils
RUN xhost +local: || true

# Box2D is needed for the CarRacing-v0 simulation, gym for standard OpenAI Gym simulations
RUN pip install Box2D gym

# This is needed for OSX clients that have upgraded to Big Sur (can most likely be deleted soon)
RUN pip install pyglet==1.5.11

# This is needed to not display simulations windows, but buffer them for headless operation
RUN apt-get install -y xvfb

# This is needed for some simulations visualizations
RUN apt-get install -y python-opengl

# This installs the cudnn7 header which tensorflow looks for if it runs with GPU support
RUN if [[ -z "$GPU" ]] ; then apt-get install -y libcudnn7 ; fi

COPY requirements.txt .
RUN pip install -r requirements.txt
