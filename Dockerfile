# Pull tensorflow image with Python 3

FROM tensorflow/tensorflow:2.3.0rc0-py3

# Set the working directory to /app
WORKDIR /app

# Transfer content from current dir to / app in contaainer
ADD . /app

# Install audio libraries
RUN apt-get update && apt-get install -y libsndfile1 libasound-dev portaudio19-dev libportaudio2 libportaudiocpp0 ffmpeg

# Install python packages
RUN pip install -r requirements.txt

# Start uWSGI using config file
CMD [ "uwsgi", "app.ini" ]