FROM anibali/pytorch:cuda-9.1

RUN ["sudo", "apt-get", "install", "-y", "libsm6", "libxext6"]

WORKDIR /facial-recognition

ADD requirements.txt .

RUN ["pip", "install", "-r", "requirements.txt", "--disable-pip-version-check", "--ignore-installed"]

RUN ["pip", "install", "--upgrade", "--force-reinstall", "numpy"]
RUN ["pip", "install", "--upgrade", "--force-reinstall", "setuptools"]

COPY *.py ./
