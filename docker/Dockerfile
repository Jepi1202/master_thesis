FROM python:3.11.9

#RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
#RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

RUN apt-get update && apt-get install -y git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 libgl1-mesa-glx\
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install cython
RUN pip install xtcocotools

RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-2.2.2+cu118.html
RUN pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-2.2.2+cu118.html
RUN pip install torch-geometric

RUN pip install pysr
RUN pip install numpy
RUN pip install matplotlib
RUN pip install jupyter
RUN pip install opencv-python

EXPOSE 8888

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]