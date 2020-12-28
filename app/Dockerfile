FROM jupyter/scipy-notebook

USER root

COPY requirements.txt ./
RUN pip install -r requirements.txt  && rm requirements.txt

#RUN pip install torch==1.7.0+cu101 torchvision==0.8.1+cu101 torchaudio==0.7.0 -f https://download.pytorch.org/whl/torch_stable.html

ENV WORK_DIR ${HOME}/work