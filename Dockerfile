FROM pytorch/pytorch:latest

RUN git clone https://github.com/lancopku/simNet.git
RUN git clone https://github.com/tylin/coco-caption.git coco
RUN 2to3 -w simNet/ && 2to3 -w coco/
ENV PYTHONPATH /workspace/
RUN pip install Cython
RUN pip install nltk matplotlib scikit-image
COPY docker_setup.py /
RUN python /docker_setup.py
RUN chmod -R 777 /workspace/
# RUN apt-get update && apt-get install -y vim
