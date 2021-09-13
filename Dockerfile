FROM tensorflow/tensorflow

WORKDIR /case-study-cicd-ml-with-jenkins

RUN mkdir docs

COPY requirements.txt .
RUN pip install -r requirements.txt


COPY docs ./docs
COPY dataset ./dataset

COPY train.py ./train.py
COPY test.py ./test.py

RUN python3 train.py