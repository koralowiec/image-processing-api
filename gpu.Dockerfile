FROM tensorflow/tensorflow:2.0.1-gpu

WORKDIR /src

COPY requirements.txt ./

RUN pip install -r requirements.txt

COPY . ./

ENV FLASK_APP="index.py"

CMD [ "flask", "run", "--host=0.0.0.0" ]