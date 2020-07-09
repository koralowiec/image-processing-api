FROM tensorflow/tensorflow:2.2.0

WORKDIR /src

COPY requirements.txt ./

RUN pip install -r requirements.txt

COPY . ./

ENV FLASK_APP="code/index.py"

CMD [ "flask", "run", "--host=0.0.0.0" ]