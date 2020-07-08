FROM tensorflow/tensorflow:2.2.0-gpu as base

WORKDIR /src

COPY requirements.txt ./

RUN pip install -r requirements.txt

COPY . ./

ENV FLASK_APP="index.py"

CMD [ "flask", "run", "--host=0.0.0.0" ]

# for local development
FROM base as dev

RUN apt install curl \
    && curl -sL https://deb.nodesource.com/setup_12.x | bash \
    && apt-get install -y nodejs

RUN npm i nodemon -g

CMD [ "nodemon" ]