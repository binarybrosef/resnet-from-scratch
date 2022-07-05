FROM python:3.9

ADD model.py .

RUN pip install tensorflow

CMD [ "python", "./model.py" ]

