FROM python:3.6.4

WORKDIR /app

COPY . . /app/

RUN pip install -r requirements.txt

EXPOSE 5000

ENV NAME OpentoAll

CMD ["python","app.py"]