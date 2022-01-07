FROM python:slim

WORKDIR "/usr/src/app"

RUN apt-get update

RUN apt-get install -y curl
RUN apt-get install -y vim

COPY ./api ./api

RUN pip install --no-cache-dir -r api/requirements.txt

EXPOSE 8000

CMD ["uvicorn", "--reload", "api.main_api:app", "--host", "0.0.0.0"]
