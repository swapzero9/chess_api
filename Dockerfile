FROM python:3.9

WORKDIR "/usr/src/app"

RUN apt-get update

RUN apt-get install -y curl
RUN apt-get install -y vim

COPY . ./api
RUN mkdir api/logs
RUN pip install --no-cache-dir -r api/requirements.txt

RUN git clone https://github.com/official-stockfish/Stockfish.git && \
    cd Stockfish/src && \
    make help && make net && \ 
    make build ARCH=armv8 && \
    make strip && \
    mv stockfish ../../api/engines/stockfish_engine/stockfish

COPY ./engines/minimax_engine/replace_init.py /usr/local/lib/python3.9/site-packages/chess/__init__.py

EXPOSE 8000

CMD ["uvicorn", "--reload", "api.main_api:app", "--host", "0.0.0.0"]
