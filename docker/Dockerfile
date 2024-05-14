FROM python:3.9.1-slim-buster

RUN pip install pandas numpy sklearn
RUN apt-get update && apt-get install -y --no-install-recommends r-base r-base-dev && rm -rf /var/lib/apt/lists/*
COPY requirements.r .
RUN Rscript requirements.r

COPY validate.py /usr/local/bin/.
COPY score.R /usr/local/bin/.
RUN chmod +x /usr/local/bin/validate.py && chmod +x /usr/local/bin/score.R
