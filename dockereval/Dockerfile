FROM ubuntu:focal

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y r-base

RUN apt-get install -y python3.10
RUN apt-get install -y pip
RUN pip install pandas numpy argparse


RUN Rscript -e "install.packages('optparse')"
RUN Rscript -e "install.packages('jsonlite')"
RUN Rscript -e "install.packages('Metrics')"

COPY validate.py /usr/local/bin/.
COPY score.R /usr/local/bin/.

RUN chmod +x /usr/local/bin/validate.py && chmod +x /usr/local/bin/score.R
