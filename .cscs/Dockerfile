ARG BASE_IMG
FROM $BASE_IMG

COPY . /src
WORKDIR /src

ENV PYTHONFAULTHANDLER=1

# Install quatrex.
RUN python3 -m pip install .
