ARG BASE_IMG
FROM $BASE_IMG

COPY . /workspace
WORKDIR /workspace

# Install quatrex.
RUN python3 -m pip install .
