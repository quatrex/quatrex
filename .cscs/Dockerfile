ARG BASE_IMG
FROM $BASE_IMG

COPY . /opt/repo
WORKDIR /opt/repo

# Install quatrex into the virtual environment.
RUN . /opt/venv/bin/activate && python3 -m pip install . --no-deps
