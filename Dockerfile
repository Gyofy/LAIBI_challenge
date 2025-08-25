FROM pytorch/pytorch

RUN apt-get update && \
    apt-get install -y gcc && \
    rm -rf /var/lib/apt/lists/*

RUN groupadd -r algorithm && useradd -m --no-log-init -r -g algorithm algorithm

RUN mkdir -p /opt/algorithm /input /output \
    && chown algorithm:algorithm /opt/algorithm /input /output

USER algorithm

WORKDIR /opt/algorithm

ENV PATH="/home/algorithm/.local/bin:${PATH}"

RUN python -m pip install --user -U pip

COPY --chown=algorithm:algorithm requirements.txt /opt/algorithm/

# Install required python packages via pip
RUN python -m pip install --user -r requirements.txt

# Copy all source code and resources
COPY --chown=algorithm:algorithm . /opt/algorithm/

# Entrypoint to python inference
ENTRYPOINT ["python", "inference.py"]