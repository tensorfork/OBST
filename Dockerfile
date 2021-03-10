FROM python:3.7
WORKDIR /jannet
ADD requirements.txt .
RUN python3 -m pip install -r requirements.txt
ENTRYPOINT ["/bin/bash"]
