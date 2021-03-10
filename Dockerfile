FROM python:3.7
WORKDIR /jannet
ADD requirements.txt .
RUN python3 -m pip install -r requirements.txt
RUN apt-get update && apt-get install python3-opencv -y --fix-missing
ENTRYPOINT ["/bin/bash"]
