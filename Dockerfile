# Local wslc deploys use 7nt/Dockerfile.elo (git clone + start-time git pull).
# This file remains for ad-hoc image builds from the Elo_Ratings repo context.
FROM python:3.12-slim

RUN apt-get update && apt-get install -y git && apt-get clean

WORKDIR /app

RUN git clone https://github.com/BSalita/Elo_Ratings.git .
RUN git clone https://github.com/BSalita/mlBridge.git mlBridge

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8505 8507 8509

CMD ["python", "-c", "import sys; sys.exit('Specify a service command via wslc run (see 7nt/elo_ratings_start.ps1)')"]
