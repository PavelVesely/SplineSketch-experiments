FROM eclipse-temurin:17.0.16_8-jdk-jammy
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-venv make \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /artifact
COPY requirements.txt .
RUN python3 -m venv /opt/venv && /opt/venv/bin/pip install --no-cache-dir -r requirements.txt
ENV PATH="/opt/venv/bin:${PATH}" MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib
COPY . .
RUN make && python reproduce.py doctor
ENTRYPOINT ["python", "reproduce.py"]
CMD ["run", "--profile", "smoke"]

