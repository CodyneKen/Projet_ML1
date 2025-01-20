FROM python:3.12-slim

COPY project.py .

COPY requirements.txt .

RUN pip3 install --upgrade pip

RUN pip3 install -r requirements.txt

CMD python project.py && evidently ui --host 0.0.0.0 --port 8082

# CMD ["tail", "-f", "/dev/null"]