FROM python:3.9-slim

WORKDIR /app

COPY ./requirements/requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY ./app /app

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]

# docker build -t guimaraesabri/data-agent:develop -f deployment/docker/Dockerfile .
# docker tag data-agent:develop guimaraesabri/data-agent:develop
# docker rmi data-agent:develop
# docker push guimaraesabri/data-agent:develop 
