FROM agrigorev/zoomcamp-model:mlops-3.9.7-slim

WORKDIR /app
COPY ["Pipfile", "Pipfile.lock", "predict_fhv_trip.py", "./"]

RUN pip install -U pip
RUN pip install pipenv
RUN pipenv install --system --deploy

ENTRYPOINT ["python3"]
CMD ["predict_fhv_trip.py"]
