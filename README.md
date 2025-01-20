# Projet_ML1

## How to use

- Run Docker Desktop
- Execute (First time only, then no flags needed):

```bash
docker compose -f serving/docker-compose.yml up --build --force-recreate
```

```bash
docker compose -f webapp/docker-compose.yml up --build --force-recreate
```

```bash
docker compose -f reporting/docker-compose.yml up --build --force-recreate
```

- Webapp : http://localhost:8081/
- Reporting : http://localhost:8082/
- Serving API : http://localhost:8080/docs

Run if network issue between API and Webapp (run serving-api first):

```bash
docker compose down webapp/docker-compose.yml
```

If the reporting displays a white blank page or can't connect to localhost after a docker restart, try deleting localhost cookies.

## Description

- User can upload .wav files or record themselves
- User can give feedback on prediction
- Every 10 feedbacks, model is trained on reference data and production data.
- The new model is used only if it performs better than the last.

## Data description

[Dataset](https://zenodo.org/records/4405783)

The corpus was recorded in the 6 basic emotions plus neutral (C=colère, T=tristesse, J=joie, P=peur, D=dégoût, S=surprise, N=neutre).

32 speakers (8 female and 24 male), between the ages of 22 and 27
