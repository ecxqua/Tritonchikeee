from api import factory

app = factory.make_app()

# Launch with:
# uvicorn app.entrypoint:app --workers <amount of workers>