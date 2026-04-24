from api import factory

app = factory.make_app()

# Launch with:
# uvicorn api.entrypoint:app --workers <amount of workers>
