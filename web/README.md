## Running
```bash
git clone https://github.com/d1scocat/Tritonchikapp.git
cd Tritonchikapp
npm install
npm run start
```

## Notes
if you use any proxy or other network redirecting application on your device, it might interfere with the internal Vite -- Electron bridge

## API
this app is useless without an api, access it at the respective repository (do this in a different repository):
```bash
git clone https://github.com/ecxqua/Tritonchikeee.git
cd Tritonchikeee
git checkout api
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m database.card_database
python -m database.migrate_dataset
python -m database.build_faiss_index
uvicorn api.entrypoint:app --port 3002
```