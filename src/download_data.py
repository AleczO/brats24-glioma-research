import os
import synapseclient
import synapseutils
from dotenv import load_dotenv

load_dotenv()
AUTH_TOKEN = os.getenv("SYNAPSE_TOKEN")


DATA_FOLDER_ID = "syn64952546" 
LOCAL_DATA_PATH = "./data/raw"

def setup_data():
    syn = synapseclient.Synapse()
    syn.login(authToken=AUTH_TOKEN)
    
    if not os.path.exists(LOCAL_DATA_PATH):
        os.makedirs(LOCAL_DATA_PATH)
        
    print(f"Starting download from Synapse ({DATA_FOLDER_ID})...")
    synapseutils.syncFromSynapse(syn, DATA_FOLDER_ID, path=LOCAL_DATA_PATH)
    print("Download complete!")

if __name__ == "__main__":
    setup_data()