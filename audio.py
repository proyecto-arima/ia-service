from pathlib import Path
import openai
import boto3

from dotenv import load_dotenv
import os
from uuid import uuid4

from pymongo import MongoClient

load_dotenv()
s3_client = boto3.client('s3', aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"), aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"))

try:
    client = MongoClient(os.getenv("MONGODB_URI"))
    client.admin.command("ping")
    print("Connected to MongoDB")
except:
    print("Failed to connect to MongoDB")
    exit()


def generate_audio(text):
    return openai.audio.speech.create(
      model="tts-1-hd",
      voice="nova",
      input=text
    ).read()

def update_content_status(id, status):
    collection.update_one({"_id": id}, {"$set": {"status": status}})
    
def upload_to_s3(buffer):
    key = f'{os.getenv("PREFIX")}/{uuid4()}.mp3'
    response = s3_client.put_object(Bucket=os.getenv("PUBLIC_BUCKET"), Key=key, Body=buffer, ContentType="audio/mpeg")
    return f'https://{os.getenv("PUBLIC_BUCKET")}.s3.amazonaws.com/{key}'

db = client['adaptaria']
collection = db['contents']
status = ["PENDING_AUDIO", "FAILED_AUDIO"]
result = collection.find_one({"status": {"$in": status}}, {"_id": 1, "generated": 1, "status": 1}, sort=[('updatedAt', 1)])

if result is None:
    print("No content found")
    client.close()
    exit(0)

print("Processing audio", result["_id"])
update_content_status(result["_id"], "PROCESSING_AUDIO")

texts = None
for generated in result["generated"]:
    if generated["type"] == "SPEECH":
        texts = generated["content"]
        break
    
if texts is None:
    print("No audio texts found")
    client.close()
    exit(0)
    
audios = [upload_to_s3(generate_audio(text["text"])) if text["audioUrl"] == "" else text["audioUrl"] for text in texts ]

for i, url in enumerate(audios):
    texts[i]["audioUrl"] = url
    
result["generated"] = [generated if generated["type"] != "SPEECH" else {"type": "SPEECH", "content": texts, "approved": False} for generated in result["generated"]]
    
collection.update_one({"_id": result["_id"]}, {"$set": {"generated": result["generated"]}})

update_content_status(result["_id"], "DONE")
print("Done")