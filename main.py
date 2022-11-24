from fastapi import FastAPI, UploadFile
import uvicorn

from model import classify

app = FastAPI()

@app.get("/")
def read_root():
    return {"message":"WebML"}

'''
@app.post("/classifyURL")
async def predicturl(url: str):
    response = requests.get(url)
    return classify(response.content)
'''

@app.post("/classifyFile/")
async def create_upload_file(file: UploadFile):
    content = await file.read()
    return classify(content)

if __name__=="__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
