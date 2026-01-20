import time
import random
import json
import string 
from urllib.parse import quote
import uuid
import requests
from curl_cffi import requests as cffi_requests
from PIL import Image
import base64
from predictor import cnn


def url_encode(data):
    return quote(str(data))

def uuidv4():
    return str(uuid.uuid4())

def current_time():
    return int(time.time())


TSH = "TH[cba9a83bbafbda3aa35158aa08d04e4c]"
TSH1 = url_encode(TSH)
BD = "www.sfr.fr"
MTPublicKey = "MTPublic-YVOJkLMVR"
randkey = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
s = uuidv4()


# 1: Setup vars
RT = current_time()
urlstep1 = f'https://service.mtcaptcha.com/mtcv1/api/getchallenge.json?sk=MTPublic-YVOJkLMVR&bd=www.sfr.fr&rt={RT}&tsh={TSH1}&act=%24&ss={s}&lf=0&tl=%24&lg=fr&tp=s'

proxies = {

    "https": f"http://9f6pvo-country-FR-session-{randkey}-time-1:w3nhjsl5@res-v1.nettify.xyz:8080"
}

browsers = ["safari","chrome"]
session = cffi_requests.Session(impersonate=random.choice(browsers))
session.proxies = proxies

r = session.get(f"{urlstep1}")

data = r.json()
ct = data['result']['challenge']['ct']
fseed = data['result']['challenge']['foldChlg']['fseed']
fslots = data['result']['challenge']['foldChlg']['fslots']
fdepth = data['result']['challenge']['foldChlg']['fdepth']
kt = "$$"

print(f"CT: {ct[:10]}... | Seed: {fseed[:10]}... | Slots: {fslots} | Depth: {fdepth} | KT: {kt}")


ct_enc = ct.replace(',', '%2C')

# Get image (req2)
urlstep2 = f'https://service.mtcaptcha.com/mtcv1/api/getimage.json?sk=MTPublic-YVOJkLMVR&ct={ct_enc}&fa=%24&ss={s}'
b = session.get(f"{urlstep2}")
data2 = b.json()

if 'result' not in data2 or 'img' not in data2['result'] or 'image64' not in data2['result']['img']:
    print("Captcha not found due to fingerprint being banned")
    exit()

image64 = data2['result']['img']['image64']
image = base64.b64decode(image64)

with open("captcha.gif", "wb") as f:
    f.write(image)
print("Captcha saved as captcha.gif")

# Calculate FA
print("Calculating FA...")  
try:
    calc_payload = {
        "fseed": fseed,
        "fslots": fslots,
        "fdepth": fdepth
    }
    
    calc_resp = requests.post("http://localhost:9091/calculate", json=calc_payload)
    if calc_resp.status_code == 200:
        fa = calc_resp.json().get("fa")
        print(f"Calculated FA: {fa}")
    else:
        print(f"Failed to calculate FA: {calc_resp.text}")
        fa = "$"
except Exception as e:
    print(f"Error calling calculate: {e}")
    fa = "$"

# Solve
print("Sending Captcha image to CNN...")
ans = cnn("captcha.gif")
print(f"CNN Answer: {ans}")

def enc(x): return url_encode(x)
    
solveUrl = f"https://service.mtcaptcha.com/mtcv1/api/solvechallenge.json?ct={enc(ct)}&sk={enc(MTPublicKey)}&st={enc(ans)}&lf=0&bd={enc(BD)}&rt={enc(RT)}&tsh={TSH1}&fa={enc(fa)}&qh=%24&act=%24&ss={enc(s)}&tl=%24&lg=fr&tp=s&kt={enc(kt)}&fs={enc(fseed)}"

print("Solving...")
solve_resp = session.get(solveUrl)


solveData = solve_resp.json()
verifyResult = solveData.get("result", {}).get("verifyResult", {})
isVerified = verifyResult.get("isVerified")

print(f"Valid Cnn answer ? : {isVerified}")
if isVerified:
    vt = verifyResult.get("verifiedToken", {}).get("vt")
    print(f"Captcha Answer: {vt}")
else:
    print("Captcha not verified cos of cnn :(")
