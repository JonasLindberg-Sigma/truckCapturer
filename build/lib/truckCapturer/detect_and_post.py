from anprmodule.predict import run
from requests import post

def form_json(reg):
    return {}

def detect_and_post(img, url):
    reg = run(src=img, model='best.pt')