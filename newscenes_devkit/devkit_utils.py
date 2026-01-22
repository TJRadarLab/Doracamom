import json 

def read_json(path):
    with open(path, 'r') as f:
        return json.load(f)


    