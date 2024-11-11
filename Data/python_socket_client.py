import socketio
from threading import Event
import time
import json 

BASE_URL = 'https://gss.alertnepal.online/'

sio = socketio.Client()
data_received = Event()
collected_data = []


def write_to_json(data, filename='precipitation_data.json'):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

sio.connect(BASE_URL)

@sio.on('precipitation')
def handle_response(response):
    # print(response)
    print("Received data", response)
    collected_data.append(response)

    data_received.set()

sio.emit('client_request',{
    'name_space':'precipitation', 
 '   location': {
        'position': {
                'latitude':  26.92777778, 
                'longitude': 87.94444444  
            },
        'radius': 20000
    }
})

data_received.wait()

print(collected_data)

write_to_json(collected_data)
