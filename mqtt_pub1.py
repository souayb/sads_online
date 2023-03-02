import paho.mqtt.client as mqtt
from random import randrange, uniform
import time 

mqttBroker = "mqtt.eclipseprojects.io"
client = mqtt.Client('Temp_Inside')
client.connect(mqttBroker)

while True:
    rand = uniform(20.0, 21.0)
    client.publish("Temperature", rand)
    print(f'published { rand}')
    time.sleep(1)