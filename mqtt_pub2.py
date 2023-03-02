import paho.mqtt.client as mqtt
from random import randrange, uniform
import time 

mqttBroker = "mqtt.eclipseprojects.io"
client = mqtt.Client('Temp_Outside')
client.connect(mqttBroker)

while True:
    rand = uniform(0.0, 10.0)
    client.publish("TEMPERATURE", rand)
    print(f'published outter { rand}')
    time.sleep(1)