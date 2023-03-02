import paho.mqtt.client as mqtt
import time 

mqttBroker = "mqtt.eclipseprojects.io"
client = mqtt.Client('SmartPhone')
client.connect(mqttBroker)

def on_message(clien, userdata, message):
    print(f" Receive message: {str(message.payload.decode('utf-8'))}")


client.loop_start()
client.subscribe("TEMPERATURE")

client.on_message = on_message

time.sleep(3)

client.loop_stop()
 