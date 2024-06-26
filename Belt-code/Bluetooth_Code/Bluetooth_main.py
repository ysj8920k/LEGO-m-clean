"""
A simple Python script to receive messages from a client over
Bluetooth using Python sockets (with Python 3.3 or above).


import socket
hostMACAddress = '30:AE:A4:FE:36:E2' # The MAC address of a Bluetooth adapter on the server. The server might have multiple Bluetooth adapters.
channel = 6 # 3 is an arbitrary choice. However, it must match the port used by the client.
s = socket.socket(socket.AF_BLUETOOTH, socket.SOCK_STREAM, 
socket.BTPROTO_RFCOMM)
s.connect((hostMACAddress,channel))
#s_sock = s.accept()
#print ("Accepted connection from "+hostMACAddress)

data = s.recv(1024)
print ("received [%s]" % data)

s.listen(1)
"""

import serial
import time
def belt_activation(message="1",COM="COM4"):
    # Replace 'COMx' with the actual COM port assigned to your USB-to-Serial converter
    serial_port = serial.Serial(COM, 115200, timeout=1)

    # Wait for the serial connection to establish
    time.sleep(2)

    # Encode the string as bytes before sending
    encoded_message = message.encode('utf-8')

    # Send the message
    serial_port.write(encoded_message)

    # Close the serial connection
    serial_port.close()
    print("closed")

while True:

    belt_activation(message="1",COM="COM4")
    time.sleep(5)
    