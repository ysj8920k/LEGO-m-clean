import socket
def Bluetooth_activate(message="ON"):
    target_address = '30:AE:A4:FE:36:E2'  # Replace with the Bluetooth address of your ESP32
    port = 1  # RFCOMM port number

    # Create a Bluetooth socket
    sock = socket.socket(socket.AF_BLUETOOTH, socket.SOCK_STREAM, socket.BTPROTO_RFCOMM)
    sock.connect((target_address, port))

    # String to send

    # Send the message
    sock.send(message.encode())

    # Close the Bluetooth socket
    sock.close()

if __name__ == '__main__':
    Bluetooth_activate()