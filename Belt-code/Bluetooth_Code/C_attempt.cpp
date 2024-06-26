#include <iostream>
#include <unistd.h>
#include <bluetooth/bluetooth.h>
#include <bluetooth/rfcomm.h>

int main() {
    const char* target_address = "XX:XX:XX:XX:XX:XX";  // Replace with ESP32 Bluetooth address
    const int port = 1;  // RFCOMM port number

    struct sockaddr_rc addr = { 0 };
    int s, status;
    char buf[1024] = { 0 };
    std::string message = "Hello, ESP32!";

    // allocate a socket
    s = socket(AF_BLUETOOTH, SOCK_STREAM, BTPROTO_RFCOMM);

    // set the connection parameters (who to connect to)
    addr.rc_family = AF_BLUETOOTH;
    addr.rc_channel = (uint8_t)port;
    str2ba(target_address, &addr.rc_bdaddr);

    // connect to server
    status = connect(s, (struct sockaddr*)&addr, sizeof(addr));

    if (status == 0) {
        // send data
        send(s, message.c_str(), message.length(), 0);
    } else {
        std::cerr << "Could not connect to the ESP32 over Bluetooth." << std::endl;
    }

    close(s);
    return 0;
}