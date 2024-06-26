#include <Arduino.h>

/****************************************************************************** 
SparkFun Easy Driver Basic Demo
Toni Klopfenstein @ SparkFun Electronics
March 2015
https://github.com/sparkfun/Easy_Driver

Simple demo sketch to demonstrate how 5 digital pins can drive a bipolar stepper motor,
using the Easy Driver (https://www.sparkfun.com/products/12779). Also shows the ability to change
microstep size, and direction of motor movement.

Development environment specifics:
Written in Arduino 1.6.0

This code is beerware; if you see me (or any other SparkFun employee) at the local, and you've found our code helpful, please buy us a round!
Distributed as-is; no warranty is given.

Example based off of demos by Brian Schmalz (designer of the Easy Driver).
http://www.schmalzhaus.com/EasyDriver/Examples/EasyDriverExamples.html
******************************************************************************/
int x;
int y;

void setup() {

pinMode(10,OUTPUT); // ustaw Pin9 jako PUL

pinMode(13,OUTPUT); // ustaw Pin8 jako DIR

pinMode(9,OUTPUT); // ustaw Pin9 jako PUL

pinMode(8,OUTPUT); // ustaw Pin8 jako DIR

Serial.begin(115200);
Serial.setTimeout(1);

}


//Main loop
void loop() {
  //while(Serial.available()){
  //Serial.readBytes(user_input,5); //Read user input and trigger appropriate function
  digitalWrite(13,HIGH); // ustaw stan wysoki dla określenia kierunku
  digitalWrite(8,LOW); // BESTEMMER RETNING
  //Serial.write("off");
  //if(Serial.available() >0){
  while (!Serial.available());
    // got at least 1 byte, read it & act on it
    //String Input_serial = Serial.read();
    y=Serial.readString().toInt();
    Serial.print(y + 1);
    if(y==1){
      // perform some action
      Serial.write(Serial.read());
      Serial.print(y + 2);
      for(x = 0; x < 6000; x++) //

        {

        digitalWrite(10,HIGH);
        if (x <=4000){
          if ( x % 3 == 0){
            digitalWrite(9,HIGH);
          }}
        delayMicroseconds(250);

        digitalWrite(10,LOW);
        if (x <=4000){
        if ( x % 3 == 0){
          digitalWrite(9,LOW);
        }}

        delayMicroseconds(250);

        }
        delay(5);
      }
    else{
      Serial.write(Serial.read());
    }
  } // end checking if serial data is available

  //}


