# AI-Based-Energy-Management-System Research
AI Based Energy Management System.
The System is an Energy Management module running on Tensorflow, omron sensor and intelligent switch. 
This repository contains the trained dataset and the model for the research.
The system is able to sense the body heat map of humans, and switches the lightings and connected power sources when the individual gets into the room, while turning it off on leaving the room. 
This helps to effectively manage the energy consumption rate.

# Keywords
Energy Management, Sustainable Energy, Home Automation, Machine Learning, Thermal Sensing, Heat Map

# Required Components and Codes

# Omron D6T-44L: 
Senses the heat map of individuals coming into a room
# Arduino: 
Reads sensor data from the Omron D6T-44L sensor and sends output through the serial port
# Dataset : 
The Dataset contains the data collected from the omron sensor through out a 24 hrs period
It also contains data collected at various times i.e morning, afternoon and night to give allowance for the effect of ambient temparature on the human body.
The datasets were collected when humans are in the room and when there are no humans in the room

The Data is collected and cleaned to remove outliners, it is then modeled using neural network algorithm with Keras and Tensorflow at the backend
