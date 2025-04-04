import wave
import pyaudio
import cv2 
import mediapipe as mp 
from math import hypot 
import numpy as np 

song = wave.open("songs/Nujabes - Lady Brown (feat. Cise Starr).wav", "rb")
p = pyaudio.PyAudio()

speed = 1.0


## initializing hand tracking stuff
mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode = False, model_complexity = 1, min_detection_confidence = 0.75, min_tracking_confidence = 0.75, max_num_hands = 2)
Draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

## initializing audio stuff
def callback(in_data, frame_count, time_info, status):
    global speed

    frames_to_read = int(frame_count * speed)

    data = song.readframes(frames_to_read)
    if data == b'':
        return None, pyaudio.paComplete
    if speed < 1.0:
        data += b'\x00' * (frame_count - frames_to_read) * song.getsampwidth() * song.getnchannels()

    return data, pyaudio.paContinue

def listen_for_speed():
    global speed
    while True:
            _,frame = cap.read()
            frame = cv2.flip(frame, 1)
            frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            Process = hands.process(frameRGB)
            landmarkList = []

            if Process.multi_hand_landmarks:
                 for handlm in Process.multi_hand_landmarks:
                    for id, lm in enumerate(handlm.landmark):
                        h, w, c = frame.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        landmarkList.append([id, cx, cy])
                    Draw.draw_landmarks(frame, handlm, mpHands.HAND_CONNECTIONS)
            if landmarkList:
                x1, y1 = landmarkList[8][1], landmarkList[8][2]
                x2, y2 = landmarkList[12][1], landmarkList[12][2]
                cv2.circle(frame, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
                cv2.circle(frame, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
                cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
                L = hypot(x2 - x1, y2 - y1)
                if L is not None:
                    new_speed = np.interp(L,[15, 220], [0.1, 5])
                else: 
                    new_speed = 1.0
                speed = max(0.1, new_speed)
            if frame is None:
                print("Warning: Received an empty frame!")
            else:
                cv2.imshow('Image', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

stream = p.open(format=p.get_format_from_width(song.getsampwidth()),
                channels=song.getnchannels(),
                rate=song.getframerate(),
                output=True,
                stream_callback=callback)

stream.start_stream()

while stream.is_active():
    pass
stream.stop_stream()
stream.close()
song.close()
p.terminate()