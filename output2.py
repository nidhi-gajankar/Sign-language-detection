import pickle
import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from tkinter import Label, Button
from PIL import Image, ImageTk

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {   0:'1', 1:'2',2:'Y',3:'4',4:'5',
                  5:'6',6:'7',7:'8',8:'9',9:'10',
                  10: 'L',11: 'hello', 12: 'love you',
                  13:'A',14:'C',15:'O',16:'Q',
               }

is_running = False
predicted_character = ""


def update_frame():
    global predicted_character
    if is_running:
        data_aux = []
        x_ = []
        y_ = []

        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            return

        H, W, _ = frame.shape

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,  # image to draw
                    hand_landmarks,  # model output
                    mp_hands.HAND_CONNECTIONS,  # hand connections
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10

            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10

            # Adjust the input feature vector to have the correct number of features
            if len(data_aux) == model.n_features_in_:
                prediction = model.predict([np.asarray(data_aux)])
                predicted_character = labels_dict[int(prediction[0])]

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                            cv2.LINE_AA)

        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=img)
        lmain.imgtk = imgtk
        lmain.configure(image=imgtk)
        lmain.after(10, update_frame)

        # Update the predicted character label
        pred_label.config(text=f"Predicted Character: {predicted_character}")


def start_video():
    global is_running
    is_running = True
    update_frame()


def stop_video():
    global is_running
    is_running = False


root = tk.Tk()
root.title("Sign Language Recognition")

# Main video stream window
lmain = Label(root)
lmain.pack()

# Frame for buttons
button_frame = tk.Frame(root)
button_frame.pack(pady=10)

start_button = Button(button_frame, text="Start", command=start_video, width=10, height=2)
start_button.pack(side="left", padx=5)

stop_button = Button(button_frame, text="Stop", command=stop_video, width=10, height=2)
stop_button.pack(side="right", padx=5)

# Frame for the predicted character label
pred_frame = tk.Frame(root)
pred_frame.pack(pady=10)

pred_label = Label(pred_frame, text="Predicted Character: ", font=('Helvetica', 14))
pred_label.pack()

root.mainloop()

cap.release()
cv2.destroyAllWindows()
