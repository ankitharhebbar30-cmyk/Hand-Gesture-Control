import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import mediapipe as mp
import pyautogui
import threading
import time
import math
from pathlib import Path


# ============================================================
# SETTINGS
# ============================================================

GESTURE_CONFIRM_FRAMES = 5
PINCH_RATIO = 0.45
CLICK_COOLDOWN = 0.8
SMOOTHING = 7


# ============================================================
# GLOBAL VARIABLES
# ============================================================

camera_running = False
camera_thread = None

current_gesture = "NONE"
current_action = "None"

last_click_time = 0


# ============================================================
# CUSTOM GESTURE MAPPINGS
# ============================================================

gesture_actions = {

    "THUMBS UP": "Volume Up",
    "THUMBS DOWN": "Volume Down",
    "OPEN PALM": "Play / Pause",
    "PEACE": "Next Track",
    "FIST": "Previous Track",
    "INDEX": "Mouse Movement",
    "LEFT CLICK": "Left Click",
    "RIGHT CLICK": "Right Click"

}


# ============================================================
# AVAILABLE ACTIONS
# ============================================================

available_actions = [

    "None",
    "Volume Up",
    "Volume Down",
    "Mute",
    "Play / Pause",
    "Next Track",
    "Previous Track",
    "Left Click",
    "Right Click",
    "Mouse Movement"

]


# ============================================================
# SCREEN
# ============================================================

screen_width, screen_height = pyautogui.size()


# ============================================================
# FINGER DETECTION
# ============================================================

def get_finger_states(hand):

    fingers = []

    # Thumb
    if hand[4].x > hand[3].x:
        fingers.append(1)
    else:
        fingers.append(0)

    # Index
    if hand[8].y < hand[6].y:
        fingers.append(1)
    else:
        fingers.append(0)

    # Middle
    if hand[12].y < hand[10].y:
        fingers.append(1)
    else:
        fingers.append(0)

    # Ring
    if hand[16].y < hand[14].y:
        fingers.append(1)
    else:
        fingers.append(0)

    # Pinky
    if hand[20].y < hand[18].y:
        fingers.append(1)
    else:
        fingers.append(0)

    return fingers


# ============================================================
# BASIC GESTURE RECOGNITION
# ============================================================

def recognize_gesture(fingers):

    if fingers == [1, 1, 1, 1, 1]:
        return "OPEN PALM"

    elif fingers == [0, 0, 0, 0, 0]:
        return "FIST"

    elif fingers == [0, 1, 0, 0, 0]:
        return "INDEX"

    elif fingers == [0, 1, 1, 0, 0]:
        return "PEACE"

    elif fingers == [1, 0, 0, 0, 0]:
        return "THUMB"

    return "UNKNOWN"


# ============================================================
# THUMB DIRECTION
# ============================================================

def get_thumb_direction(hand):

    if hand[4].y < hand[2].y - 0.08:
        return "THUMBS UP"

    elif hand[4].y > hand[2].y + 0.08:
        return "THUMBS DOWN"

    return "THUMB"


# ============================================================
# DISTANCE
# ============================================================

def distance(p1, p2):

    return math.sqrt(

        (p1.x - p2.x) ** 2
        +
        (p1.y - p2.y) ** 2

    )


# ============================================================
# PINCH DETECTION
# ============================================================

def is_pinching(hand):

    pinch_distance = distance(
        hand[4],
        hand[8]
    )

    hand_size = distance(
        hand[0],
        hand[9]
    )

    if hand_size == 0:
        return False

    ratio = pinch_distance / hand_size

    return ratio < PINCH_RATIO


# ============================================================
# PERFORM SELECTED ACTION
# ============================================================

def perform_action(action):

    global current_action

    if action == "None":

        current_action = "None"


    elif action == "Volume Up":

        pyautogui.press("volumeup")

        current_action = "Volume Up"


    elif action == "Volume Down":

        pyautogui.press("volumedown")

        current_action = "Volume Down"


    elif action == "Mute":

        pyautogui.press("volumemute")

        current_action = "Mute"


    elif action == "Play / Pause":

        pyautogui.press("playpause")

        current_action = "Play / Pause"


    elif action == "Next Track":

        pyautogui.press("nexttrack")

        current_action = "Next Track"


    elif action == "Previous Track":

        pyautogui.press("prevtrack")

        current_action = "Previous Track"


    elif action == "Left Click":

        pyautogui.click()

        current_action = "Left Click"


    elif action == "Right Click":

        pyautogui.rightClick()

        current_action = "Right Click"


    elif action == "Mouse Movement":

        current_action = "Mouse Movement"


# ============================================================
# HAND CONNECTIONS
# ============================================================

connections = [

    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),

    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),

    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),

    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),

    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),

    (0, 17)

]


# ============================================================
# CAMERA ENGINE
# ============================================================

def camera_loop():

    global camera_running
    global current_gesture
    global current_action
    global last_click_time

    model_path = Path(
        "models/hand_landmarker.task"
    )

    BaseOptions = mp.tasks.BaseOptions

    HandLandmarker = (
        mp.tasks.vision.HandLandmarker
    )

    HandLandmarkerOptions = (
        mp.tasks.vision.HandLandmarkerOptions
    )

    RunningMode = (
        mp.tasks.vision.RunningMode
    )

    options = HandLandmarkerOptions(

        base_options=BaseOptions(
            model_asset_path=str(model_path)
        ),

        running_mode=RunningMode.VIDEO,

        num_hands=1
    )


    cap = cv2.VideoCapture(0)

    timestamp = 0

    candidate = "UNKNOWN"

    candidate_count = 0

    stable = "UNKNOWN"

    previous_x = screen_width // 2
    previous_y = screen_height // 2


    with HandLandmarker.create_from_options(
        options
    ) as landmarker:

        while camera_running:

            success, frame = cap.read()

            if not success:
                break


            # Mirror camera

            frame = cv2.flip(
                frame,
                1
            )


            # Convert to RGB

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )


            # MediaPipe image

            image = mp.Image(

                image_format=
                mp.ImageFormat.SRGB,

                data=rgb

            )


            # Detect

            result = landmarker.detect_for_video(

                image,

                timestamp

            )

            timestamp += 1


            # ==================================================
            # HAND DETECTED
            # ==================================================

            if result.hand_landmarks:

                hand = result.hand_landmarks[0]


                fingers = get_finger_states(
                    hand
                )


                gesture = recognize_gesture(
                    fingers
                )


                # Thumb direction

                if gesture == "THUMB":

                    gesture = get_thumb_direction(
                        hand
                    )


                # ==================================================
                # PINCH
                # ==================================================

                pinch = is_pinching(hand)


                if (

                    pinch
                    and fingers == [0, 1, 0, 0, 0]

                ):

                    gesture = "LEFT CLICK"


                elif (

                    pinch
                    and fingers == [0, 1, 1, 0, 0]

                ):

                    gesture = "RIGHT CLICK"


                # ==================================================
                # STABILIZATION
                # ==================================================

                if gesture == candidate:

                    candidate_count += 1

                else:

                    candidate = gesture

                    candidate_count = 1


                # ==================================================
                # CONFIRM GESTURE
                # ==================================================

                if candidate_count >= GESTURE_CONFIRM_FRAMES:

                    if stable != candidate:

                        stable = candidate

                        current_gesture = stable


                        # Get selected action

                        action = gesture_actions.get(
                            stable,
                            "None"
                        )


                        # ==================================================
                        # CLICK PROTECTION
                        # ==================================================

                        if action in [
                            "Left Click",
                            "Right Click"
                        ]:

                            now = time.time()


                            if (
                                now - last_click_time
                                > CLICK_COOLDOWN
                            ):

                                perform_action(
                                    action
                                )

                                last_click_time = now


                        elif action != "Mouse Movement":

                            perform_action(
                                action
                            )


                # ==================================================
                # MOUSE MOVEMENT
                # ==================================================

                selected_action = gesture_actions.get(
                    stable,
                    "None"
                )


                if selected_action == "Mouse Movement":

                    target_x = int(

                        hand[8].x
                        * screen_width

                    )


                    target_y = int(

                        hand[8].y
                        * screen_height

                    )


                    mouse_x = (

                        previous_x

                        +

                        (
                            target_x
                            -
                            previous_x
                        )
                        / SMOOTHING

                    )


                    mouse_y = (

                        previous_y

                        +

                        (
                            target_y
                            -
                            previous_y
                        )
                        / SMOOTHING

                    )


                    pyautogui.moveTo(

                        int(mouse_x),

                        int(mouse_y)

                    )


                    previous_x = mouse_x
                    previous_y = mouse_y


                # ==================================================
                # DRAW HAND
                # ==================================================

                height, width, _ = (
                    frame.shape
                )

                points = []


                for landmark in hand:

                    x = int(
                        landmark.x * width
                    )

                    y = int(
                        landmark.y * height
                    )


                    points.append(
                        (x, y)
                    )


                    cv2.circle(

                        frame,

                        (x, y),

                        5,

                        (0, 255, 0),

                        -1

                    )


                # Connections

                for start, end in connections:

                    cv2.line(

                        frame,

                        points[start],

                        points[end],

                        (255, 0, 0),

                        2

                    )


                # ==================================================
                # CAMERA TEXT
                # ==================================================

                selected_action = gesture_actions.get(

                    stable,

                    "None"

                )


                cv2.putText(

                    frame,

                    f"Gesture: {stable}",

                    (20, 45),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.8,

                    (0, 255, 0),

                    2

                )


                cv2.putText(

                    frame,

                    f"Action: {selected_action}",

                    (20, 80),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.7,

                    (0, 255, 255),

                    2

                )


            else:

                current_gesture = "NONE"

                stable = "UNKNOWN"


            # ==================================================
            # SHOW CAMERA
            # ==================================================

            cv2.imshow(

                "Hand Gesture Control",

                frame

            )


            # Quit with Q

            if cv2.waitKey(1) & 0xFF == ord("q"):

                camera_running = False

                break


    cap.release()

    cv2.destroyAllWindows()

    current_gesture = "NONE"

    current_action = "None"


# ============================================================
# START CAMERA
# ============================================================

def start_camera():

    global camera_running
    global camera_thread

    if camera_running:
        return


    camera_running = True


    status_label.config(
        text="● CAMERA ON"
    )


    camera_thread = threading.Thread(

        target=camera_loop,

        daemon=True

    )


    camera_thread.start()


# ============================================================
# STOP CAMERA
# ============================================================

def stop_camera():

    global camera_running

    camera_running = False


    status_label.config(
        text="● CAMERA OFF"
    )


# ============================================================
# SAVE MAPPINGS
# ============================================================

def save_mappings():

    for gesture, combo in mapping_widgets.items():

        gesture_actions[gesture] = (
            combo.get()
        )


    messagebox.showinfo(

        "Saved",

        "Gesture mappings saved!"

    )


# ============================================================
# RESET MAPPINGS
# ============================================================

def reset_mappings():

    defaults = {

        "THUMBS UP": "Volume Up",
        "THUMBS DOWN": "Volume Down",
        "OPEN PALM": "Play / Pause",
        "PEACE": "Next Track",
        "FIST": "Previous Track",
        "INDEX": "Mouse Movement",
        "LEFT CLICK": "Left Click",
        "RIGHT CLICK": "Right Click"

    }


    gesture_actions.update(
        defaults
    )


    for gesture, combo in mapping_widgets.items():

        combo.set(
            defaults[gesture]
        )


# ============================================================
# UPDATE GUI
# ============================================================

def update_gui():

    gesture_label.config(
        text=current_gesture
    )


    action_label.config(
        text=current_action
    )


    if camera_running:

        status_label.config(
            text="● CAMERA ON"
        )

    else:

        status_label.config(
            text="● CAMERA OFF"
        )


    root.after(
        100,
        update_gui
    )


# ============================================================
# GUI
# ============================================================

root = tk.Tk()

root.title(
    "Hand Gesture Control"
)

root.geometry(
    "900x760"
)

root.resizable(
    False,
    False
)


# ============================================================
# TITLE
# ============================================================

title = tk.Label(

    root,

    text="HAND GESTURE CONTROL",

    font=("Arial", 24, "bold")

)

title.pack(
    pady=15
)


subtitle = tk.Label(

    root,

    text="AI-powered computer control using hand gestures",

    font=("Arial", 11)

)

subtitle.pack()


# ============================================================
# STATUS
# ============================================================

status_label = tk.Label(

    root,

    text="● CAMERA OFF",

    font=("Arial", 15, "bold")

)

status_label.pack(
    pady=15
)


# ============================================================
# CURRENT GESTURE
# ============================================================

gesture_label = tk.Label(

    root,

    text="NONE",

    font=("Arial", 25, "bold")

)

gesture_label.pack()


action_label = tk.Label(

    root,

    text="None",

    font=("Arial", 13)

)

action_label.pack(
    pady=3
)


# ============================================================
# MAPPING TITLE
# ============================================================

mapping_title = tk.Label(

    root,

    text="CUSTOM GESTURE MAPPING",

    font=("Arial", 14, "bold")

)

mapping_title.pack(
    pady=(15, 8)
)


# ============================================================
# MAPPING FRAME
# ============================================================

mapping_frame = tk.Frame(
    root
)

mapping_frame.pack()


mapping_widgets = {}


gestures = [

    "THUMBS UP",
    "THUMBS DOWN",
    "OPEN PALM",
    "PEACE",
    "FIST",
    "INDEX",
    "LEFT CLICK",
    "RIGHT CLICK"

]


for row, gesture in enumerate(gestures):

    gesture_text = tk.Label(

        mapping_frame,

        text=gesture,

        width=18,

        anchor="w",

        font=("Arial", 10)

    )

    gesture_text.grid(

        row=row,

        column=0,

        padx=10,

        pady=4

    )


    combo = ttk.Combobox(

        mapping_frame,

        values=available_actions,

        state="readonly",

        width=25

    )


    combo.set(
        gesture_actions[gesture]
    )


    combo.grid(

        row=row,

        column=1,

        padx=10,

        pady=4

    )


    mapping_widgets[gesture] = combo


# ============================================================
# SAVE / RESET
# ============================================================

mapping_button_frame = tk.Frame(
    root
)

mapping_button_frame.pack(
    pady=12
)


save_button = tk.Button(

    mapping_button_frame,

    text="SAVE MAPPINGS",

    font=("Arial", 10, "bold"),

    width=18,

    command=save_mappings

)

save_button.grid(

    row=0,

    column=0,

    padx=8

)


reset_button = tk.Button(

    mapping_button_frame,

    text="RESET DEFAULTS",

    font=("Arial", 10, "bold"),

    width=18,

    command=reset_mappings

)

reset_button.grid(

    row=0,

    column=1,

    padx=8

)


# ============================================================
# CAMERA BUTTONS
# ============================================================

button_frame = tk.Frame(
    root
)

button_frame.pack(
    pady=10
)


start_button = tk.Button(

    button_frame,

    text="▶ START CAMERA",

    font=("Arial", 11, "bold"),

    width=18,

    command=start_camera

)

start_button.grid(

    row=0,

    column=0,

    padx=10

)


stop_button = tk.Button(

    button_frame,

    text="■ STOP CAMERA",

    font=("Arial", 11, "bold"),

    width=18,

    command=stop_camera

)

stop_button.grid(

    row=0,

    column=1,

    padx=10

)


# ============================================================
# FOOTER
# ============================================================

footer = tk.Label(

    root,

    text="Hand Gesture Control • AIML Project • V4",

    font=("Arial", 9)

)

footer.pack(

    side="bottom",

    pady=8

)


# ============================================================
# START GUI UPDATE
# ============================================================

update_gui()

root.mainloop()