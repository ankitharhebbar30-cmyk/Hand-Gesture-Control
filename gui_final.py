import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import mediapipe as mp
import pyautogui
import threading
import time
import math
from pathlib import Path
from datetime import datetime


# ============================================================
# SETTINGS
# ============================================================

GESTURE_CONFIRM_FRAMES = 5

PINCH_START_RATIO = 0.42
PINCH_RELEASE_RATIO = 0.60

CLICK_COOLDOWN = 0.8
SCREENSHOT_COOLDOWN = 2.0

SMOOTHING = 3


# ============================================================
# GLOBALS
# ============================================================

camera_running = False
camera_thread = None

current_gesture = "NONE"
current_action = "None"

last_click_time = 0
last_screenshot_time = 0

left_pinch_active = False
right_pinch_active = False

screen_width, screen_height = pyautogui.size()


# ============================================================
# SCREENSHOT FOLDER
# ============================================================

SCREENSHOT_FOLDER = Path("screenshots")
SCREENSHOT_FOLDER.mkdir(exist_ok=True)


# ============================================================
# GESTURE ACTIONS
# ============================================================

gesture_actions = {
    "THUMBS UP": "Volume Up",
    "THUMBS DOWN": "Volume Down",
    "OPEN PALM": "Play / Pause",
    "PEACE": "Next Track",
    "FIST": "Previous Track",
    "INDEX": "Mouse Movement",
    "TWO OPEN PALMS": "Screenshot"
}


available_actions = [
    "None",
    "Volume Up",
    "Volume Down",
    "Mute",
    "Play / Pause",
    "Next Track",
    "Previous Track",
    "Mouse Movement",
    "Screenshot"
]


# ============================================================
# FINGER STATES
# ============================================================

def get_finger_states(hand):

    fingers = []

    # Thumb
    if hand[4].x > hand[3].x:
        fingers.append(1)
    else:
        fingers.append(0)

    # Index
    fingers.append(
        1 if hand[8].y < hand[6].y else 0
    )

    # Middle
    fingers.append(
        1 if hand[12].y < hand[10].y else 0
    )

    # Ring
    fingers.append(
        1 if hand[16].y < hand[14].y else 0
    )

    # Pinky
    fingers.append(
        1 if hand[20].y < hand[18].y else 0
    )

    return fingers


# ============================================================
# GESTURE RECOGNITION
# ============================================================

def recognize_gesture(fingers):

    if fingers == [1, 1, 1, 1, 1]:
        return "OPEN PALM"

    if fingers == [0, 0, 0, 0, 0]:
        return "FIST"

    if fingers == [0, 1, 0, 0, 0]:
        return "INDEX"

    if fingers == [0, 1, 1, 0, 0]:
        return "PEACE"

    if fingers == [1, 0, 0, 0, 0]:
        return "THUMB"

    return "UNKNOWN"


# ============================================================
# THUMB DIRECTION
# ============================================================

def get_thumb_direction(hand):

    if hand[4].y < hand[2].y - 0.06:
        return "THUMBS UP"

    if hand[4].y > hand[2].y + 0.06:
        return "THUMBS DOWN"

    return "THUMB"


# ============================================================
# DISTANCE
# ============================================================

def distance(p1, p2):

    return math.sqrt(
        (p1.x - p2.x) ** 2 +
        (p1.y - p2.y) ** 2
    )


# ============================================================
# PINCH RATIO
# ============================================================

def pinch_ratio(hand, finger_index):

    finger_distance = distance(
        hand[4],
        hand[finger_index]
    )

    hand_size = distance(
        hand[0],
        hand[9]
    )

    if hand_size == 0:
        return 999

    return finger_distance / hand_size


# ============================================================
# TAKE SCREENSHOT
# ============================================================

def take_screenshot():

    timestamp = datetime.now().strftime(
        "%Y-%m-%d_%H-%M-%S"
    )

    filename = (
        SCREENSHOT_FOLDER /
        f"screenshot_{timestamp}.png"
    )

    pyautogui.screenshot(
        str(filename)
    )

    print(
        f"Screenshot saved: {filename}"
    )


# ============================================================
# PERFORM ACTION
# ============================================================

def perform_action(action):

    global current_action
    global last_click_time

    now = time.time()

    if now - last_click_time < CLICK_COOLDOWN:
        return

    if action == "Volume Up":

        pyautogui.press("volumeup")

    elif action == "Volume Down":

        pyautogui.press("volumedown")

    elif action == "Mute":

        pyautogui.press("volumemute")

    elif action == "Play / Pause":

        pyautogui.press("playpause")

    elif action == "Next Track":

        pyautogui.press("nexttrack")

    elif action == "Previous Track":

        pyautogui.press("prevtrack")

    elif action == "Screenshot":

        take_screenshot()

    current_action = action

    last_click_time = now


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
# CAMERA LOOP
# ============================================================

def camera_loop():

    global camera_running
    global current_gesture
    global current_action

    global last_click_time
    global last_screenshot_time

    global left_pinch_active
    global right_pinch_active

    cap = None

    try:

        model_path = Path(
            "models/hand_landmarker.task"
        )

        if not model_path.exists():

            print(
                "ERROR: models/hand_landmarker.task "
                "not found."
            )

            camera_running = False
            return


        options = mp.tasks.vision.HandLandmarkerOptions(

            base_options=mp.tasks.BaseOptions(
                model_asset_path=str(model_path)
            ),

            running_mode=(
                mp.tasks.vision.RunningMode.VIDEO
            ),

            num_hands=2

        )


        cap = cv2.VideoCapture(0)

        if not cap.isOpened():

            print(
                "ERROR: Camera could not open."
            )

            camera_running = False
            return


        timestamp = 0

        candidate = "UNKNOWN"
        candidate_count = 0
        stable = "UNKNOWN"

        previous_x = screen_width // 2
        previous_y = screen_height // 2


        with mp.tasks.vision.HandLandmarker.create_from_options(
            options
        ) as landmarker:


            while camera_running:

                success, frame = cap.read()

                if not success:

                    print(
                        "Camera frame error."
                    )

                    break


                frame = cv2.flip(
                    frame,
                    1
                )


                rgb = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )


                image = mp.Image(

                    image_format=(
                        mp.ImageFormat.SRGB
                    ),

                    data=rgb

                )


                result = landmarker.detect_for_video(
                    image,
                    timestamp
                )

                timestamp += 1


                # ====================================================
                # HANDS DETECTED
                # ====================================================

                if result.hand_landmarks:

                    hands = result.hand_landmarks


                    # =================================================
                    # TWO HANDS
                    # =================================================

                    if len(hands) >= 2:

                        hand1 = hands[0]
                        hand2 = hands[1]

                        fingers1 = get_finger_states(
                            hand1
                        )

                        fingers2 = get_finger_states(
                            hand2
                        )


                        open1 = sum(fingers1) >= 4
                        open2 = sum(fingers2) >= 4


                        # TWO OPEN PALMS
                        if open1 and open2:

                            current_gesture = (
                                "TWO OPEN PALMS"
                            )

                            current_action = (
                                "Screenshot"
                            )


                            now = time.time()


                            if (
                                now -
                                last_screenshot_time
                                >= SCREENSHOT_COOLDOWN
                            ):

                                take_screenshot()

                                last_screenshot_time = now


                        # TWO FISTS
                        elif (
                            sum(fingers1) == 0
                            and
                            sum(fingers2) == 0
                        ):

                            current_gesture = (
                                "TWO FISTS"
                            )

                            current_action = (
                                "No Action"
                            )


                        else:

                            current_gesture = (
                                "TWO HANDS"
                            )

                            current_action = (
                                "No Action"
                            )


                    # =================================================
                    # ONE HAND
                    # =================================================

                    else:

                        hand = hands[0]

                        fingers = get_finger_states(
                            hand
                        )


                        gesture = recognize_gesture(
                            fingers
                        )


                        # THUMB
                        if gesture == "THUMB":

                            gesture = get_thumb_direction(
                                hand
                            )


                        # =================================================
                        # PROTECTED GESTURES
                        # =================================================

                        is_fist = (
                            fingers ==
                            [0, 0, 0, 0, 0]
                        )


                        is_thumb = (
                            gesture in [
                                "THUMBS UP",
                                "THUMBS DOWN"
                            ]
                        )


                        is_open_palm = (
                            fingers ==
                            [1, 1, 1, 1, 1]
                        )


                        # =================================================
                        # PINCH
                        # =================================================

                        if (
                            not is_fist
                            and
                            not is_thumb
                            and
                            not is_open_palm
                        ):

                            index_ratio = pinch_ratio(
                                hand,
                                8
                            )

                            middle_ratio = pinch_ratio(
                                hand,
                                12
                            )


                            # LEFT CLICK
                            if (
                                fingers ==
                                [1, 1, 0, 0, 0]
                                and
                                index_ratio <
                                PINCH_START_RATIO
                            ):

                                if not left_pinch_active:

                                    pyautogui.click()

                                    current_gesture = (
                                        "LEFT CLICK"
                                    )

                                    current_action = (
                                        "Left Click"
                                    )

                                    left_pinch_active = True


                            elif (
                                index_ratio >
                                PINCH_RELEASE_RATIO
                            ):

                                left_pinch_active = False


                            # RIGHT CLICK
                            if (
                                fingers ==
                                [1, 0, 1, 0, 0]
                                and
                                middle_ratio <
                                PINCH_START_RATIO
                            ):

                                if not right_pinch_active:

                                    pyautogui.rightClick()

                                    current_gesture = (
                                        "RIGHT CLICK"
                                    )

                                    current_action = (
                                        "Right Click"
                                    )

                                    right_pinch_active = True


                            elif (
                                middle_ratio >
                                PINCH_RELEASE_RATIO
                            ):

                                right_pinch_active = False


                        else:

                            left_pinch_active = False
                            right_pinch_active = False


                        # =================================================
                        # NORMAL GESTURES
                        # =================================================

                        if gesture != "UNKNOWN":

                            if gesture == candidate:

                                candidate_count += 1

                            else:

                                candidate = gesture
                                candidate_count = 1


                            if (
                                candidate_count >=
                                GESTURE_CONFIRM_FRAMES
                            ):

                                stable = candidate

                                current_gesture = stable

                                action = (
                                    gesture_actions.get(
                                        stable,
                                        "None"
                                    )
                                )


                                if (
                                    stable != "FIST"
                                    and
                                    stable not in [
                                        "THUMBS UP",
                                        "THUMBS DOWN"
                                    ]
                                ):

                                    if action != "None":

                                        perform_action(
                                            action
                                        )


                        # FORCE FIST
                        if is_fist:

                            stable = "FIST"

                            current_gesture = "FIST"

                            current_action = (
                                gesture_actions[
                                    "FIST"
                                ]
                            )


                        # FORCE THUMBS
                        elif is_thumb:

                            stable = gesture

                            current_gesture = gesture

                            current_action = (
                                gesture_actions.get(
                                    gesture,
                                    "None"
                                )
                            )


                        # =================================================
                        # MOUSE MOVEMENT
                        # =================================================

                        if stable == "INDEX":

                            target_x = int(
                                hand[8].x *
                                screen_width
                            )

                            target_y = int(
                                hand[8].y *
                                screen_height
                            )


                            mouse_x = (
                                previous_x +
                                (
                                    target_x -
                                    previous_x
                                ) / SMOOTHING
                            )


                            mouse_y = (
                                previous_y +
                                (
                                    target_y -
                                    previous_y
                                ) / SMOOTHING
                            )


                            pyautogui.moveTo(
                                int(mouse_x),
                                int(mouse_y)
                            )


                            previous_x = mouse_x
                            previous_y = mouse_y


                    # =================================================
                    # DRAW LANDMARKS
                    # =================================================

                    height, width, _ = frame.shape


                    for detected_hand in hands:

                        points = []


                        for landmark in detected_hand:

                            x = int(
                                landmark.x *
                                width
                            )

                            y = int(
                                landmark.y *
                                height
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


                        for start, end in connections:

                            cv2.line(
                                frame,
                                points[start],
                                points[end],
                                (255, 0, 0),
                                2
                            )


                    # =================================================
                    # TEXT
                    # =================================================

                    cv2.putText(
                        frame,
                        f"Gesture: {current_gesture}",
                        (20, 45),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2
                    )


                    cv2.putText(
                        frame,
                        f"Action: {current_action}",
                        (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2
                    )


                else:

                    current_gesture = "NONE"
                    current_action = "None"

                    candidate = "UNKNOWN"
                    candidate_count = 0
                    stable = "UNKNOWN"

                    left_pinch_active = False
                    right_pinch_active = False


                cv2.imshow(
                    "Hand Gesture Control",
                    frame
                )


                if (
                    cv2.waitKey(1) & 0xFF
                    == ord("q")
                ):

                    camera_running = False
                    break


    except Exception as e:

        print()
        print("================================")
        print("CAMERA ERROR")
        print("================================")
        print(e)
        print("================================")
        print()

        camera_running = False


    finally:

        if cap is not None:
            cap.release()

        cv2.destroyAllWindows()

        camera_running = False


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
        daemon=False
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

        gesture_actions[
            gesture
        ] = combo.get()


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
        "TWO OPEN PALMS": "Screenshot"

    }


    gesture_actions.update(
        defaults
    )


    for gesture, combo in mapping_widgets.items():

        combo.set(
            defaults[gesture]
        )


# ============================================================
# GUI UPDATE
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
    "Hand Gesture Control - FINAL"
)

root.geometry(
    "900x800"
)

root.resizable(
    False,
    False
)


title = tk.Label(
    root,
    text="HAND GESTURE CONTROL",
    font=("Arial", 25, "bold")
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


status_label = tk.Label(
    root,
    text="● CAMERA OFF",
    font=("Arial", 15, "bold")
)

status_label.pack(
    pady=15
)


gesture_title = tk.Label(
    root,
    text="CURRENT GESTURE",
    font=("Arial", 12, "bold")
)

gesture_title.pack()


gesture_label = tk.Label(
    root,
    text="NONE",
    font=("Arial", 25, "bold")
)

gesture_label.pack(
    pady=5
)


action_label = tk.Label(
    root,
    text="None",
    font=("Arial", 13)
)

action_label.pack()


mapping_title = tk.Label(
    root,
    text="GESTURE CONTROLS",
    font=("Arial", 14, "bold")
)

mapping_title.pack(
    pady=(15, 8)
)


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
    "TWO OPEN PALMS"

]


for row, gesture in enumerate(gestures):

    label = tk.Label(
        mapping_frame,
        text=gesture,
        width=20,
        anchor="w",
        font=("Arial", 10)
    )

    label.grid(
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


    mapping_widgets[
        gesture
    ] = combo


# ============================================================
# BUTTONS
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
# HELP
# ============================================================

help_text = tk.Label(
    root,
    text=(
        "✋ Open Palm = Play/Pause    "
        "✌ Peace = Next Track    "
        "👍 Thumbs Up = Volume Up\n"
        "👎 Thumbs Down = Volume Down    "
        "✊ Fist = Previous Track    "
        "☝ Index = Mouse\n"
        "🤏 Thumb + Index = Left Click    "
        "🤏 Thumb + Middle = Right Click\n"
        "👐 Two Open Palms = Screenshot"
    ),
    font=("Arial", 9),
    justify="center"
)

help_text.pack(
    pady=8
)


footer = tk.Label(
    root,
    text="Hand Gesture Control • AIML Project • FINAL",
    font=("Arial", 9)
)

footer.pack(
    side="bottom",
    pady=8
)


# ============================================================
# SAFE CLOSE
# ============================================================

def close_program():

    global camera_running

    camera_running = False


    if camera_thread is not None:

        camera_thread.join(
            timeout=3
        )


    root.destroy()


root.protocol(
    "WM_DELETE_WINDOW",
    close_program
)


# ============================================================
# RUN
# ============================================================

update_gui()

root.mainloop()