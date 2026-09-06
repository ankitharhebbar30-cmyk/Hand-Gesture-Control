import cv2
import mediapipe as mp
import pyautogui
import time
import math
from pathlib import Path


# ==========================================
# SCREEN
# ==========================================

screen_width, screen_height = pyautogui.size()


# ==========================================
# SETTINGS
# ==========================================

GESTURE_CONFIRM_FRAMES = 5

PINCH_RATIO = 0.45

CLICK_COOLDOWN = 0.8

SMOOTHING = 7


# ==========================================
# FINGER DETECTION
# ==========================================

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


# ==========================================
# GESTURE RECOGNITION
# ==========================================

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

    else:
        return "UNKNOWN"


# ==========================================
# THUMB DIRECTION
# ==========================================

def get_thumb_direction(hand):

    thumb_tip = hand[4]

    thumb_base = hand[2]

    if thumb_tip.y < thumb_base.y - 0.08:

        return "THUMBS UP"

    elif thumb_tip.y > thumb_base.y + 0.08:

        return "THUMBS DOWN"

    return "THUMB"


# ==========================================
# DISTANCE BETWEEN LANDMARKS
# ==========================================

def get_distance(point1, point2):

    return math.sqrt(

        (point1.x - point2.x) ** 2

        +

        (point1.y - point2.y) ** 2

    )


# ==========================================
# IMPROVED PINCH DETECTION
# ==========================================

def is_pinching(hand):

    thumb_tip = hand[4]

    index_tip = hand[8]

    wrist = hand[0]

    middle_base = hand[9]


    # Distance between thumb and index

    pinch_distance = get_distance(

        thumb_tip,

        index_tip

    )


    # Size of the hand

    hand_size = get_distance(

        wrist,

        middle_base

    )


    # Avoid division by zero

    if hand_size == 0:

        return False


    # Normalize pinch distance

    ratio = pinch_distance / hand_size


    print(
        f"Pinch ratio: {ratio:.2f}"
    )


    return ratio < PINCH_RATIO


# ==========================================
# COMPUTER ACTIONS
# ==========================================

def perform_action(gesture):

    if gesture == "THUMBS UP":

        print("VOLUME UP")

        pyautogui.press("volumeup")

        return "Volume Up"


    elif gesture == "THUMBS DOWN":

        print("VOLUME DOWN")

        pyautogui.press("volumedown")

        return "Volume Down"


    elif gesture == "OPEN PALM":

        print("PLAY / PAUSE")

        pyautogui.press("playpause")

        return "Play / Pause"


    elif gesture == "PEACE":

        print("NEXT TRACK")

        pyautogui.press("nexttrack")

        return "Next Track"


    elif gesture == "FIST":

        print("PREVIOUS TRACK")

        pyautogui.press("prevtrack")

        return "Previous Track"


    return ""


# ==========================================
# MEDIAPIPE
# ==========================================

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


VisionRunningMode = (

    mp.tasks.vision.RunningMode

)


options = HandLandmarkerOptions(

    base_options=BaseOptions(

        model_asset_path=str(model_path)

    ),

    running_mode=VisionRunningMode.VIDEO,

    num_hands=1

)


# ==========================================
# CAMERA
# ==========================================

cap = cv2.VideoCapture(0)

timestamp = 0


# ==========================================
# GESTURE STABILIZATION
# ==========================================

candidate_gesture = "UNKNOWN"

candidate_count = 0

stable_gesture = "UNKNOWN"


# ==========================================
# ACTION DISPLAY
# ==========================================

current_action = "None"


# ==========================================
# CLICK CONTROL
# ==========================================

last_click_time = 0


# ==========================================
# MOUSE
# ==========================================

previous_mouse_x = screen_width // 2

previous_mouse_y = screen_height // 2


# ==========================================
# HAND CONNECTIONS
# ==========================================

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


# ==========================================
# MAIN PROGRAM
# ==========================================

with HandLandmarker.create_from_options(

    options

) as landmarker:

    while True:

        success, frame = cap.read()


        if not success:

            print(
                "Could not access camera"
            )

            break


        # Mirror

        frame = cv2.flip(
            frame,
            1
        )


        # RGB

        rgb_frame = cv2.cvtColor(

            frame,

            cv2.COLOR_BGR2RGB

        )


        # MediaPipe image

        mp_image = mp.Image(

            image_format=
            mp.ImageFormat.SRGB,

            data=rgb_frame

        )


        # Detect

        result = landmarker.detect_for_video(

            mp_image,

            timestamp

        )


        timestamp += 1


        # ==================================
        # HAND DETECTED
        # ==================================

        if result.hand_landmarks:

            hand = result.hand_landmarks[0]


            # Finger states

            fingers = get_finger_states(
                hand
            )


            # Raw gesture

            raw_gesture = recognize_gesture(
                fingers
            )


            # Thumb direction

            if raw_gesture == "THUMB":

                raw_gesture = get_thumb_direction(
                    hand
                )


            # ==================================
            # PINCH
            # ==================================

            pinch = is_pinching(hand)


            # ==================================
            # SPECIAL GESTURES
            # ==================================

            special_gesture = None


            # Right click

            if (

                pinch

                and fingers
                == [0, 1, 1, 0, 0]

            ):

                special_gesture = (
                    "RIGHT CLICK"
                )


            # Left click

            elif (

                pinch

                and fingers
                == [0, 1, 0, 0, 0]

            ):

                special_gesture = (
                    "LEFT CLICK"
                )


            # ==================================
            # GESTURE TO STABILIZE
            # ==================================

            if special_gesture:

                gesture_for_stability = (
                    special_gesture
                )

            else:

                gesture_for_stability = (
                    raw_gesture
                )


            # ==================================
            # STABILIZATION
            # ==================================

            if (

                gesture_for_stability

                == candidate_gesture

            ):

                candidate_count += 1

            else:

                candidate_gesture = (
                    gesture_for_stability
                )

                candidate_count = 1


            # ==================================
            # CONFIRM
            # ==================================

            if (

                candidate_count

                >= GESTURE_CONFIRM_FRAMES

            ):

                if (

                    stable_gesture

                    != candidate_gesture

                ):

                    stable_gesture = (
                        candidate_gesture
                    )


                    # ==================================
                    # NORMAL ACTIONS
                    # ==================================

                    if stable_gesture not in [

                        "INDEX",

                        "LEFT CLICK",

                        "RIGHT CLICK"

                    ]:

                        current_action = (
                            perform_action(
                                stable_gesture
                            )
                        )


                    # ==================================
                    # LEFT CLICK
                    # ==================================

                    elif (

                        stable_gesture
                        == "LEFT CLICK"

                    ):

                        now = time.time()


                        if (

                            now
                            - last_click_time

                            > CLICK_COOLDOWN

                        ):

                            print(
                                "LEFT CLICK"
                            )

                            pyautogui.click()


                            current_action = (
                                "Left Click"
                            )


                            last_click_time = now


                    # ==================================
                    # RIGHT CLICK
                    # ==================================

                    elif (

                        stable_gesture
                        == "RIGHT CLICK"

                    ):

                        now = time.time()


                        if (

                            now
                            - last_click_time

                            > CLICK_COOLDOWN

                        ):

                            print(
                                "RIGHT CLICK"
                            )

                            pyautogui.rightClick()


                            current_action = (
                                "Right Click"
                            )


                            last_click_time = now


            # ==================================
            # MOUSE MOVEMENT
            # ==================================

            if stable_gesture == "INDEX":

                index_x = hand[8].x

                index_y = hand[8].y


                target_x = int(

                    index_x
                    * screen_width

                )


                target_y = int(

                    index_y
                    * screen_height

                )


                current_mouse_x = (

                    previous_mouse_x

                    +

                    (

                        target_x

                        -

                        previous_mouse_x

                    )

                    / SMOOTHING

                )


                current_mouse_y = (

                    previous_mouse_y

                    +

                    (

                        target_y

                        -

                        previous_mouse_y

                    )

                    / SMOOTHING

                )


                pyautogui.moveTo(

                    int(current_mouse_x),

                    int(current_mouse_y)

                )


                previous_mouse_x = (
                    current_mouse_x
                )

                previous_mouse_y = (
                    current_mouse_y
                )


            # ==================================
            # DRAW LANDMARKS
            # ==================================

            height, width, _ = (
                frame.shape
            )

            points = []


            for landmark in hand:

                x = int(

                    landmark.x
                    * width

                )

                y = int(

                    landmark.y
                    * height

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


            # ==================================
            # CONNECTIONS
            # ==================================

            for start, end in connections:

                cv2.line(

                    frame,

                    points[start],

                    points[end],

                    (255, 0, 0),

                    2

                )


            # ==================================
            # DISPLAY
            # ==================================

            cv2.putText(

                frame,

                f"Gesture: {stable_gesture}",

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


        # ==================================
        # NO HAND
        # ==================================

        else:

            candidate_gesture = "UNKNOWN"

            candidate_count = 0

            stable_gesture = "UNKNOWN"

            current_action = "None"


        # ==================================
        # CAMERA
        # ==================================

        cv2.imshow(

            "Hand Gesture Control",

            frame

        )


        # ==================================
        # QUIT
        # ==================================

        if (

            cv2.waitKey(1) & 0xFF

            == ord("q")

        ):

            break


# ==========================================
# CLEANUP
# ==========================================

cap.release()

cv2.destroyAllWindows()