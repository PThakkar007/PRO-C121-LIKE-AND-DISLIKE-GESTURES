import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

finger_tips = [8, 12, 16, 20]
thumb_tip = 4
finger_fold_status = []

while True:
    ret, img = cap.read()
    img = cv2.flip(img, 1)
    h, w, c = img.shape
    results = hands.process(img)

    if results.multi_hand_landmarks:
        for hand_landmark in results.multi_hand_landmarks:
            lm_list = []
            for id, lm in enumerate(hand_landmark.landmark):
                lm_list.append(lm)

            # Draw circles around the fingertips
            for finger_tip in finger_tips:
                x = int(lm_list[finger_tip].x * w)
                y = int(lm_list[finger_tip].y * h)
                cv2.circle(img, (x, y), 10, (255, 0, 0), -1)

                # Check if the finger is folded or not
                if lm_list[finger_tip].x < lm_list[finger_tip - 2].x:
                    cv2.circle(img, (x, y), 10, (0, 255, 0), -1)
                    finger_fold_status.append(True)
                else:
                    finger_fold_status.append(False)

            # Check if all fingers are folded and thumb is up for "LIKE" gesture
            if all(finger_fold_status) and lm_list[thumb_tip].y < lm_list[thumb_tip - 1].y:
                print("LIKE")
                cv2.putText(img, "LIKE", (int(w/2)-50, int(h/2)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

            # Check if all fingers are folded and thumb is down for "DISLIKE" gesture
            if all(finger_fold_status) and lm_list[thumb_tip].y > lm_list[thumb_tip - 1].y:
                print("DISLIKE")
                cv2.putText(img, "DISLIKE", (int(w/2)-80, int(h/2)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)

            mp_draw.draw_landmarks(img, hand_landmark,
            mp_hands.HAND_CONNECTIONS, mp_draw.DrawingSpec((0,0,255),2,2),
            mp_draw.DrawingSpec((0,255,0),4,2))
    
    finger_fold_status.clear()
    cv2.imshow("hand tracking", img)
    cv2.waitKey(1)
