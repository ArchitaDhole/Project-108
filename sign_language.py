import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

thumb_tip = 4

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

            thumb_y = int(lm_list[thumb_tip].y * h)
            thumb_x = int(lm_list[thumb_tip].x * w)

            cv2.circle(img, (thumb_x, thumb_y), 15, (255, 0, 0), cv2.FILLED)

            if lm_list[thumb_tip].y < lm_list[thumb_tip-1].y < lm_list[thumb_tip-2].y:
                print("LIKE")
                cv2.putText(img, "LIKE", (20, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

            if lm_list[thumb_tip].y > lm_list[thumb_tip-1].y > lm_list[thumb_tip-2].y:
                print("DISLIKE")
                cv2.putText(img, "DISLIKE", (20, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            mp_draw.draw_landmarks(img, hand_landmark,
                                   mp_hands.HAND_CONNECTIONS, mp_draw.DrawingSpec(
                                       (0, 0, 255), 2, 2),
                                   mp_draw.DrawingSpec((0, 255, 0), 4, 2))

    cv2.imshow("hand tracking", img)
    key = cv2.waitKey(1)
    if key == 32:
        break
