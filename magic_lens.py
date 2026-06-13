import cv2
import mediapipe as mp
import numpy as np
import math

class HandTracker:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils

    def process_frame(self, frame):
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)
        h, w, c = frame.shape

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

            if len(results.multi_hand_landmarks) == 2:
                
                sorted_hands = sorted(results.multi_hand_landmarks, key=lambda hand: hand.landmark[0].x)
                left_hand = sorted_hands[0]
                right_hand = sorted_hands[1]

                def get_pt(hand, landmark_idx):
                    lm = hand.landmark[landmark_idx]
                    return [int(lm.x * w), int(lm.y * h)]

                l_thumb, l_index, l_middle = get_pt(left_hand, 4), get_pt(left_hand, 8), get_pt(left_hand, 12)
                l_ring, l_pinky = get_pt(left_hand, 16), get_pt(left_hand, 20)
                
                r_thumb, r_index, r_middle = get_pt(right_hand, 4), get_pt(right_hand, 8), get_pt(right_hand, 12)
                r_ring, r_pinky = get_pt(right_hand, 16), get_pt(right_hand, 20)

                for pt in [l_thumb, r_thumb]: cv2.circle(frame, pt, 10, (0, 0, 255), -1)     
                for pt in [l_index, r_index]: cv2.circle(frame, pt, 10, (0, 255, 0), -1)     
                for pt in [l_middle, r_middle]: cv2.circle(frame, pt, 10, (255, 0, 0), -1)   
                for pt in [l_ring, r_ring]: cv2.circle(frame, pt, 10, (255, 255, 0), -1)     
                for pt in [l_pinky, r_pinky]: cv2.circle(frame, pt, 10, (0, 255, 255), -1)   

                def get_dist(pt1, pt2):
                    return math.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1])

                dist_thumbs = get_dist(l_thumb, r_thumb)
                dist_indexes = get_dist(l_index, r_index)
                dist_middles = get_dist(l_middle, r_middle)
                dist_rings = get_dist(l_ring, r_ring)
                dist_pinkies = get_dist(l_pinky, r_pinky)

                cv2.putText(frame, f"Thumb Dist: {int(dist_thumbs)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"Index Dist: {int(dist_indexes)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"Middle Dist: {int(dist_middles)}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, "Pinky/Ring: Gesture Mode", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                TOUCH_THRESHOLD = 450  
                polygon_points = []
                active_filter = 0

                # --- HELPER FUNCTIONS FOR GESTURE DETECTION ---
                def is_up(hand, tip_idx, base_idx):
                    return hand.landmark[tip_idx].y < hand.landmark[base_idx].y

                def is_down(hand, tip_idx, base_idx):
                    return hand.landmark[tip_idx].y > hand.landmark[base_idx].y

                # Check if BOTH hands have Pinky UP and Ring DOWN
                filter_4_gesture = (
                    is_up(left_hand, 20, 17) and is_down(left_hand, 16, 14) and
                    is_up(right_hand, 20, 17) and is_down(right_hand, 16, 14)
                )

                # ---  LOGIC ---
                combinations = {
                    1: {"name": "LENS: GRAYSCALE", "avg_dist": (dist_thumbs + dist_indexes) / 2, "pts": [l_thumb, r_thumb, r_index, l_index], "color": (255,255,255)},
                    2: {"name": "LENS: CYBER", "avg_dist": (dist_indexes + dist_middles) / 2, "pts": [l_index, r_index, r_middle, l_middle], "color": (0,255,0)},
                    3: {"name": "LENS: BONE-FROST", "avg_dist": (dist_middles + dist_rings) / 2, "pts": [l_middle, r_middle, r_ring, l_ring], "color": (255,200,150)},
                    4: {"name": "LENS: INVERTED", "avg_dist": 9999, "pts": [l_ring, r_ring, r_pinky, l_pinky], "color": (0,0,255)}
                }

                valid_combos = {k: v for k, v in combinations.items() if k != 4 and v["avg_dist"] < TOUCH_THRESHOLD}

                if filter_4_gesture:
                    valid_combos[4] = combinations[4]
                    valid_combos[4]["avg_dist"] = -1  # Negative guarantees it wins cuz it's a special gesture

                if valid_combos:
                    best_combo_id = min(valid_combos, key=lambda k: valid_combos[k]["avg_dist"])
                    active_filter = best_combo_id
                    polygon_points = valid_combos[best_combo_id]["pts"]
                    
                    text_color = valid_combos[best_combo_id]["color"]
                    filter_name = valid_combos[best_combo_id]["name"]
                    cv2.putText(frame, filter_name, (w - 350, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)


                # --- AR MASKING ---
                if polygon_points:
                    hull = cv2.convexHull(np.array(polygon_points))
                    mask = np.zeros_like(frame)
                    cv2.fillPoly(mask, [hull], (255, 255, 255))
                    filtered_frame = frame.copy()
                    
                    if active_filter == 1:
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        high_contrast = cv2.convertScaleAbs(gray, alpha=1.8, beta=20)
                        filtered_frame = cv2.cvtColor(high_contrast, cv2.COLOR_GRAY2BGR)
                        
                    elif active_filter == 2:
                        filtered_frame[:, :, 0] = 0 
                        filtered_frame[:, :, 2] = 0 
                        filtered_frame[:, :, 1] = cv2.convertScaleAbs(frame[:, :, 1], alpha=2.2, beta=30)
                        
                    elif active_filter == 3:
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        high_contrast = cv2.convertScaleAbs(gray, alpha=1.5, beta=10)
                        colored_bone = cv2.applyColorMap(high_contrast, cv2.COLORMAP_BONE)
                        hsv = cv2.cvtColor(colored_bone, cv2.COLOR_BGR2HSV)
                        hsv[:, :, 1] = cv2.convertScaleAbs(hsv[:, :, 1], alpha=1.5, beta=0)
                        filtered_frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                        
                    elif active_filter == 4:
                        inv = cv2.bitwise_not(frame)
                        hsv = cv2.cvtColor(inv, cv2.COLOR_BGR2HSV)
                        hsv[:, :, 1] = cv2.convertScaleAbs(hsv[:, :, 1], alpha=1.8, beta=0)
                        hsv[:, :, 2] = cv2.convertScaleAbs(hsv[:, :, 2], alpha=1.3, beta=20)
                        filtered_frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

                    frame = np.where(mask == 255, filtered_frame, frame)
                    border_color = combinations[active_filter]["color"]
                    cv2.polylines(frame, [hull], isClosed=True, color=border_color, thickness=4)

        return frame, results
def main():
    cap = cv2.VideoCapture(0) # Keep your specific camera index (0, 1, 2, etc.) here
    tracker = HandTracker()

    print("Pipeline active. Press 'q' to quit.")

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        processed_frame, results = tracker.process_frame(frame)
        cv2.imshow("Magic Lens Prototype", processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()