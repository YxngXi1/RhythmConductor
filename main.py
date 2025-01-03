import cv2
import time
from modules.hand_tracker import HandTracker
import math
from playsound import playsound 
import pygame

# Constants
SCREEN_WIDTH = 1470
SCREEN_HEIGHT = 1664
BOX_LEFT = (0, 0, SCREEN_WIDTH // 3, SCREEN_HEIGHT)   # Left third of the screen
BOX_RIGHT = (SCREEN_WIDTH, 0, SCREEN_WIDTH, SCREEN_HEIGHT)  # Right third of the screen

# Specific places within the boxes
SPECIFIC_PLACES_LEFT = [(100, 100), (200, 200), (300, 300)]
SPECIFIC_PLACES_RIGHT = [(SCREEN_WIDTH - 100, 100), (SCREEN_WIDTH - 200, 200), (SCREEN_WIDTH - 300, 300)]

def is_hand_pinched(hand_landmarks):
    """Check if the hand is pinched (distance between landmarks 4 and 8 is small)."""
    thumb_tip = hand_landmarks.landmark[4]
    index_tip = hand_landmarks.landmark[8]
    distance = ((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2) ** 0.5
    return distance < 0.05  # Adjust the threshold as needed

def is_finger_near_circle(finger_x, finger_y, circle_x, circle_y, radius):
    distance = math.sqrt((finger_x - circle_x) ** 2 + (finger_y - circle_y) ** 2)
    return distance < radius

def is_line_near_circle(x1, y1, x2, y2, circle_x, circle_y, radius):
    # Calculate the distance from the line segment to the circle's center
    A = circle_x - x1
    B = circle_y - y1
    C = x2 - x1
    D = y2 - y1

    dot = A * C + B * D
    len_sq = C * C + D * D
    param = -1
    if len_sq != 0:
        param = dot / len_sq

    if param < 0:
        xx = x1
        yy = y1
    elif param > 1:
        xx = x2
        yy = y2
    else:
        xx = x1 + param * C
        yy = y1 + param * D

    dx = circle_x - xx
    dy = circle_y - yy
    distance = math.sqrt(dx * dx + dy * dy)

    return distance < radius

def draw_shrinking_circle(frame, start_time, duration=2, start_radius=200, end_radius=50):
    """Draw a shrinking circle at (100, 100) over a specified duration."""
    elapsed_time = time.time() - start_time
    if elapsed_time <= duration:
        radius = int(start_radius - (start_radius - end_radius) * (elapsed_time / duration))
        cv2.circle(frame, (250, 500), radius, (0, 0, 255), -1)  # Red color in BGR
    else:
        return False  # Indicate that the shrinking is complete
    return True

def game1(tracker):
    cap = cv2.VideoCapture(0)
    print('circle time')

    coordinates = [(1324, 151), (1651, 800),  (1324, 800)]
    sequence = [0, 2, 1]  # Updated sequence
    sequence_index = 0
    last_time = time.time()
    shrinking_start_time = None
    shrinking_radius = 100  # Initial radius of the shrinking circle
    circle_visible = True
    game_started = False  # Flag to check if the game has started
    score = 0  # Initialize score
    cycle_count = 0  # Counter for the cycles
    large_circle_start_time = None  # Start time for the large circle
    shrinking_circle_count = 0  # Counter for the shrinking circles
    rectangle_start_times = []  # Start times for the moving rectangles

    # Initialize pygame mixer
    pygame.mixer.init()

    while True:
        # OpenCV Frame
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        
        # Mediapipe Hand Detection
        results = tracker.detect_hands(frame)
        frame = tracker.draw_hands(frame, results)
        
        if not game_started:
            # Draw the initial circle on the frame
            initial_circle_coordinates = coordinates[0]
            initial_circle_radius = 50
            initial_circle_color = (0, 255, 0)  # Green color in BGR
            initial_circle_thickness = -1  # Fill the circle
            cv2.circle(frame, initial_circle_coordinates, initial_circle_radius, initial_circle_color, initial_circle_thickness)
            
            # Check for initial hit to start the game
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    for i in range(len(hand_landmarks.landmark) - 1):
                        x1 = int(hand_landmarks.landmark[i].x * frame.shape[1])
                        y1 = int(hand_landmarks.landmark[i].y * frame.shape[0])
                        x2 = int(hand_landmarks.landmark[i + 1].x * frame.shape[1])
                        y2 = int(hand_landmarks.landmark[i + 1].y * frame.shape[0])
        
                        if is_finger_near_circle(x1, y1, coordinates[0][0], coordinates[0][1], 50):
                            print("Initial hit detected, starting game")
                            pygame.mixer.music.load('audio.mp3')
                            pygame.mixer.music.play()
                            game_started = True
                            rectangle_start_times = [time.time() + 25, time.time() + 26, time.time() + 27]  # Set the start times for the rectangles
                            break
                    if game_started:
                        break
        else:
            current_time = time.time()
            if current_time - last_time >= 400/1000:
                sequence_index = (sequence_index + 1) % len(sequence)
                last_time = current_time
                shrinking_start_time = current_time  # Reset shrinking start time
                shrinking_radius = 100  # Reset shrinking radius
                circle_visible = True  # Make the circle visible again if it was destroyed
                cycle_count += 1  # Increment the cycle count

                # Check if it's time to show the large circle
                if cycle_count % 12 == 0:
                    large_circle_start_time = current_time

            current_index = sequence[sequence_index]
            x_coordinate, y_coordinate = coordinates[current_index]
            center_coordinates = (x_coordinate, y_coordinate)
            radius = 50
            color = (255, 0, 0)  # Blue color in BGR
            thickness = -1  # Fill the circle

            if circle_visible:
                # Draw the main circle on the frame
                cv2.circle(frame, center_coordinates, radius, color, thickness)

                # Calculate the shrinking circle's radius
                if shrinking_start_time is not None:
                    elapsed_time = current_time - shrinking_start_time
                    if elapsed_time <= 300/1000:
                        shrinking_radius = int(100 - (50 * (elapsed_time / (400/1000))))
                    else:
                        shrinking_radius = 50  # Ensure it doesn't shrink below the main circle's radius

                # Draw the shrinking translucent circle
                overlay = frame.copy()
                translucent_color = (255, 0, 0, 128)  # Blue color with alpha for translucency
                cv2.circle(overlay, center_coordinates, shrinking_radius, translucent_color, thickness)
                cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

            if results.multi_hand_landmarks and circle_visible:
                for hand_landmarks in results.multi_hand_landmarks:
                    for i in range(len(hand_landmarks.landmark) - 1):
                        x1 = int(hand_landmarks.landmark[i].x * frame.shape[1])
                        y1 = int(hand_landmarks.landmark[i].y * frame.shape[0])
                        x2 = int(hand_landmarks.landmark[i + 1].x * frame.shape[1])
                        y2 = int(hand_landmarks.landmark[i + 1].y * frame.shape[0])

                        if is_finger_near_circle(x1, y1, x_coordinate, y_coordinate, radius) or \
                           is_line_near_circle(x1, y1, x2, y2, x_coordinate, y_coordinate, radius):
                            print("points!")
                            circle_visible = False  # Make the circle disappear
                            score += 1  # Increment score
                            print(f"points! {score}")
                            break  # Exit the loop once a landmark is near the circle

            # Draw the large shrinking circle if it's time and the count is less than 2
            if large_circle_start_time is not None and shrinking_circle_count < 1:
                if not draw_shrinking_circle(frame, large_circle_start_time):
                    large_circle_start_time = None  # Reset the start time once shrinking is complete
                    shrinking_circle_count += 1  # Increment the shrinking circle count

            # Draw the moving rectangles if it's time
            for start_time in rectangle_start_times:
                if current_time >= start_time:
                    draw_moving_rectangle(frame, start_time)
        
        # Display the score in the top left corner
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_color = (255, 255, 255)  # White color
        font_thickness = 2
        cv2.putText(frame, f'Score: {score}', (10, 30), font, font_scale, font_color, font_thickness)

        # Show OpenCV frame
        cv2.imshow("Game Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    pygame.mixer.quit()

def main():
    # Initialize
    global cap
    cap = cv2.VideoCapture(0)

    tracker = HandTracker()
    hand_positions = {}
    previous_y = None

    while True:
        # OpenCV Frame
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        
        # Mediapipe Hand Detection
        results = tracker.detect_hands(frame)
        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                x = int(hand_landmarks.landmark[8].x * frame.shape[1])
                y = int(hand_landmarks.landmark[8].y * frame.shape[0])
                hand_positions[idx] = (x, y)
                
                # Check for sudden y-direction movement
                if previous_y is not None and abs(y - previous_y) > 200:  # Threshold for sudden change
                    print('game started')
                    cap.release()
                    game1(tracker)
                        
                previous_y = y
        
        # Draw hand landmarks
        frame = tracker.draw_hands(frame, results)
            
        
        # Show OpenCV frame
        cv2.imshow("Game Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()