import cv2

def main():
    # Open a connection to the webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video capture.")
        return

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break

        # Define the circle parameters
        center_coordinates = (frame.shape[1] // 2, frame.shape[0] // 2)
        radius = 50
        color = (255, 0, 0)  # Blue color in BGR
        thickness = 2  # Thickness of 2 px

        # Draw the circle on the frame
        cv2.circle(frame, center_coordinates, radius, color, thickness)

        # Display the resulting frame
        cv2.imshow('Video Capture with Circle', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()