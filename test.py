import cv2

video_path = "outputs/output_20241206093216.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()


# Process each frame
while True:
    ret, frame = cap.read()
    if not ret:  # Break if no frames left
        continue

    # # Apply a blur effect
    # blurred_frame = cv2.GaussianBlur(frame, (15, 15), 0)  # Adjust kernel size as needed

    # Display the blurred frame
    cv2.imshow("Blurred Video", frame)

    # Save the frame as an image
    cv2.imwrite("test_frames/frame.jpg", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
