import cv2


# def show_camera_feed():
#     # Open the default camera (usually camera index 0)
#     cap = cv2.VideoCapture(1)

#     while True:
#         # Read a frame from the camera
#         ret, frame = cap.read()

#         # Convert to the YCrCb color space
#         ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)

#         # Threshold off the brightness
#         # Extract the Y channel (luminance)
#         Y_channel = ycrcb[:, :, 0]

#         # Apply thresholding on the Y channel
#         _, thresholded = cv2.threshold(Y_channel, 220, 255, cv2.THRESH_BINARY)

#         # Display the results
#         cv2.imshow("Y Channel", Y_channel)
#         cv2.imshow("Thresholded", thresholded)

#         # Display the frame
#         cv2.imshow("Camera Feed", frame)

#         # Break the loop if 'q' key is pressed
#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break

#     # Release the camera and close the window
#     cap.release()
#     cv2.destroyAllWindows()


def show_camera_feed():
    cap = cv2.VideoCapture(1)

    while True:
        ret, frame = cap.read()

        # Threshold from grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Threshold
        _, thresholded = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

        thresholded = cv2.erode(thresholded, None, iterations=1)

        # Display the results

        cv2.imshow("Thresholded", thresholded)

        cv2.imshow("Camera Feed", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


# # Call the function to start the live camera feed
show_camera_feed()
