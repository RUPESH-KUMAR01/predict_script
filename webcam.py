import cv2

from predict import predict_webcam


def webcam():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run YOLOv8 inference on the frame
            results = predict_webcam(frame)
            # Display the annotated frame
            cv2.imshow("YOLOv8 Inference", results)

            # Break the loop if 'q' is pressed or window is closed
            if cv2.waitKey(1) & 0xFF == ord("q") or cv2.getWindowProperty('YOLOv8 Inference', cv2.WND_PROP_VISIBLE) < 1:
                break
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

webcam()