import cv2
import numpy as np
import mediapipe as mp

# Prepare DrawingSpec for drawing the face landmarks later.
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


def extract_landmarks_from_video(video_path, output_directory):
    cap = cv2.VideoCapture(video_path)

    counter = 0
    stage = None
    squat_start = False
    squat_count = 0
    output_file = None

    # Video writer parameters
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        try:
            while cap.isOpened():
                ret, frame = cap.read()

                if not ret:
                    break

                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = pose.process(image)

                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                try:
                    landmarks = results.pose_landmarks.landmark

                    # Get coordinates
                    hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                           landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                    ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

                    # Calculate angle
                    angle = calculate_angle(hip, knee, ankle)

                    # Squat counter logic
                    if angle > 170:
                        stage = "up"
                        if not squat_start:
                            squat_start = True
                            if output_file is not None:
                                output_file.release()
                            output_file = cv2.VideoWriter(f'{output_directory}/squat_{squat_count}.mp4', fourcc, fps,
                                                          (width, height))
                    if angle < 140 and stage == 'up':
                        stage = "down"
                        counter += 1
                        squat_start = False
                        squat_count += 1

                    # Render detections
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                              mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2,
                                                                     circle_radius=2),
                                              mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                              )

                    cv2.putText(image, 'STAGE:' + str(stage),
                                (60, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA
                                )

                    cv2.putText(image, 'SQUATS:' + str(counter),
                                (60, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA
                                )
                    if output_file is not None:
                        output_file.write(frame)

                except:
                    pass

                cv2.imshow('Squat Counter', image)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

        finally:
            if output_file is not None:
                output_file.release()
            cap.release()
            cv2.destroyAllWindows()


extract_landmarks_from_video('squat.mp4', 'output_directory')


def main():
    video_path = 'pushups-sample_user.mp4'
    output_directory = 'output'
    extract_landmarks_from_video(video_path, output_directory)


if __name__ == "__main__":
    main()
