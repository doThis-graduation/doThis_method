import cv2
import numpy as np
import mediapipe as mp

# Prepare DrawingSpec for drawing the face landmarks later.
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

class Exercise:
    def __init__(self, video_path, output_directory):
        self.video_path = video_path
        self.output_directory = output_directory
        self.cap = cv2.VideoCapture(video_path)

        # Video writer parameters
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    def calculate_angle(self, a, b, c):
        a = np.array(a)  # First
        b = np.array(b)  # Mid
        c = np.array(c)  # End

        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)

        if angle > 180.0:
            angle = 360 - angle

        return angle

    def extract_landmarks_from_video(self):
        raise NotImplementedError("Please Implement this method")

class Squat(Exercise):
    def extract_landmarks_from_video(self):
        counter = 0
        stage = None
        squat_start = False
        squat_count = 0
        output_file = None

        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            try:
                while self.cap.isOpened():
                    ret, frame = self.cap.read()

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
                        angle = self.calculate_angle(hip, knee, ankle)

                        # Squat counter logic
                        if angle > 170:
                            stage = "up"
                            if not squat_start:
                                squat_start = True
                                if output_file is not None:
                                    output_file.release()
                                output_file = cv2.VideoWriter(f'{self.output_directory}/squat_{squat_count}.mp4', self.fourcc, self.fps,
                                                              (self.width, self.height))
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
                self.cap.release()
                cv2.destroyAllWindows()

class Pushup(Exercise):
    def extract_landmarks_from_video(self):
        counter = 0
        stage = None
        pushup_start = False
        pushup_count = 0
        output_file = None

        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            try:
                while self.cap.isOpened():
                    ret, frame = self.cap.read()

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
                        shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                               landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                        elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                                landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                        wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                                 landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                        # Calculate angle
                        angle = self.calculate_angle(shoulder, elbow, wrist)

                        # Pushup counter logic
                        if angle > 170:
                            stage = "up"
                            if not pushup_start:
                                pushup_start = True
                                if output_file is not None:
                                    output_file.release()
                                output_file = cv2.VideoWriter(f'{self.output_directory}/pushup_{pushup_count}.mp4', self.fourcc, self.fps,
                                                              (self.width, self.height))
                        if angle < 80 and stage == 'up':
                            stage = "down"
                            counter += 1
                            pushup_start = False
                            pushup_count += 1

                        # Render detections
                        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2,
                                                                         circle_radius=2),
                                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                                  )

                        cv2.putText(image, 'STAGE:' + str(stage),
                                    (60, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA
                                    )

                        cv2.putText(image, 'PUSHUPS:' + str(counter),
                                    (60, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA
                                    )
                        if output_file is not None:
                            output_file.write(frame)

                    except:
                        pass

                    cv2.imshow('Pushup Counter', image)

                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break

            finally:
                if output_file is not None:
                    output_file.release()
                self.cap.release()
                cv2.destroyAllWindows()

class Pullup(Exercise):
    def extract_landmarks_from_video(self):
        counter = 0
        stage = None
        pullup_start = False
        pullup_count = 0
        output_file = None

        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            try:
                while self.cap.isOpened():
                    ret, frame = self.cap.read()

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
                        shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                               landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                        elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                                landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                        wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                                 landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                        # Calculate angle
                        angle = self.calculate_angle(shoulder, elbow, wrist)

                        # Pullup counter logic
                        if angle > 160:
                            stage = "down"
                            if not pullup_start:
                                pullup_start = True
                                if output_file is not None:
                                    output_file.release()
                                output_file = cv2.VideoWriter(f'{self.output_directory}/pullup_{pullup_count}.mp4', self.fourcc, self.fps,
                                                              (self.width, self.height))
                        if angle < 80 and stage == 'down':
                            stage = "up"
                            counter += 1
                            pullup_start = False
                            pullup_count += 1

                        # Render detections
                        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2,
                                                                         circle_radius=2),
                                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                                  )

                        cv2.putText(image, 'STAGE:' + str(stage),
                                    (60, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA
                                    )

                        cv2.putText(image, 'PULLUPS:' + str(counter),
                                    (60, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA
                                    )
                        if output_file is not None:
                            output_file.write(frame)

                    except:
                        pass

                    cv2.imshow('Pullup Counter', image)

                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break

            finally:
                if output_file is not None:
                    output_file.release()
                self.cap.release()
                cv2.destroyAllWindows()

def main():
    video_path = 'pushup.mp4'
    output_directory = 'output'
    pushup = Pushup(video_path, output_directory)
    pushup.extract_landmarks_from_video()


if __name__ == "__main__":
    main()
