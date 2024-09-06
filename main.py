import time
import argparse
import logging
import cv2
import numpy as np
import mediapipe as mp
#import blynklib
from pygame import mixer
from threading import Timer
from Utils import get_face_area
from Eye_Dector_Module import EyeDetector as EyeDet
from Pose_Estimation_Module import HeadPoseEstimator as HeadPoseEst
from Attention_Scorer_Module import AttentionScorer as AttScorer
# ids="UVu5OziedBh03M_7BCvrs__aEmXGrNCH"
#ids="qIasRJvFZK488dP_dMeRqSIVje_4v7fY"

# Initialize the mixer
mixer.init()
sound = mixer.Sound('mixkit-alert-alarm-1005 (3).wav')

# Configure logging
logging.basicConfig(filename='driver_state_detection.log',
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

camera_matrix = np.array(
    [[899.12150372, 0., 644.26261492],
     [0., 899.45280671, 372.28009436],
     [0, 0,  1]], dtype="double")

dist_coeffs = np.array(
    [[-0.03792548, 0.09233237, 0.00419088, 0.00317323, -0.15804257]], dtype="double")

def _get_landmarks(lms):
    surface = 0
    biggest_face = None
    for lms0 in lms:
        landmarks = [np.array([point.x, point.y, point.z]) \
                        for point in lms0.landmark]

        landmarks = np.array(landmarks)

        landmarks[landmarks[:, 0] < 0., 0] = 0.
        landmarks[landmarks[:, 0] > 1., 0] = 1.
        landmarks[landmarks[:, 1] < 0., 1] = 0.
        landmarks[landmarks[:, 1] > 1., 1] = 1.

        dx = landmarks[:, 0].max() - landmarks[:, 0].min()
        dy = landmarks[:, 1].max() - landmarks[:, 1].min()
        new_surface = dx * dy
        if new_surface > surface:
            biggest_face = landmarks
   
    return biggest_face

def play_alarm():
    try:
        sound.play()
    except:
        pass

def stop_alarm():
    try:
        sound.stop()
    except:
        pass

def main():

    parser = argparse.ArgumentParser(description='Driver State Detection')

    parser.add_argument('-c', '--camera', type=int,
                        default=0, metavar='', help='Camera number, default is 0 (webcam)')


    parser.add_argument('--show_fps', type=bool, default=True,
                        metavar='', help='Show the actual FPS of the capture stream, default is true')
    parser.add_argument('--show_proc_time', type=bool, default=True,
                        metavar='', help='Show the processing time for a single frame, default is true')
    parser.add_argument('--show_eye_proc', type=bool, default=False,
                        metavar='', help='Show the eyes processing, default is false')
    parser.add_argument('--show_axis', type=bool, default=True,
                        metavar='', help='Show the head pose axis, default is true')
    parser.add_argument('--verbose', type=bool, default=False,
                        metavar='', help='Prints additional info, default is false')

    parser.add_argument('--smooth_factor', type=float, default=0.5,
                        metavar='', help='Sets the smooth factor for the head pose estimation keypoint smoothing, default is 0.5')
    parser.add_argument('--ear_thresh', type=float, default=0.15,
                        metavar='', help='Sets the EAR threshold for the Attention Scorer, default is 0.15')
    parser.add_argument('--ear_time_thresh', type=float, default=2,
                        metavar='', help='Sets the EAR time (seconds) threshold for the Attention Scorer, default is 2 seconds')
    parser.add_argument('--gaze_thresh', type=float, default=0.015,
                        metavar='', help='Sets the Gaze Score threshold for the Attention Scorer, default is 0.2')
    parser.add_argument('--gaze_time_thresh', type=float, default=2, metavar='',
                        help='Sets the Gaze Score time (seconds) threshold for the Attention Scorer, default is 2. seconds')
    parser.add_argument('--pitch_thresh', type=float, default=20,
                        metavar='', help='Sets the PITCH threshold (degrees) for the Attention Scorer, default is 30 degrees')
    parser.add_argument('--yaw_thresh', type=float, default=20,
                        metavar='', help='Sets the YAW threshold (degrees) for the Attention Scorer, default is 20 degrees')
    parser.add_argument('--roll_thresh', type=float, default=20,
                        metavar='', help='Sets the ROLL threshold (degrees) for the Attention Scorer, default is 30 degrees')
    parser.add_argument('--pose_time_thresh', type=float, default=2.5,
                        metavar='', help='Sets the Pose time threshold (seconds) for the Attention Scorer, default is 2.5 seconds')

    args = parser.parse_args()

    if args.verbose:
        print(f"Arguments and Parameters used:\n{args}\n")

    if not cv2.useOptimized():
        try:
            cv2.setUseOptimized(True)  
        except:
            print(
                "OpenCV optimization could not be set to True, the script may be slower than expected")

    detector = mp.solutions.face_mesh.FaceMesh(static_image_mode=False,
                                               min_detection_confidence=0.5,
                                               min_tracking_confidence=0.5,
                                               refine_landmarks=True)

    Eye_det = EyeDet(show_processing=args.show_eye_proc)

    Head_pose = HeadPoseEst(show_axis=args.show_axis)

    t0 = time.perf_counter()
    Scorer = AttScorer(t_now=t0, ear_thresh=args.ear_thresh, gaze_time_thresh=args.gaze_time_thresh,
                       roll_thresh=args.roll_thresh, pitch_thresh=args.pitch_thresh,
                       yaw_thresh=args.yaw_thresh, ear_time_thresh=args.ear_time_thresh,
                       gaze_thresh=args.gaze_thresh, pose_time_thresh=args.pose_time_thresh,
                       verbose=args.verbose)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():  # if the camera can't be opened exit the program
        print("Cannot open camera")
        exit()

    i = 0
    time.sleep(0.01)
    alarm_timer = None
    alarm_duration = 5  # duration in seconds
    while True:  
        t_now = time.perf_counter()
        fps = i / (t_now - t0)
        if fps == 0:
            fps = 10

        ret, frame = cap.read()  # read a frame from the webcam

        if not ret:  
            print("Can't receive frame from camera/stream end")
            break

        if args.camera == 0:
            frame = cv2.flip(frame, 2)

        e1 = cv2.getTickCount()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        frame_size = frame.shape[1], frame.shape[0]

        gray = np.expand_dims(cv2.bilateralFilter(gray, 5, 10, 10), axis=2)
        gray = np.concatenate([gray, gray, gray], axis=2)

        lms = detector.process(gray).multi_face_landmarks

        if lms:  # process the frame only if at least a face is found
            landmarks = _get_landmarks(lms)

            Eye_det.show_eye_keypoints(
                color_frame=frame, landmarks=landmarks, frame_size=frame_size)

            ear = Eye_det.get_EAR(frame=gray, landmarks=landmarks)

            tired, perclos_score = Scorer.get_PERCLOS(t_now, fps, ear)

            gaze = Eye_det.get_Gaze_Score(
                frame=gray, landmarks=landmarks, frame_size=frame_size)

            frame_det, roll, pitch, yaw = Head_pose.get_pose(
                frame=frame, landmarks=landmarks, frame_size=frame_size)
           
            asleep, looking_away, distracted = Scorer.eval_scores(t_now=t_now,
                                                                  ear_score=ear,
                                                                  gaze_score=gaze,
                                                                  head_roll=roll,
                                                                  head_pitch=pitch,
                                                                  head_yaw=yaw)

            if frame_det is not None:
                frame = frame_det

            if ear is not None:
                cv2.putText(frame, "EAR:" + str(round(ear, 3)), (10, 50),
                            cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, cv2.LINE_AA)
            if perclos_score is not None:
                cv2.putText(frame, "PERCLOS:" + str(round(perclos_score, 3)), (10, 80),
                            cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, cv2.LINE_AA)
            if gaze is not None:
                cv2.putText(frame, "Gaze_Score:" + str(gaze.round(3)), (10, 110),
                            cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, cv2.LINE_AA)
           
            if roll is not None:
                cv2.putText(frame, "roll:" + str(roll.round(1)[0]), (10, 140),
                            cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, cv2.LINE_AA)
            if pitch is not None:
                cv2.putText(frame, "pitch:" + str(pitch.round(1)[0]), (10, 170),
                            cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, cv2.LINE_AA)
            if yaw is not None:
                cv2.putText(frame, "yaw:" + str(yaw.round(1)[0]), (10, 210),
                            cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, cv2.LINE_AA)

            if tired:
                print("tired")
                logging.info("Tired driver detected.")
                #blynklib.alert(ids, "alert", "Tired Driver")
                cv2.putText(frame, "TIRED!", (300, 40),
                            cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 1, cv2.LINE_AA)
                if not alarm_timer:
                    play_alarm()
                    alarm_timer = Timer(alarm_duration, stop_alarm)
                    alarm_timer.start()
            else:
                if alarm_timer and not alarm_timer.is_alive():
                    alarm_timer = None

            if asleep:
                print("asleep")
                logging.info("Driver is asleep.")
                #blynklib.alert(ids, "alert", "Driver asleep")
                cv2.putText(frame, "ASLEEP!", (300, 60),
                            cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 1, cv2.LINE_AA)
                if not alarm_timer:
                    play_alarm()
                    alarm_timer = Timer(alarm_duration, stop_alarm)
                    alarm_timer.start()
            else:
                if alarm_timer and not alarm_timer.is_alive():
                    alarm_timer = None

            if looking_away:
                print("LOOKING AWAY!")
                logging.info("Driver is looking away.")
                #blynklib.alert(ids, "alert", "Driver looking away")
                cv2.putText(frame, "LOOKING AWAY!", (300, 80),
                            cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 1, cv2.LINE_AA)
                if not alarm_timer:
                    play_alarm()
                    alarm_timer = Timer(alarm_duration, stop_alarm)
                    alarm_timer.start()
            else:
                if alarm_timer and not alarm_timer.is_alive():
                    alarm_timer = None

            if distracted:
                print("Distracted")
                logging.info("Driver is distracted.")
                #blynklib.alert(ids, "alert", "Driver distracted")
                cv2.putText(frame, "DISTRACTED!", (340, 100),
                            cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 1, cv2.LINE_AA)
                if not alarm_timer:
                    play_alarm()
                    alarm_timer = Timer(alarm_duration, stop_alarm)
                    alarm_timer.start()
            else:
                if alarm_timer and not alarm_timer.is_alive():
                    alarm_timer = None

        e2 = cv2.getTickCount()
        proc_time_frame_ms = ((e2 - e1) / cv2.getTickFrequency()) * 1000
        if args.show_fps:
            cv2.putText(frame, "FPS:" + str(round(fps)), (10, 400), cv2.FONT_HERSHEY_PLAIN, 2,
                        (255, 0, 255), 1)

        cv2.imshow("Press 'q' to terminate", frame)

        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

        i += 1

    cap.release()
    cv2.destroyAllWindows()
    logging.info("Program terminated.")

if __name__ == "__main__":
    main()