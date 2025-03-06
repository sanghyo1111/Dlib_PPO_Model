import cv2
import dlib
import json
import os
import numpy as np
import random

# dlib 얼굴 검출기 및 랜드마크 예측기 로드
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def apply_circular_noise_patch(image, landmarks, patch_radius=10, noise_intensity=30):
    noisy_image = image.copy()
    for (x, y) in landmarks:
        mask = np.zeros_like(image, dtype=np.uint8)
        cv2.circle(mask, (x, y), patch_radius, (255, 255, 255), -1)
        noise = np.random.normal(0, noise_intensity, image.shape).astype(np.uint8)
        noisy_image = cv2.bitwise_and(noisy_image, 255 - mask) + cv2.bitwise_and(noise, mask)
    return noisy_image

def extract_landmarks_from_video(video_path, output_dir, apply_noise=False):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    landmarks_data = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        
        for face in faces:
            landmarks = predictor(gray, face)
            points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)]
            landmarks_data.append({"frame": frame_count, "landmarks": points})
            
            if apply_noise:
                frame = apply_circular_noise_patch(frame, points)
        
        frame_count += 1
    
    cap.release()
    
    # 결과 저장
    output_path = os.path.join(output_dir, "landmarks.json")
    with open(output_path, "w") as f:
        json.dump(landmarks_data, f, indent=4)
    
    print(f"Landmarks saved to {output_path}")

# 실행 예시
extract_landmarks_from_video("WIN_20250306_01_31_54_Pro.mp4", "output", apply_noise=True)
