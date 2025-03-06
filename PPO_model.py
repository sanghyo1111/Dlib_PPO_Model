import cv2
import dlib
import json
import os
import numpy as np
import random
import gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

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

class FaceNoiseEnv(gym.Env):
    def __init__(self, video_path, landmarks_json):
        super(FaceNoiseEnv, self).__init__()
        
        self.cap = cv2.VideoCapture(video_path)
        with open(landmarks_json, "r") as f:
            self.landmarks_data = json.load(f)
        
        self.current_frame = 0
        self.num_landmarks = 68
        
        # Action space: (패치 위치 index, 크기, 강도)
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32)
        
        # Observation space: 얼굴 랜드마크 좌표
        self.observation_space = gym.spaces.Box(
            low=0, high=640, shape=(self.num_landmarks * 2,), dtype=np.float32
        )
    
    def step(self, action):
        if self.current_frame >= len(self.landmarks_data):
            done = True
            return self.state, 0, done, {}
        
        # 랜드마크 데이터 로드
        original_landmarks = np.array(self.landmarks_data[self.current_frame]["landmarks"]).reshape(-1, 2)
        
        # 패치 적용 (특정 랜드마크 위치에 적용)
        landmark_idx = int(action[0] * (self.num_landmarks - 1))  # 랜드마크 인덱스 선택
        patch_x, patch_y = original_landmarks[landmark_idx]  # 선택된 랜드마크 좌표
        patch_radius = int(action[1] * 20) + 5  # 최소 반경 5 픽셀 이상
        noise_intensity = int(action[2] * 50) + 10  # 최소 노이즈 강도 10 이상
        
        # 현재 프레임 로드
        ret, frame = self.cap.read()
        if not ret:
            done = True
            return self.state, 0, done, {}
        
        # 원형 패치 적용
        frame = apply_circular_noise_patch(frame, [(patch_x, patch_y)], patch_radius, noise_intensity)
        
        # dlib으로 변경된 랜드마크 추출
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        modified_landmarks = original_landmarks.copy()
        
        for face in faces:
            landmarks = predictor(gray, face)
            modified_landmarks = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)])
            break
        
        # 보상 계산 (좌표 변화량)
        reward = np.sum(np.linalg.norm(original_landmarks - modified_landmarks, axis=1))
        
        print(f"Frame: {self.current_frame}, Landmark: {landmark_idx}, Patch Pos: ({patch_x}, {patch_y}), Radius: {patch_radius}, Intensity: {noise_intensity}, Reward: {reward}")
        
        self.current_frame += 1
        done = self.current_frame >= len(self.landmarks_data)
        self.state = modified_landmarks.flatten()
        
        return self.state, reward, done, {}
    
    def reset(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.current_frame = 0
        self.state = np.array(self.landmarks_data[self.current_frame]["landmarks"]).flatten()
        return self.state
    
    def render(self, mode="human"):
        pass


# 환경 생성
env = DummyVecEnv([lambda: FaceNoiseEnv("input_video.mp4", "output/landmarks.json")])

# PPO 모델 학습
model = PPO("MlpPolicy", env, verbose=1)
print("Starting PPO training...")
model.learn(total_timesteps=10000)
print("PPO training complete.")

# 학습된 모델 저장
model.save("ppo_face_noise")
print("Model saved as ppo_face_noise.")

# def test_model_on_image(image_path, model_path):
#     model = PPO.load(model_path)
#     print("Model loaded successfully.")
    
#     image = cv2.imread(image_path)
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     faces = detector(gray)
    
#     for face in faces:
#         landmarks = predictor(gray, face)
#         original_landmarks = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)])
        
#         # 모델 예측 (노이즈 패치 적용 위치, 크기, 강도 결정)
#         action, _ = model.predict(original_landmarks.flatten(), deterministic=True)
#         landmark_idx = int(action[0] * 67)
#         patch_x, patch_y = original_landmarks[landmark_idx]
#         patch_radius = int(action[1] * 20) + 5
#         noise_intensity = int(action[2] * 50) + 10
        
#         # 노이즈 패치 적용
#         noisy_image = apply_circular_noise_patch(image, [(patch_x, patch_y)], patch_radius, noise_intensity)
        
#         print(f"Testing on Image - Landmark: {landmark_idx}, Patch Pos: ({patch_x}, {patch_y}), Radius: {patch_radius}, Intensity: {noise_intensity}")
        
#         # 결과 출력
#         cv2.imshow("Original Image", image)
#         cv2.imshow("Noisy Image", noisy_image)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()
#         break
