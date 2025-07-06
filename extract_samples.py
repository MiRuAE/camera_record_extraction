import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
import shutil
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import random

# 필요한 디렉토리 생성
def create_directories(base_dir):
    dirs = {
        'cluster_samples': os.path.join(base_dir, 'cluster_samples'),
        'scene_change_samples': os.path.join(base_dir, 'scene_change_samples'),
        'combined_samples': os.path.join(base_dir, 'combined_samples'),
        'thumbnails': os.path.join(base_dir, 'thumbnails')
    }

    for dir_name, dir_path in dirs.items():
        os.makedirs(dir_path, exist_ok=True)

    return dirs

# 이미지 특징 추출 함수
def extract_features(image_path, size=(64, 64)):
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None

        # 이미지 크기 조정
        resized = cv2.resize(img, size)

        # 색상 히스토그램 특징 (RGB 채널)
        hist_features = []
        for i in range(3):
            hist = cv2.calcHist([resized], [i], None, [32], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            hist_features.extend(hist)

        # 이미지 내 에지 정보 추가 (차선 감지에 중요)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_features = edges.flatten() / 255.0  # 정규화

        # 이미지 하단부 더 중요하게 고려 (차선에서 일반적으로 중요한 영역)
        bottom_half = resized[size[1]//2:, :, :]
        bottom_resized = cv2.resize(bottom_half, (32, 16))
        bottom_features = bottom_resized.flatten() / 255.0

        # 모든 특징 결합
        combined_features = np.concatenate([
            np.array(hist_features),
            edge_features * 0.5,  # 가중치 부여
            bottom_features * 2.0  # 가중치 부여
        ])

        return combined_features
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

# 클러스터링 기반 대표 이미지 추출
def extract_cluster_samples(source_dir, output_dir, num_clusters=50, sample_interval=20):
    # 파일 목록 가져오기
    all_files = sorted([f for f in os.listdir(source_dir) if f.endswith('.jpg')])

    # 너무 많은 파일이 있으므로 샘플링하여 처리
    sampled_files = all_files[::sample_interval]  # 모든 n번째 이미지만 사용
    print(f"총 {len(all_files)} 이미지 중 {len(sampled_files)} 개 샘플링됨")

    # 특징 추출
    features_list = []
    valid_files = []

    print("특징 추출 중...")
    for file in tqdm(sampled_files):
        file_path = os.path.join(source_dir, file)
        features = extract_features(file_path)
        if features is not None:
            features_list.append(features)
            valid_files.append(file)

    # 특징 정규화
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features_list)

    # K-means 클러스터링
    print(f"{num_clusters}개 클러스터로 군집화 중...")
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(scaled_features)

    # 각 클러스터의 대표 이미지 선택 (클러스터 중심에 가장 가까운 이미지)
    representative_images = []

    for i in range(num_clusters):
        cluster_indices = np.where(clusters == i)[0]
        if len(cluster_indices) > 0:
            # 클러스터 내 이미지의 특징 벡터
            cluster_features = [scaled_features[idx] for idx in cluster_indices]
            cluster_files = [valid_files[idx] for idx in cluster_indices]

            # 클러스터 중심에 가장 가까운 이미지 찾기
            centroid = kmeans.cluster_centers_[i]
            distances = cdist([centroid], cluster_features, 'euclidean')[0]
            representative_idx = np.argmin(distances)
            representative_images.append(cluster_files[representative_idx])

    # 대표 이미지 복사
    print("대표 이미지 저장 중...")
    for i, img_file in enumerate(representative_images):
        src_path = os.path.join(source_dir, img_file)
        dst_path = os.path.join(output_dir, f"cluster_{i:03d}_{img_file}")
        shutil.copy(src_path, dst_path)

    return representative_images

# 장면 변화 감지 기반 샘플 추출
def extract_scene_change_samples(source_dir, output_dir, threshold=30.0, interval=5):
    # 파일 목록 가져오기
    all_files = sorted([f for f in os.listdir(source_dir) if f.endswith('.jpg')])

    # 너무 많은 파일이 있으므로 일정 간격으로 처리
    sampled_files = all_files[::interval]
    print(f"장면 변화 감지용 {len(sampled_files)} 개 샘플링됨")

    keyframes = [sampled_files[0]]  # 첫 번째 프레임은 항상 포함
    prev_file = os.path.join(source_dir, sampled_files[0])
    prev_frame = cv2.imread(prev_file, cv2.IMREAD_COLOR)
    prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    print("장면 변화 감지 중...")
    for i in tqdm(range(1, len(sampled_files))):
        curr_file = sampled_files[i]
        curr_path = os.path.join(source_dir, curr_file)

        curr_frame = cv2.imread(curr_path, cv2.IMREAD_COLOR)
        if curr_frame is None:
            continue

        curr_frame_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        # 이전 프레임과의 차이 계산
        diff = cv2.absdiff(curr_frame_gray, prev_frame_gray)
        score = np.mean(diff)

        # MSE 계산을 추가로 수행
        h, w = prev_frame_gray.shape
        mse = np.sum((prev_frame_gray.astype("float") - curr_frame_gray.astype("float")) ** 2)
        mse /= float(h * w)

        # 차이가 임계값을 넘으면 키프레임으로 선택
        if score > threshold or mse > threshold * 5:
            keyframes.append(curr_file)
            prev_frame = curr_frame
            prev_frame_gray = curr_frame_gray

    # 대표 이미지 복사
    print("장면 변화 기반 이미지 저장 중...")
    for i, img_file in enumerate(keyframes):
        src_path = os.path.join(source_dir, img_file)
        dst_path = os.path.join(output_dir, f"scene_{i:03d}_{img_file}")
        shutil.copy(src_path, dst_path)

    return keyframes

# 섬네일 생성 함수
def create_thumbnails(source_dir, sample_files, output_dir, grid_size=(5, 10), thumb_size=(100, 100)):
    n_samples = min(grid_size[0] * grid_size[1], len(sample_files))
    if n_samples < len(sample_files):
        # 골고루 분포된 샘플 선택
        indices = np.linspace(0, len(sample_files) - 1, n_samples, dtype=int)
        selected_samples = [sample_files[i] for i in indices]
    else:
        selected_samples = sample_files

    # 섬네일 이미지 생성
    thumbnail_grid = Image.new('RGB',
                             (grid_size[1] * thumb_size[0],
                              grid_size[0] * thumb_size[1]))

    for i, img_file in enumerate(selected_samples):
        if i >= grid_size[0] * grid_size[1]:
            break

        src_path = os.path.join(source_dir, img_file)
        try:
            img = Image.open(src_path)
            img = img.resize(thumb_size)

            row = i // grid_size[1]
            col = i % grid_size[1]
            thumbnail_grid.paste(img, (col * thumb_size[0], row * thumb_size[1]))
        except Exception as e:
            print(f"Error creating thumbnail for {src_path}: {e}")

    # 섬네일 저장
    grid_path = os.path.join(output_dir, "thumbnail_grid.jpg")
    thumbnail_grid.save(grid_path, quality=95)
    print(f"섬네일 그리드가 {grid_path}에 저장되었습니다.")

    return grid_path

# 두 결과 병합 및 중복 제거
def combine_samples(source_dir, cluster_samples, scene_samples, output_dir):
    # 두 샘플링 방법의 결과 합치기
    all_samples = list(set(cluster_samples) | set(scene_samples))
    print(f"클러스터링: {len(cluster_samples)}, 장면변화: {len(scene_samples)}, 중복제거 후: {len(all_samples)}")

    # 결과 복사
    for i, img_file in enumerate(all_samples):
        src_path = os.path.join(source_dir, img_file)
        dst_path = os.path.join(output_dir, f"sample_{i:04d}_{img_file}")
        shutil.copy(src_path, dst_path)

    return all_samples

# 메인 함수
def main():
    # 경로 설정
    # source_dir = "/Users/jsyun/Programming/Embedded_TARS_AI/lane_recorded"
    # output_base_dir = "/Users/jsyun/Programming/Embedded_TARS_AI/lane_samples"

    source_dir = "./img_recorded"
    output_base_dir = "./img_samples"

    # 디렉토리 생성
    dirs = create_directories(output_base_dir)

    # 클러스터링 기반 샘플 추출
    cluster_samples = extract_cluster_samples(
        source_dir,
        dirs['cluster_samples'],
        num_clusters=50,  # 추출할 대표 이미지 수
        sample_interval=20  # 처리 속도를 위한 샘플링 간격
    )

    # 장면 변화 감지 기반 샘플 추출
    scene_samples = extract_scene_change_samples(
        source_dir,
        dirs['scene_change_samples'],
        threshold=40.0,  # 장면 변화 감지 임계값 (조정 필요)
        interval=10  # 처리 속도를 위한 샘플링 간격
    )

    # 두 결과 병합
    combined_samples = combine_samples(
        source_dir,
        cluster_samples,
        scene_samples,
        dirs['combined_samples']
    )

    # 섬네일 생성
    create_thumbnails(
        source_dir,
        combined_samples,
        dirs['thumbnails']
    )

    print(f"\n작업 완료! 총 {len(combined_samples)} 개의 대표 이미지가 선택되었습니다.")
    print(f"결과물은 {output_base_dir} 디렉토리에 저장되었습니다.")

if __name__ == "__main__":
    main()
