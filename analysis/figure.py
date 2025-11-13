import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import itertools

# --- 1. 정사면체 꼭짓점 좌표 설정 ---
r = 1.0 # 구의 반지름

vertices_aligned = r * np.array([
    [0, 0, 1], 
    [2*np.sqrt(2)/3, 0, -1/3], 
    [-np.sqrt(2)/3, np.sqrt(6)/3, -1/3], 
    [-np.sqrt(2)/3, -np.sqrt(6)/3, -1/3]  
])

# --- 2. Matplotlib을 이용한 3D 시각화 ---

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 1. 구(Hypersphere) 그리기 (투명하게)
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x = r * np.outer(np.cos(u), np.sin(v))
y = r * np.outer(np.sin(u), np.sin(v))
z = r * np.outer(np.ones(np.size(u)), np.cos(v))

ax.plot_surface(x, y, z, color='lightblue', alpha=0.1, linewidth=0, antialiased=False)

# 2. 정사면체의 모서리 (점선)
edges = list(itertools.combinations(range(4), 2))

for i, j in edges:
    v_i = vertices_aligned[i]
    v_j = vertices_aligned[j]
    ax.plot([v_i[0], v_j[0]], [v_i[1], v_j[1]], [v_i[2], v_j[2]], 
            color='darkblue', linestyle='--', linewidth=1.5)

# 3. 중심에서 각 꼭짓점까지 화살표 및 텍스트 주석
center = np.array([0, 0, 0])
arrow_ratio = 0.2

for i in range(4):
    v_i = vertices_aligned[i]
    
    ax.quiver(center[0], center[1], center[2], 
              v_i[0], v_i[1], v_i[2], 
              length=np.linalg.norm(v_i), 
              normalize=False, 
              color='black', arrow_length_ratio=arrow_ratio, linewidth=1.5)

    offset = 0.25 
    direction_vector = v_i / np.linalg.norm(v_i) 
    text_pos = v_i + direction_vector * offset 
    
    ax.text(text_pos[0], text_pos[1], text_pos[2], 
            f'Class {i+1} target center vector', 
            color='white', fontsize=9, ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.2", fc="darkslateblue", ec="midnightblue", lw=0.5, alpha=0.8))


# 4. 원점 표시 및 레이블 추가
ax.scatter(0, 0, 0, c='black', marker='o', s=30) 
o_offset = 0.1
ax.text(center[0] + o_offset, center[1], center[2] + o_offset,
        'O', color='black', fontsize=12, ha='left', va='bottom')


# --- 5. Class 1과 Class 2 사이 각도 표시 (추가된 부분) ---
v1 = vertices_aligned[0] # Class 1 벡터
v2 = vertices_aligned[1] # Class 2 벡터

# 두 벡터의 코사인 각도 계산
dot_product = np.dot(v1, v2)
norm_v1 = np.linalg.norm(v1)
norm_v2 = np.linalg.norm(v2)
cos_theta = dot_product / (norm_v1 * norm_v2)
angle_rad = np.arccos(cos_theta) # 라디안 각도

# 각도를 나타낼 호의 반지름 (원점에서 너무 멀지 않게)
arc_r = 0.4 

# 두 벡터가 만드는 평면의 법선 벡터 (두 벡터의 외적)
normal_vec = np.cross(v1, v2)
normal_vec = normal_vec / np.linalg.norm(normal_vec) # 단위 법선 벡터

# 회전 행렬을 사용하여 v1을 v2 방향으로 회전시키며 호를 그립니다.
# rotation_axis = normal_vec, rotation_angle = angle_rad
# 로드리게스의 회전 공식 (Rodrigues' rotation formula)을 직접 구현하거나,
# quaternions 같은 라이브러리를 사용해야 하지만, 간단하게는 중간 벡터를 생성합니다.

# 두 벡터 사이의 보간된 벡터들을 생성하여 호를 그림
num_points = 50
arc_points = []
for alpha in np.linspace(0, angle_rad, num_points):
    # v1을 normal_vec 주위로 alpha만큼 회전
    # rotation matrix based on axis-angle representation
    K = np.array([
        [0, -normal_vec[2], normal_vec[1]],
        [normal_vec[2], 0, -normal_vec[0]],
        [-normal_vec[1], normal_vec[0], 0]
    ])
    R = np.identity(3) + np.sin(alpha) * K + (1 - np.cos(alpha)) * np.dot(K, K)
    
    rotated_v = np.dot(R, v1)
    arc_point = arc_r * (rotated_v / np.linalg.norm(rotated_v))
    arc_points.append(arc_point)

arc_points = np.array(arc_points)

ax.plot(arc_points[:, 0], arc_points[:, 1], arc_points[:, 2], 
        color='purple', linestyle='-', linewidth=2)

# 각도 텍스트 (theta) 위치
# v1과 v2 사이의 중간 각도 방향으로 배치
mid_angle = angle_rad / 2
K_mid = np.array([
    [0, -normal_vec[2], normal_vec[1]],
    [normal_vec[2], 0, -normal_vec[0]],
    [-normal_vec[1], normal_vec[0], 0]
])
R_mid = np.identity(3) + np.sin(mid_angle) * K_mid + (1 - np.cos(mid_angle)) * np.dot(K_mid, K_mid)
mid_vec = np.dot(R_mid, v1)
mid_vec = mid_vec / np.linalg.norm(mid_vec)

theta_text_pos = mid_vec * (arc_r + 0.1) # 호의 바깥쪽에 텍스트 배치

ax.text(theta_text_pos[0], theta_text_pos[1], theta_text_pos[2], 
        r'$\theta$', # LaTeX 형식으로 세타 표시
        color='purple', fontsize=14, ha='center', va='center',
        bbox=dict(boxstyle="round,pad=0.2", fc="lavender", ec="purple", lw=0.5, alpha=0.7))


# 6. 플롯 설정 (축, 제목 제거)
ax.set_box_aspect([1, 1, 1]) 
ax.set_xlim([-r * 1.2, r * 1.2])
ax.set_ylim([-r * 1.2, r * 1.2])
ax.set_zlim([-r * 1.2, r * 1.2])

ax.set_axis_off() 

# 7. 시점(Viewpoint) 조정
ax.view_init(elev=30, azim=45) 

plt.show()