import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)

# == 토글: 포인트, 센터 표시 On/Off ==
show_points = True   # 데이터포인트 표시
show_centers = True  # 각 클래스 센터 표시

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

## ==== Case 1: dominant class1, 각 클래스 밀집 ====
class1_mean = 0
class1_std = np.deg2rad(35)
n1 = 120
angles1 = np.random.normal(class1_mean, class1_std, n1)
angles1 = np.clip(angles1, -2*np.pi/3, 2*np.pi/3)

class2_mean = 5*np.pi/6
class2_std = np.deg2rad(12)
n2 = 40
angles2 = np.random.normal(class2_mean, class2_std, n2)
angles2 = np.clip(angles2, 2*np.pi/3, np.pi)

class3_mean = -5*np.pi/6
class3_std = np.deg2rad(12)
n3 = 40
angles3 = np.random.normal(class3_mean, class3_std, n3)
angles3 = np.clip(angles3, -np.pi, -2*np.pi/3)

x1, y1 = np.cos(angles1), np.sin(angles1)
x2, y2 = np.cos(angles2), np.sin(angles2)
x3, y3 = np.cos(angles3), np.sin(angles3)

# 센터 좌표 (case1)
centers1 = [
    (np.cos(class1_mean), np.sin(class1_mean)),
    (np.cos(class2_mean), np.sin(class2_mean)),
    (np.cos(class3_mean), np.sin(class3_mean))
]

ax1.plot(np.cos(np.linspace(0, 2 * np.pi, 400)), np.sin(np.linspace(0, 2 * np.pi, 400)), 'k--', linewidth=1)
if show_points:
    ax1.scatter(x1, y1, c='red', s=38, alpha=0.8, label='Class 1 (dominant)')
    ax1.scatter(x2, y2, c='blue', s=38, alpha=0.8, label='Class 2')
    ax1.scatter(x3, y3, c='green', s=38, alpha=0.8, label='Class 3')
if show_centers:
    for i, (xc, yc) in enumerate(centers1):
        ax1.scatter(xc, yc, marker='o', s=160, c=['red', 'blue', 'green'][i],
                    edgecolor='black', linewidth=2, zorder=10, label=f'Center {i+1}')
ax1.set_title('Case 1: Dominant class, points clustered', fontsize=13)
ax1.set_aspect('equal')
ax1.set_xlim(-1.2, 1.2)
ax1.set_ylim(-1.2, 1.2)
ax1.axis('off')
ax1.grid(True, alpha=0.25)
ax1.legend(loc='best')

## ==== Case 2: 균등 클래스, 완전히 균등하게 분포, 센터 표시 ====
n_eq = 60
arc_width = 2 * np.pi / 3  # 120도 == 클래스당 1/3 원
std_eq = np.deg2rad(11)    # 각 클래스 안에서 밀집 정도 (적당히 분산)
arc_centers2 = [np.pi/6, np.pi, 5*np.pi/3]
colors = ['red', 'blue', 'green']
labels = ['Class 1', 'Class 2', 'Class 3']
points_case2 = []

for i, center_angle in enumerate(arc_centers2):
    # 각도 범위 계산 (균등하게 120도씩)
    arc_start = center_angle - arc_width / 2
    arc_end = center_angle + arc_width / 2
    angles = np.random.normal(center_angle, std_eq, n_eq)
    angles = np.clip(angles, arc_start, arc_end)  # 할당 호 안에서만 분포
    x = np.cos(angles)
    y = np.sin(angles)
    points_case2.append((x, y))
    if show_points:
        ax2.scatter(x, y, c=colors[i], s=38, alpha=0.8, label=labels[i])
    if show_centers:
        xc, yc = np.cos(center_angle), np.sin(center_angle)
        ax2.scatter(xc, yc, marker='o', s=160, c=colors[i], edgecolor='black', linewidth=2, zorder=10, label=f'Center {i+1}')

ax2.plot(np.cos(np.linspace(0, 2 * np.pi, 400)), np.sin(np.linspace(0, 2 * np.pi, 400)), 'k--', linewidth=1)
ax2.set_title('Case 2: Classes clustered, exactly equal arc', fontsize=13)
ax2.set_aspect('equal')
ax2.set_xlim(-1.2, 1.2)
ax2.set_ylim(-1.2, 1.2)
ax2.axis('off')
ax2.grid(True, alpha=0.25)
ax2.legend(loc='best')

plt.tight_layout()
plt.show()
