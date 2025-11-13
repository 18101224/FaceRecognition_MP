import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)

def sample_on_circle(center_angle, n, spread):
    # center_angle: 중앙 각도(rad), n: 데이터 수, spread: 각도 std
    angles = np.random.normal(center_angle, spread, n)
    x = np.cos(angles)
    y = np.sin(angles)
    return x, y

# === CASE 1: 불균형 & 대표점 불균형 ===
dom_angles = [0, np.pi / 4, np.pi]      # 세 클래스 대표점 (불균형하게)
n_data = [100, 30, 30]                  # dominant: 클래스 0, 적은 포인트: 1, 2
spreads = [0.07, 0.09, 0.22]            # 초록색(class 2)만 std 더 크게!

x0, y0 = sample_on_circle(dom_angles[0], n_data[0], spreads[0])
x1, y1 = sample_on_circle(dom_angles[1], n_data[1], spreads[1])
x2, y2 = sample_on_circle(dom_angles[2], n_data[2], spreads[2])

centers1 = np.array([[np.cos(a), np.sin(a)] for a in dom_angles])

# === CASE 2: 균등 & 대표점 균등 ===
eq_angles = [0, 2*np.pi/3, 4*np.pi/3]   # 120도 간격으로 대표점 균등
n_eq = [50, 50, 50]
spread_eq = [0.07, 0.07, 0.22]          # 초록만 std 더 크게
xq0, yq0 = sample_on_circle(eq_angles[0], n_eq[0], spread_eq[0])
xq1, yq1 = sample_on_circle(eq_angles[1], n_eq[1], spread_eq[1])
xq2, yq2 = sample_on_circle(eq_angles[2], n_eq[2], spread_eq[2])

centers2 = np.array([[np.cos(a), np.sin(a)] for a in eq_angles])

# === Plot ===
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
theta = np.linspace(0, 2*np.pi, 400)

# CASE 1
ax1.plot(np.cos(theta), np.sin(theta), 'k--', linewidth=1)
ax1.scatter(x0, y0, c='red', s=30, alpha=0.7, label='Class 0')
ax1.scatter(x1, y1, c='blue', s=30, alpha=0.7, label='Class 1')
ax1.scatter(x2, y2, c='green', s=30, alpha=0.7, label='Class 2 (wide spread)')
ax1.scatter(centers1[:,0], centers1[:,1], c=['red','blue','green'], edgecolor='black', s=140, marker='o', label='Centers')
ax1.set_title('Case 1: 불균형 분포', fontsize=13)

# CASE 2
ax2.plot(np.cos(theta), np.sin(theta), 'k--', linewidth=1)
ax2.scatter(xq0, yq0, c='red', s=30, alpha=0.7, label='Class 0')
ax2.scatter(xq1, yq1, c='blue', s=30, alpha=0.7, label='Class 1')
ax2.scatter(xq2, yq2, c='green', s=30, alpha=0.7, label='Class 2 (wide spread)')
ax2.scatter(centers2[:,0], centers2[:,1], c=['red','blue','green'], edgecolor='black', s=140, marker='o', label='Centers')
ax2.set_title('Case 2: 균등 분포', fontsize=13)

for ax in [ax1, ax2]:
    ax.set_aspect('equal')
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.axis('off')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
