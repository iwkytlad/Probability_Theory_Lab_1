import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation

# ==========================================
# ПАРАМЕТРЫ СТАКАНЧИКА
# ==========================================
H = 0.13
R_TOP = 0.04
R_BOT = 0.025

CM_FRACTION = 0.40
Y_CM = CM_FRACTION * H
INERTIA_K = 1.3
G = 9.81

angle_side_rad = np.arctan2(H, R_TOP - R_BOT)
angle_side_deg = np.degrees(angle_side_rad)
CRITICAL_ANGLE = angle_side_deg / 2  # ~41.7°

# 0: Дно-Лево, 1: Дно-Право, 2: Верх-Право, 3: Верх-Лево
cup_local = np.array([
    [-R_BOT, -Y_CM],
    [R_BOT, -Y_CM],
    [R_TOP, H - Y_CM],
    [-R_TOP, H - Y_CM]
])


def get_cup_coords(x_pos, y_pos, angle):
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    rotated = (R @ cup_local.T).T
    rotated[:, 0] += x_pos
    rotated[:, 1] += y_pos
    return rotated


# ==========================================
# ВВОД И ДАННЫЕ
# ==========================================
try:
    df = pd.read_csv('throws_100k_fixed.csv')
except FileNotFoundError:
    print("❌ Файл 'throws_100k_fixed.csv' не найден. Запусти Код 1!")
    exit()

print("\n" + "=" * 70)
print("ДЕМОНСТРАЦИЯ АНИМАЦИИ БРОСКА (ИСПРАВЛЕНО)")
print("=" * 70)

while True:
    try:
        user_input = input("\nВведите номер строки (ID) от 1 до 100000 (или 0 для выхода): ")
        row_idx = int(user_input)
        if row_idx == 0: break
        if 1 <= row_idx <= len(df):
            row = df.iloc[row_idx - 1]
            break
        else:
            print(f"⚠️ Неверный ID. Допустимый диапазон: 1 - {len(df)}")
    except ValueError:
        print("️ Пожалуйста, введите целое число.")

theta_deg = np.degrees(row['theta_0_рад'])
theta_final_deg = np.degrees(row['theta_final_рад'])
angle_to_table = row['angle_to_table_град']
outcome_expected = row['Исход']

print("\n" + "=" * 70)
print(f"📊 ДАННЫЕ ИСПЫТАНИЯ № {int(row['ID'])}")
print("=" * 70)
print(f"🎯 ОЖИДАЕМЫЙ ИСХОД : {outcome_expected.upper()}")
print("-" * 70)
print(f"📏 Высота броска (y0)      : {row['y0_м']:.4f} м")
print(f"🔄 Нач. угол (θ0)          : {theta_deg:.2f}°")
print(f"🌀 Угл. скорость (ω)       : {row['omega_рад_с']:.4f} рад/с")
print(f"➡️ Гориз. скорость (Vx)    : {row['v_x_м_с']:.4f} м/с")
print(f"⬆️ Верт. скорость (Vy)     : {row['v_y_м_с']:.4f} м/с")
print(f"📐 Угол при касании (θf)   : {theta_final_deg:.2f}°")
print(f"📐 Угол дна/верха к столу  : {angle_to_table:.2f}°")
print(f"📐 Критический порог       : {CRITICAL_ANGLE:.2f}°")
print("=" * 70)
print("⏳ Генерация анимации...")

# ==========================================
# РАСЧЕТ ТРАЕКТОРИИ
# ==========================================
y0 = row['y0_м']
theta0 = row['theta_0_рад']
omega = row['omega_рад_с']
vx = row['v_x_м_с']
vy = row['v_y_м_с']

omega_eff = omega / np.sqrt(INERTIA_K)
a_c, b_c, c_c = 0.5 * G, -vy, Y_CM - y0
disc = b_c ** 2 - 4 * a_c * c_c
t_hit = (-b_c + np.sqrt(disc)) / (2 * a_c)

theta_at_hit = theta0 + omega_eff * t_hit

# ==========================================
# ПАРАМЕТРЫ ПЛАВНОЙ АНИМАЦИИ (60 FPS)
# ==========================================
TARGET_FPS = 60
# Коэффициент замедления: 1.0 = реальное время, 0.5 = замедление в 2 раза (красивее)
SLOW_MO_FACTOR = 0.6
anim_duration = (t_hit * 1.3) / SLOW_MO_FACTOR

# Гарантируем минимум 180 кадров, чтобы анимация была идеально плавной
N_FRAMES = max(180, int(anim_duration * TARGET_FPS))
T = np.linspace(0, t_hit * 1.3, N_FRAMES)

# ==========================================
# НАСТРОЙКА ГРАФИКА
# ==========================================
fig, ax = plt.subplots(figsize=(14, 9))

ax.axhline(0, color='#2C3E50', linewidth=4, zorder=1)
ax.fill_between([-0.5, 3.0], -0.15, 0, color='#BDC3C7', zorder=0)

t_traj = np.linspace(0, t_hit, 50)
x_traj = vx * t_traj
y_traj = y0 + vy * t_traj - 0.5 * G * t_traj ** 2
ax.plot(x_traj, y_traj, '--', color='#7F8C8D', linewidth=2, zorder=0, alpha=0.7)

c0 = get_cup_coords(0, y0, theta0)
cup = plt.Polygon(c0, closed=True, fc='#3498DB', ec='black', alpha=0.85, lw=1.5, zorder=2)
ax.add_patch(cup)

corner_markers = []
for i in range(4):
    marker, = ax.plot([c0[i, 0]], [c0[i, 1]], 'o', color='#F39C12', ms=8, zorder=5, markeredgecolor='black')
    corner_markers.append(marker)

cm_dot, = ax.plot([0], [y0], 'k+', ms=15, mew=3, zorder=4)

# Красная линия = Дно/Верх, Зеленая = Бок
line_bt, = ax.plot([], [], color='#E74C3C', linewidth=3, zorder=6)
line_side, = ax.plot([], [], color='#27AE60', linewidth=3, zorder=6)

# Красивая информационная панель в правом верхнем углу (всегда читаема)
result_text = ax.text(0.98, 0.95, 'Ожидание касания...', transform=ax.transAxes,
                      fontsize=12, fontweight='bold', verticalalignment='top', horizontalalignment='right',
                      bbox=dict(boxstyle='round,pad=0.6', facecolor='white', alpha=0.9, edgecolor='#2C3E50', linewidth=2))

touch_dot, = ax.plot([], [], 'ro', ms=12, zorder=5)
touch_txt = ax.annotate('', (0, 0), textcoords="offset points", xytext=(0, 25), ha='center', fontsize=14,
                        fontweight='bold', color='#C0392B')

x_max = max(vx * t_hit * 1.2, 1.5)
y_max = max(y0 + 0.3, 0.8)
ax.set_xlim(-0.5, x_max)
ax.set_ylim(-0.15, y_max)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3, zorder=0)
ax.set_xlabel('Расстояние X (м)', fontsize=12, fontweight='bold')
ax.set_ylabel('Высота Y (м)', fontsize=12, fontweight='bold')

title_text = f"Испытание №{int(row['ID'])} | Исход: {outcome_expected.upper()}\nθ₀={theta_deg:.1f}°, ω={omega:.2f} рад/с"
ax.set_title(title_text, fontsize=14, fontweight='bold', color='#2C3E50', pad=15)

data_text = (f"y₀={row['y0_м']:.3f} м | θ₀={theta_deg:.1f}° | ω={omega:.2f} рад/с | "
             f"Vx={vx:.2f} м/с | Vy={vy:.2f} м/с | θ_f={theta_final_deg:.1f}°")
fig.text(0.5, 0.92, data_text, ha='center', fontsize=11,
         fontfamily='monospace', color='#34495E',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='#ECF0F1', edgecolor='#BDC3C7', alpha=0.9))

stopped = [False]
frozen_pts = [None]


def extend_line_to_border(P0, V, xlim, ylim):
    V_norm = V / np.linalg.norm(V)
    t_vals = []
    if V_norm[0] != 0:
        t_xmin = (xlim[0] - P0[0]) / V_norm[0]
        t_xmax = (xlim[1] - P0[0]) / V_norm[0]
        if t_xmin > 0: t_vals.append(t_xmin)
        if t_xmax > 0: t_vals.append(t_xmax)
    if V_norm[1] != 0:
        t_ymin = (ylim[0] - P0[1]) / V_norm[1]
        t_ymax = (ylim[1] - P0[1]) / V_norm[1]
        if t_ymin > 0: t_vals.append(t_ymin)
        if t_ymax > 0: t_vals.append(t_ymax)
    if t_vals:
        t_min = min(t for t in t_vals if t > 0)
        return P0 + V_norm * t_min
    return P0 + V_norm * 0.5


def frame(i):
    if stopped[0]:
        cup.set_xy(frozen_pts[0])
        for j in range(4):
            corner_markers[j].set_data([frozen_pts[0][j, 0]], [frozen_pts[0][j, 1]])
        return cup, cm_dot, touch_dot, touch_txt, line_bt, line_side, result_text, *corner_markers
    t = T[i]
    x = vx * t
    y = y0 + vy * t - 0.5 * G * t ** 2
    angle = theta0 + omega_eff * t

    pts = get_cup_coords(x, y, angle)
    min_y = np.min(pts[:, 1])

    if min_y <= 0 and not stopped[0]:
        x_hit = vx * t_hit
        y_hit = y0 + vy * t_hit - 0.5 * G * t_hit ** 2
        angle_hit = theta0 + omega_eff * t_hit

        pts_hit = get_cup_coords(x_hit, y_hit, angle_hit)
        min_y_hit = np.min(pts_hit[:, 1])
        pts_hit[:, 1] -= min_y_hit

        stopped[0] = True
        frozen_pts[0] = pts_hit.copy()

        cup.set_ec('#C0392B')
        cup.set_lw(3)

        idx = np.argmin(pts_hit[:, 1])
        P0 = pts_hit[idx]

        touch_dot.set_data([P0[0]], [0])
        touch_txt.xy = (P0[0], 0.06)
        touch_txt.set_text(f'✓ ПЕРВОЕ КАСАНИЕ: {outcome_expected.upper()}!')

        # === ПРАВИЛЬНАЯ ГЕОМЕТРИЯ ===
        # Определяем векторы Дна/Верха и Бока в зависимости от того, какая вершина коснулась стола
        if idx == 0:  # Коснулось Дно-Лево
            P_bt = pts_hit[1]  # Дно
            P_s = pts_hit[3]  # Левый бок
        elif idx == 1:  # Коснулось Дно-Право
            P_bt = pts_hit[0]  # Дно
            P_s = pts_hit[2]  # Правый бок
        elif idx == 2:  # Коснулось Верх-Право
            P_bt = pts_hit[3]  # Верх
            P_s = pts_hit[1]  # Правый бок
        else:  # Коснулось Верх-Лево (idx == 3)
            P_bt = pts_hit[2]  # Верх
            P_s = pts_hit[0]  # Левый бок

        V_bt = P_bt - P0
        V_s = P_s - P0

        # Считаем углы с горизонталью [0, 90]
        ang_bt = np.degrees(np.arctan2(V_bt[1], V_bt[0])) % 180
        ang_bt = min(ang_bt, 180 - ang_bt)

        ang_s = np.degrees(np.arctan2(V_s[1], V_s[0])) % 180
        ang_s = min(ang_s, 180 - ang_s)

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        P_bt_end = extend_line_to_border(P0, V_bt, xlim, ylim)
        P_s_end = extend_line_to_border(P0, V_s, xlim, ylim)

        line_bt.set_data([P0[0], P_bt_end[0]], [P0[1], P_bt_end[1]])
        line_side.set_data([P0[0], P_s_end[0]], [P0[1], P_s_end[1]])

        offset = 0.05
        # === ОБНОВЛЕНИЕ ИНФО-ПАНЕЛИ ПРИ УДАРЕ ===
        result_text.set_text(
            f"ПЕРВОЕ КАСАНИЕ: {outcome_expected.upper()}\n\n"
            f"Угол дна/верха: {ang_bt:.1f}°\n"
            f"Угол бока: {ang_s:.1f}°"
        )

        # Проверка регистрации (оставляем в консоль для отладки)
        if ang_bt < CRITICAL_ANGLE:
            expected = "ВЕРХ" if theta_final_deg >= 180 else "ДНО"
        else:
            expected = "БОК"

        if expected == outcome_expected.upper():
            print(f"✅ Регистрация: {outcome_expected} (АБСОЛЮТНО ВЕРНО)")
        else:
            print(f"❌ ОШИБКА: По геометрии должен быть {expected}, а в таблице {outcome_expected}")

        print(f"\n📐 Угол Дна/Верха (красный): {ang_bt:.1f}° (в таблице: {angle_to_table:.2f}°)")
        print(f"📐 Угол Бока (зеленый)     : {ang_s:.1f}°")

        # Проверка регистрации
        if ang_bt < CRITICAL_ANGLE:
            expected = "ВЕРХ" if theta_final_deg >= 180 else "ДНО"
        else:
            expected = "БОК"

        if expected == outcome_expected.upper():
            print(f"✅ Регистрация: {outcome_expected} (АБСОЛЮТНО ВЕРНО)")
        else:
            print(f"❌ ОШИБКА: По геометрии должен быть {expected}, а в таблице {outcome_expected}")

    cup.set_xy(pts)
    cm_dot.set_data([x], [y])
    for j in range(4):
        corner_markers[j].set_data([pts[j, 0]], [pts[j, 1]])

    return cup, cm_dot, touch_dot, touch_txt, line_bt, line_side, result_text, *corner_markers

anim = FuncAnimation(fig, frame, frames=N_FRAMES, interval=1000/TARGET_FPS, blit=True)
output_filename = f'anim_throw_{int(row["ID"])}.mp4' # <-- Меняем расширение!

writer = FFMpegWriter(fps=60, bitrate=2000)
anim.save(output_filename, writer=writer, dpi=200)
plt.close(fig)

print(f"\n✅ Сохранено: {output_filename}")
print(f"📊 Кадров: {N_FRAMES} | FPS: 60 | Длительность: {N_FRAMES / 60:.1f} сек")