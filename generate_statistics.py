import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ПАРАМЕТРЫ СТАКАНЧИКА
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

# Геометрия стаканчика
# 0: Дно-Лево, 1: Дно-Право, 2: Верх-Право, 3: Верх-Лево
cup_local = np.array([
    [-R_BOT, -Y_CM],
    [R_BOT, -Y_CM],
    [R_TOP, H - Y_CM],
    [-R_TOP, H - Y_CM]
])

# ПАРАМЕТРЫ БРОСКА
N_SIM = 100000
Y0_MEAN, Y0_STD = 0.50, 0.05
OMEGA_MAX = 15.0
VX_MEAN, VX_STD = 1.5, 0.3
VY_MEAN, VY_STD = 0.2, 0.4

np.random.seed(42)

# Генерация начальных условий
y0 = np.clip(np.random.normal(Y0_MEAN, Y0_STD, N_SIM), 0.30, 0.70)
theta_0 = np.random.uniform(0, 2 * np.pi, N_SIM)
omega = np.random.uniform(-OMEGA_MAX, OMEGA_MAX, N_SIM)
v_x = np.clip(np.random.normal(VX_MEAN, VX_STD, N_SIM), 0.5, 3.0)
v_y = np.clip(np.random.normal(VY_MEAN, VY_STD, N_SIM), -0.5, 1.0)

# РАСЧЕТ ТРАЕКТОРИИ

# 1. Сначала находим t_max - время, когда ЦМ точно ниже нуля (верхняя граница)
a_coef, b_coef, c_coef = 0.5 * G, -v_y, Y_CM - y0
discriminant = b_coef ** 2 - 4 * a_coef * c_coef
t_max = (-b_coef + np.sqrt(discriminant)) / (2 * a_coef)

# 2. Бинарный поиск точного времени касания нижней точкой
t_low = np.zeros(N_SIM)
t_high = t_max.copy()
t_hit = t_max / 2  # Начальное приближение


# Функция для расчета минимальной Y координаты вершин в момент t
def get_min_y(t):
    x_cm = v_x * t
    y_cm = y0 + v_y * t - 0.5 * G * t ** 2
    theta = theta_0 + (omega / np.sqrt(INERTIA_K)) * t

    cos_t = np.cos(theta)[:, np.newaxis, np.newaxis]
    sin_t = np.sin(theta)[:, np.newaxis, np.newaxis]
    R = np.block([[cos_t, -sin_t], [sin_t, cos_t]])

    # cup_local: (4, 2) -> rotated: (N, 4, 2)
    rotated = np.einsum('nij,kj->nki', R, cup_local)
    pts_world_y = rotated[:, :, 1] + y_cm[:, np.newaxis]
    return np.min(pts_world_y, axis=1)


# 20 итераций дают точность ~1e-6 секунды
for _ in range(20):
    min_y = get_min_y(t_hit)
    # Если min_y <= 0, значит касание произошло либо раньше, либо в этот момент
    mask_early = min_y <= 0
    t_high[mask_early] = t_hit[mask_early]
    t_low[~mask_early] = t_hit[~mask_early]
    t_hit = (t_low + t_high) / 2

# t_hit - время первого касания
x_cm_hit = v_x * t_hit
y_cm_hit = y0 + v_y * t_hit - 0.5 * G * t_hit ** 2
theta_final_rad = theta_0 + (omega / np.sqrt(INERTIA_K)) * t_hit

omega_eff = omega / np.sqrt(INERTIA_K)
theta_final_rad = theta_0 + omega_eff * t_hit

# ГЕОМЕТРИЧЕСКАЯ РЕГИСТРАЦИЯ

# Для каждого броска вычисляем координаты 4 вершин в момент t_hit
# и находим точку касания (минимальный Y)

# Позиция центра масс в момент касания
x_cm = v_x * t_hit
y_cm = y0 + v_y * t_hit - 0.5 * G * t_hit ** 2

# Вычисляем координаты всех 4 вершин для всех бросков
# cup_local: (4, 2), theta_final_rad: (N,)

cos_theta = np.cos(theta_final_rad)
sin_theta = np.sin(theta_final_rad)

# Матрица поворота для каждого броска: (N, 2, 2)
R = np.zeros((N_SIM, 2, 2))
R[:, 0, 0] = cos_theta
R[:, 0, 1] = -sin_theta
R[:, 1, 0] = sin_theta
R[:, 1, 1] = cos_theta

# Поворачиваем все 4 точки для всех бросков
# cup_local: (4, 2) -> (1, 4, 2)
# R: (N, 2, 2) -> (N, 1, 2, 2)
# Результат: (N, 4, 2)
rotated = np.einsum('nij,kj->nki', R, cup_local)

# Смещаем к позиции центра масс
pts_world = rotated.copy()
pts_world[:, :, 0] += x_cm[:, np.newaxis]
pts_world[:, :, 1] += y_cm[:, np.newaxis]

# Находим точку касания (минимальный Y) для каждого броска
# pts_world: (N, 4, 2)
min_y_idx = np.argmin(pts_world[:, :, 1], axis=1)  # (N,) индекс точки касания

# Вычисляем углы сторон со столом
# Для каждой точки касания определяем две стороны

# Массивы для хранения углов
ang_bt = np.zeros(N_SIM)  # Угол стороны дно/верх
ang_s = np.zeros(N_SIM)  # Угол стороны бок

# Для каждого броска вычисляем векторы сторон
for idx_val in range(4):
    mask = (min_y_idx == idx_val)
    if not np.any(mask):
        continue

    # Определяем соседние точки в зависимости от idx
    if idx_val == 0:  # Дно-Лево
        idx_bt = 1  # Дно-Право
        idx_s = 3  # Верх-Лево (левый бок)
    elif idx_val == 1:  # Дно-Право
        idx_bt = 0  # Дно-Лево
        idx_s = 2  # Верх-Право (правый бок)
    elif idx_val == 2:  # Верх-Право
        idx_bt = 3  # Верх-Лево
        idx_s = 1  # Дно-Право (правый бок)
    else:  # idx_val == 3, Верх-Лево
        idx_bt = 2  # Верх-Право
        idx_s = 0  # Дно-Лево (левый бок)

    # Векторы сторон
    P0 = pts_world[mask, idx_val]  # Точка касания
    P_bt = pts_world[mask, idx_bt]  # Вторая точка стороны дно/верх
    P_s = pts_world[mask, idx_s]  # Вторая точка стороны бок

    V_bt = P_bt - P0
    V_s = P_s - P0

    # Углы с горизонталью [0, 180]
    ang_bt_val = np.degrees(np.arctan2(V_bt[:, 1], V_bt[:, 0])) % 180
    ang_bt_val = np.minimum(ang_bt_val, 180 - ang_bt_val)

    ang_s_val = np.degrees(np.arctan2(V_s[:, 1], V_s[:, 0])) % 180
    ang_s_val = np.minimum(ang_s_val, 180 - ang_s_val)

    ang_bt[mask] = ang_bt_val
    ang_s[mask] = ang_s_val

is_vertical = ang_bt < CRITICAL_ANGLE
is_horizontal = ang_bt >= CRITICAL_ANGLE

# Для вертикальных: определяем Дно или Верх по индексу точки касания
is_bottom = is_vertical & ((min_y_idx == 0) | (min_y_idx == 1))
is_top = is_vertical & ((min_y_idx == 2) | (min_y_idx == 3))
is_side = is_horizontal

outcomes = np.select(
    [is_bottom, is_top, is_side],
    ['Дно', 'Верх', 'Бок'],
    default='Ошибка'
)

# Вычисляем угол дна/верха к столу для таблицы
theta_final_deg = np.degrees(theta_final_rad) % 360
theta_final_deg = np.where(theta_final_deg < 0, theta_final_deg + 360, theta_final_deg)

angle_to_table = theta_final_deg % 180
angle_to_table = np.where(angle_to_table > 90, 180 - angle_to_table, angle_to_table)

# СОЗДАНИЕ ТАБЛИЦЫ

df = pd.DataFrame({
    'ID': np.arange(1, N_SIM + 1),
    'y0_м': np.round(y0, 4),
    'theta_0_рад': np.round(theta_0, 4),
    'omega_рад_с': np.round(omega, 4),
    'v_x_м_с': np.round(v_x, 4),
    'v_y_м_с': np.round(v_y, 4),
    't_полета_с': np.round(t_hit, 4),
    'x_смещение_м': np.round(v_x * t_hit, 4),
    'theta_final_рад': np.round(theta_final_rad, 4),
    'theta_final_град': np.round(theta_final_deg, 2),
    'angle_to_table_град': np.round(angle_to_table, 2),
    'ang_bt_град': np.round(ang_bt, 2),  # Угол стороны дно/верх
    'ang_s_град': np.round(ang_s, 2),  # Угол стороны бок
    'touch_idx': min_y_idx,  # Индекс точки касания
    'Исход': outcomes
})

csv_file = 'throws_100k_fixed.csv'
df.to_csv(csv_file, index=False, encoding='utf-8-sig', float_format='%.4f')

# Статистика
stats = df['Исход'].value_counts()
for outcome in ['Дно', 'Бок', 'Верх']:
    count = stats.get(outcome, 0)
    pct = count / N_SIM * 100
    print(f"  {outcome}: {count} ({pct:.1f}%)")

# Визуализация
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes[0, 0].hist(y0, bins=50, edgecolor='black', alpha=0.7)
axes[0, 0].set_xlabel('Высота (м)')
axes[0, 0].set_ylabel('Частота')
axes[0, 0].set_title('Высота броска')
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].hist(omega, bins=50, edgecolor='black', alpha=0.7, color='green')
axes[0, 1].set_xlabel('Omega (рад/с)')
axes[0, 1].set_ylabel('Частота')
axes[0, 1].set_title('Угловая скорость')
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].hist(v_x, bins=50, edgecolor='black', alpha=0.7, color='orange')
axes[1, 0].set_xlabel('Vx (м/с)')
axes[1, 0].set_ylabel('Частота')
axes[1, 0].set_title('Горизонтальная скорость')
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].hist(v_y, bins=50, edgecolor='black', alpha=0.7, color='red')
axes[1, 1].set_xlabel('Vy (м/с)')
axes[1, 1].set_ylabel('Частота')
axes[1, 1].set_title('Вертикальная скорость')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('distributions.png', dpi=300)
plt.show()
