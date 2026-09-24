import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ЗАГРУЗКА ДАННЫХ И НАСТРОЙКИ
csv_file = 'throws_100k_fixed.csv'

try:
    df = pd.read_csv(csv_file)
except FileNotFoundError:
    print(f"Ошибка: Файл '{csv_file}' не найден. Сначала запустите Код 1!")
    exit()

N_total = len(df)
# Теоретические вероятности
p_theory = {
    'Дно': 0.22,
    'Бок': 0.53,
    'Верх': 0.25
}
outcomes = ['Дно', 'Бок', 'Верх']
colors = {'Дно': '#1f77b4', 'Бок': '#2ca02c', 'Верх': '#d62728'}  # Синий, Зеленый, Красный

# ВЫЧИСЛЕНИЕ НАКОПЛЕННЫХ ЧАСТОТ
# Создаем массив номеров испытаний: n = 1, 2, 3, ..., 100000
n_array = np.arange(1, N_total + 1)

cum_freqs = {}
for outcome in outcomes:
    # 1 if X_j == outcome else 0
    indicator = (df['Исход'] == outcome).astype(int)
    # Накопленная сумма, деленная на n
    cum_freqs[outcome] = np.cumsum(indicator) / n_array

# ТАБЛИЦА ДЛЯ ОТЧЕТА (N = 10, 100, 1000, 10000, 100000)
N_values = [10, 100, 1000, 10000, 100000]


for N in N_values:
    idx = N - 1  # Индекс в массиве (начинается с 0)
    p_dno = cum_freqs['Дно'][idx]
    p_bok = cum_freqs['Бок'][idx]
    p_verh = cum_freqs['Верх'][idx]
    print(f"{N:<12} | {p_dno:<10.4f} | {p_bok:<10.4f} | {p_verh:<10.4f} | {p_dno + p_bok + p_verh:<10.4f}")



# ПОСТРОЕНИЕ ГРАФИКА СХОДИМОСТИ
plt.figure(figsize=(12, 7))

for outcome in outcomes:
    # Рисуем кривую накопленной частоты
    plt.plot(n_array, cum_freqs[outcome],
             label=f'Эксп. частота: {outcome}',
             color=colors[outcome],
             linewidth=1.5,
             alpha=0.8)

    # Рисуем горизонтальную линию теоретической вероятности
    plt.axhline(p_theory[outcome],
                color=colors[outcome],
                linestyle='--',
                linewidth=2,
                label=f'Теория ({outcome}): {p_theory[outcome]}')

# Настройка осей и оформления
plt.xscale('log')  # Логарифмическая шкала по X обязательна для такого диапазона!
plt.xticks(N_values, labels=[f'$10^{int(np.log10(n))}$' for n in N_values])  # Красивые подписи 10^1, 10^2...
plt.xlabel('Число компьютерных испытаний (n)', fontsize=12, fontweight='bold')
plt.ylabel('Относительная частота (ˆp_i)', fontsize=12, fontweight='bold')
plt.title('Сходимость относительных частот к теоретическим вероятностям',
          fontsize=14, fontweight='bold', pad=15)

handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), loc='center right', fontsize=11, framealpha=0.9)

plt.grid(True, which="both", ls="--", alpha=0.5)
plt.tight_layout()

# Сохранение и показ
plot_filename = 'convergence_plot.png'
plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
plt.show()
