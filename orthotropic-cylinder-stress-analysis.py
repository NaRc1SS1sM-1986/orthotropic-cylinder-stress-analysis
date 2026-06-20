# =====================================================================
# БЛОК 1: ИМПОРТ БИБЛИОТЕК И НАСТРОЙКА ОКРУЖЕНИЯ
# =====================================================================

import numpy as np                 # Для работы с массивами, матрицами и математическими функциями
import matplotlib.pyplot as plt    # Для построения графиков
from scipy.integrate import quad   # Для численного интегрирования гладких функций

# Глобальные настройки для красивого отображения графиков (полезно для диплома)
plt.rcParams['figure.dpi'] = 120   # Повышаем разрешение (четкость) графиков
plt.rcParams['axes.grid'] = True   # Сетка на графиках будет включена по умолчанию
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.alpha'] = 0.7

# =====================================================================
# БЛОК 2: ВХОДНЫЕ ПАРАМЕТРЫ (ИСХОДНЫЕ ДАННЫЕ)
# =====================================================================

# --- 2.1 Упругие свойства материала ---
# Текстурированный титановый сплав (Ti-6Al-4V)
E1, E2, E3 = 115e9, 105e9, 95e9  # Модули Юнга в Па
nu12, nu13, nu23 = 0.33, 0.31, 0.32  # Коэффициенты Пуассона

# --- 2.2 Параметры задачи и нагрузки ---
kappa = 1.2      # Параметр мощности дисклинации (в формулах: \varkappa)
lam = 1.1        # Параметр осевого растяжения (lambda - зарезервировано в Python)
p = 0#50e6          # Внутреннее давление

# --- 2.3 Геометрия цилиндра ---
r0 = 0.5   # Начальный внутренний радиус
r1 = 3.0  # Начальный внешний радиус

# --- 2.4 Параметры для функций плотности дислокаций f(r) ---

# 1) Для степенной функции: f(r) = A * r^beta
A_pow = 5.0
beta_pow = 1.5

# 2) Для синусоидальной функции: f(r) = B * sin(C * r)
B_sin = 2.0
C_sin = 5.0

# 3) Для дельта-функции Дирака (сосредоточенный дефект)
D_dirac = 2.0     # Амплитуда
rd = 1.75 # Радиус залегания дефекта (r0 < rd < r1)

# =====================================================================
# БЛОК 3: ВЫЧИСЛЕНИЕ ЗАВИСИМЫХ КОНСТАНТ (ПО ТВОИМ ФОРМУЛАМ)
# =====================================================================

# --- 3.1 Вычисление недостающих коэффициентов Пуассона ---
# Используем соотношение: nu_ij / Ei = nu_ji / Ej => nu_ji = nu_ij * (Ej / Ei)
nu21 = nu12 * (E2 / E1)
nu31 = nu13 * (E3 / E1)
nu32 = nu23 * (E3 / E2)

det_check = 1 - nu12 * nu21 - nu23 * nu32 - nu13 * nu31 - 2 * nu12 * nu23 * nu31

assert det_check > 0, (
    f"КРИТИЧЕСКАЯ ОШИБКА: Комбинация коэффициентов Пуассона физически невозможна! "
    f"Определитель матрицы деформации равен {det_check:.4f} (должен быть > 0). "
    f"Алгоритм остановлен во избежание сбоя солвера."
)
# =====================================================================

# (Построение матриц, задание системы ДУ, граничных условий и расчет)
print("Проверка пройдена успешно. Свойства материала корректны.")

# --- 3.2 Расчет податливостей и коэффициентов Ar, Aphi, k, Q ---

a11 = (1.0 - nu13*nu31) / E1
a22 = (1.0 - nu23*nu32) / E2
a12 = (nu21 + nu31*nu23) / E2 

# Если Ar и Aphi — это твои обозначения для констант матрицы:
Ar = 1.0 - nu31*(lam - 1)   # Замени на формулу из листа, если Ar — это комбинация
Aphi = 1.0 - nu32*(lam - 1) # Замени на формулу из листа, если Aphi — это комбинация

# Параметр анизотропии
k = kappa * np.sqrt(a11 / a22)

# Константа Q 
Q = kappa * (kappa * Ar - Aphi) / a22 

# --- 3.3 Вспомогательные коэффициенты ---
M_coeff = kappa / (2.0 * k * a22)
Dr_const = Q / (1.0 - k**2)

# =====================================================================
# БЛОК 4: ОПРЕДЕЛЕНИЕ ФУНКЦИЙ ПЛОТНОСТИ ДИСЛОКАЦИЙ f(r)
# =====================================================================

def f_power(r):
    """Степенная плотность: f(r) = A * r^beta"""
    return A_pow * (r ** beta_pow)

def f_sin(r):
    """Синусоидальная плотность: f(r) = B * sin(C * r)"""
    return B_sin * np.sin(C_sin * r)

# Для дельта-функции мы не пишем функцию возврата значения, 
# так как она обрабатывается в интеграле как скачок.
# Мы будем вызывать этот случай через строковый ключ 'dirac'

# =====================================================================
# БЛОК 5: МАТЕМАТИЧЕСКОЕ ЯДРО (ВЫЧИСЛЕНИЕ ИНТЕГРАЛОВ)
# =====================================================================
from scipy.integrate import quad

def compute_phi(r, f_type):
    """
    Вычисляет интегралы Ф1(r) и Ф2(r) согласно формулам (11).
    Интегралы:
    Ф1(r) = int(f(xi) * xi^(1-k))
    Ф2(r) = int(f(xi) * xi^(1+k))
    """
    
    # Задаем подынтегральные функции согласно формуле (11)
    # f(xi) * xi^(1-k)  и  f(xi) * xi^(1+k)
    
    if f_type == 'dirac':
        # Для дельта-функции в точке rd: интеграл — это просто значение функции
        # Если r < rd, интеграл равен 0. Если r >= rd, равен значению f(rd)
        # Учтем, что для дельта-функции мы берем значение в точке rd
        if r >= rd:
            phi1 = D_dirac * (rd**(1.0 - k))
            phi2 = D_dirac * (rd**(1.0 + k))
        else:
            phi1 = 0.0
            phi2 = 0.0
        return phi1, phi2

    elif f_type == 'power':
        # Используем лямбда-функции для интегратора
        phi1, _ = quad(lambda xi: f_power(xi) * (xi**(1.0 - k)), r0, r)
        phi2, _ = quad(lambda xi: f_power(xi) * (xi**(1.0 + k)), r0, r)
        return phi1, phi2

    elif f_type == 'sin':
        phi1, _ = quad(lambda xi: f_sin(xi) * (xi**(1.0 - k)), r0, r)
        phi2, _ = quad(lambda xi: f_sin(xi) * (xi**(1.0 + k)), r0, r)
        return phi1, phi2
    
    else:
        raise ValueError("Неизвестный тип функции f_type")
    
# =====================================================================
# БЛОК 6: РЕШЕНИЕ СИСТЕМЫ И РАСЧЕТ НАПРЯЖЕНИЙ
# =====================================================================

def calculate_stresses(f_type, num_points=100):
    """
    1. Находит константы C1, C2 через граничные условия.
    2. Вычисляет распределение напряжений по радиусу.
    """
    
    # --- 6.1 Решение системы для C1 и C2 ---
    # Значения интегралов на границах r0 и r1
    phi1_0, phi2_0 = compute_phi(r0, f_type) 
    phi1_1, phi2_1 = compute_phi(r1, f_type)
    
    # Расчет частного решения Dr* на границах r0 и r1
    # D_r*(r) = M_coeff * (Ф1(r) * r^k - Ф2(r) * r^(-k)) + Dr_const
    #dr_star_0 = M_coeff * (phi1_0 * (r0**(k-1)) - phi2_0 * (r0**(-k-1))) + Dr_const
    #dr_star_1 = M_coeff * (phi1_1 * (r1**(k-1)) - phi2_1 * (r1**(-k-1))) + Dr_const

    beta1 = a22*k / kappa - a12
    beta2 = a22*k / kappa + a12
    psi = Aphi + Dr_const * (a22 / kappa - a12) 
    b1 = - Dr_const - M_coeff * (r1**(k-1)*phi1_1 - r1**(-k-1)*phi2_1) 
    b2 = - Dr_const - p*lam*psi / kappa
    
    # Матрица системы:
    A = np.array([
        [r1**(k-1), r1**(-k-1)],
        [r0**(k-1)*(1 + p*lam*beta1 / kappa), r0**(-k-1)*(1 - p*lam*beta2 / kappa)]
    ])
    B = np.array([b1, b2])
    
    # Находим C1 и C2
    C1, C2 = np.linalg.solve(A, B)
    
    # --- 6.2 Расчет напряжений по радиусу ---
    r_values = np.linspace(r0, r1, num_points)
    Dr = []
    Dphi = []
    Dz = []
    
    for r in r_values:
        # Интегралы в текущей точке
        phi1, phi2 = compute_phi(r, f_type)
        
        # Общее решение Dr
        dr_star = M_coeff * (phi1 * (r**(k-1)) - phi2 * (r**(-k-1))) + Dr_const
        dr_val = C1 * (r**(k-1)) + C2 * (r**(-k-1)) + dr_star
        
        # Расчет Dphi и Dz
        # ЗДЕСЬ НУЖНЫ ТВОИ ФОРМУЛЫ ИЗ ЛИСТОВ.
        # В качестве примера привожу общую формулу, которая часто встречается:
        #dphi_val = k * (C1 * r**(k-1) - C2 * r**(-k-1)) + M_coeff * (phi1 * (r**k) + phi2 * (r**(-k))) + Q

        dphi_val = (C1*k*r**(k-1) - C2*k*r**(-k-1) + Dr_const) / kappa + (r**(k-1)*phi1 + r**(-k-1)*phi2) / 2 / a22

        dz_val = E3*(lam - 1) + nu31*dr_val + nu32*dphi_val
        
        Dr.append(dr_val)
        Dphi.append(dphi_val)
        Dz.append(dz_val)
        
    return r_values, np.array(Dr), np.array(Dphi), np.array(Dz)    

# =====================================================================
# БЛОК 7: ЗАПУСК РАСЧЕТА И ВИЗУАЛИЗАЦИЯ (ОФОРМЛЕНИЕ)
# =====================================================================
import matplotlib.pyplot as plt

def stress_plot(f_type):
    # Вычисляем размерные напряжения
    r_vals, Dr, Dphi, Dz = calculate_stresses(f_type)
    
    # --- ЭТАП ОБЕЗРАЗМЕРИВАНИЯ ---
    # Чтобы ось X визуально шла ровно от 0 до 1, нормируем радиус по толщине цилиндра.
    r_dimless = (r_vals - r0) / (r1 - r0)
    
    # Нормируем напряжения строго на E1
    Dr_dimless = Dr / E1
    Dphi_dimless = Dphi / E1
    Dz_dimless = Dz / E1
    
    # --- ФОРМИРОВАНИЕ ТЕКСТА ДЛЯ ЗАГОЛОВКОВ ---
    # Задаем вид функции и собираем строку с константами
    if f_type == 'power':
        func_title = r"Безразмерные напряжения для плотности дислокаций: $f(r) = A \cdot r^\beta$"
        params_title = f"$p={p}$, $\\lambda={lam}$, $\\kappa={kappa}$ | $A={A_pow}$, $\\beta={beta_pow}$"
    elif f_type == 'sin':
        func_title = r"Безразмерные напряжения для плотности дислокаций: $f(r) = B \cdot \sin(C \cdot r)$"
        params_title = f"$p={p}$, $\\lambda={lam}$, $\\kappa={kappa}$ | $B={B_sin}$, $C={C_sin}$"
    elif f_type == 'dirac':
        func_title = r"Безразмерные напряжения для плотности дислокаций: $f(r) = D \cdot \delta(r - r_d)$"
        params_title = f"$p={p}$, $\\lambda={lam}$, $\\kappa={kappa}$ | $D={D_dirac}$, $r_d={rd}$"
    else:
        func_title = "Неизвестная функция"
        params_title = ""

    # --- ПОСТРОЕНИЕ ГРАФИКОВ ---
    plt.figure(figsize=(10, 6))
    
    # Используем LaTeX-разметку для красивых греческих букв в легенде
    plt.plot(r_dimless, Dr_dimless, label=r'$D_r$ (Радиальные)', linewidth=2)
    plt.plot(r_dimless, Dphi_dimless, label=r'$D_\varphi$ (Окружные)', linewidth=2)
    plt.plot(r_dimless, Dz_dimless, label=r'$D_z$ (Осевые)', linestyle='--', linewidth=2)
    
    # Добавляем многострочный заголовок
    plt.title(f"{func_title}\n{params_title}", fontsize=14, pad=15)
    
    # Обновленные подписи осей
    plt.xlabel(r"Относительный радиус, $(r - r_0) / (r_1 - r_0)$", fontsize=12)
    plt.ylabel(r"Относительные напряжения, $D / E_1$", fontsize=12)
    
    # Настройки отображения
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12, loc='best')
    
    # Жестко фиксируем границы оси X от 0 до 1
    plt.xlim(0, 1)
    
    plt.tight_layout()
    plt.show()

# Пример вызова:
stress_plot('power')
stress_plot('sin')
stress_plot('dirac')

# ==========================================
# ПРОДОЛЬНАЯ СИЛА
# =====================================================================
# БЛОК 8: РАСЧЕТ ПРОДОЛЬНОЙ СИЛЫ И ПОСТРОЕНИЕ СЕМЕЙСТВА КРИВЫХ Fz(kappa)
# =====================================================================
from scipy.integrate import simpson

def plot_longitudinal_force(f_type, param_sets, kappa_min=0.9, kappa_max=1.5, num_points_kappa=100):
    """
    Функция пересчитывает константы, находит распределение напряжений Dz
    и интегрирует его для получения суммарной продольной силы Fz в зависимости от kappa.
    
    Параметры:
    f_type           - тип функции дислокаций ('power', 'sin', 'dirac')
    param_sets       - список словарей с параметрами для каждой кривой. 
                       Например: [{'A': 5, 'beta': 1}, {'A': 5, 'beta': 2}]
    kappa_min, max   - диапазон изменения мощности дисклинации (ось X)
    """
    kappa_values = np.linspace(kappa_min, kappa_max, num_points_kappa)
    
    # 1. Объявляем ВСЕ константы глобальными
    global kappa, lam, A_pow, beta_pow, B_sin, C_sin, D_dirac, rd
    global k, Q, M_coeff, Dr_const, Ar, Aphi
    
    # 2. Сохраняем исходные глобальные значения
    orig_kappa = kappa
    orig_lam = lam
    orig_A_pow = A_pow
    orig_beta_pow = beta_pow
    orig_B_sin = B_sin
    orig_C_sin = C_sin
    orig_D_dirac = D_dirac
    orig_rd = rd

    # Для расчета продольной силы торцы жестко закреплены
    lam = 1.0 

    # Повторяем базовые упругие расчеты податливостей
    nu21 = nu12 * (E2 / E1)
    nu31 = nu13 * (E3 / E1)
    nu32 = nu23 * (E3 / E2)
    a11 = (1.0 - nu13*nu31) / E1
    a22 = (1.0 - nu23*nu32) / E2
    a12 = (nu21 + nu31*nu23) / E2

    plt.figure(figsize=(10, 6))

    # --- ЦИКЛ 1: Перебираем наборы параметров (разные линии) ---
    for params in param_sets:
        
        # Динамически обновляем параметры для текущей кривой.
        # Метод .get(ключ, значение_по_умолчанию) берет значение из словаря, 
        # а если ключа нет, оставляет исходное глобальное значение.
        if f_type == 'power':
            A_pow = params.get('A', orig_A_pow)
            beta_pow = params.get('beta', orig_beta_pow)
            line_label = rf'$A = {A_pow}$, $\beta = {beta_pow}$'
            
        elif f_type == 'sin':
            B_sin = params.get('B', orig_B_sin)
            C_sin = params.get('C', orig_C_sin)
            line_label = rf'$B = {B_sin}$, $C = {C_sin}$'
            
        elif f_type == 'dirac':
            D_dirac = params.get('D', orig_D_dirac)
            rd = params.get('rd', orig_rd)
            line_label = rf'$D = {D_dirac}$, $r_d = {rd}$'
            
        Fz_for_current_amp = []

        # --- ЦИКЛ 2: Шагаем по оси X (изменяем мощность дисклинации kappa) ---
        for k_val in kappa_values:
            # Защита от деления на ноль при kappa около нуля (на всякий случай)
            if np.isclose(k_val, 0.0):
                kappa = 1e-8  
            else:
                kappa = k_val
            
            # Динамический пересчет коэффициентов
            Ar = 1.0 - nu31 * (lam - 1.0)
            Aphi = 1.0 - nu32 * (lam - 1.0)
            k = kappa * np.sqrt(a11 / a22)
            Q = kappa * (kappa * Ar - Aphi) / a22 
            M_coeff = kappa / (2.0 * k * a22)
            Dr_const = Q / (1.0 - k**2)
            
            # Расчет напряжений
            r_vals, _, _, Dz = calculate_stresses(f_type, num_points=200)
            
            # Интегрирование Dz * r * 2pi
            integrand = Dz * r_vals
            integral_val = simpson(integrand, x=r_vals)
            Fz = 2.0 * np.pi * integral_val
            
            Fz_for_current_amp.append(Fz)
            
        plt.plot(kappa_values, Fz_for_current_amp, label=line_label, linewidth=2)

    # --- 3. ВОССТАНОВЛЕНИЕ ИСХОДНОГО СОСТОЯНИЯ ---
    kappa = orig_kappa
    lam = orig_lam
    A_pow = orig_A_pow
    beta_pow = orig_beta_pow
    B_sin = orig_B_sin
    C_sin = orig_C_sin
    D_dirac = orig_D_dirac
    rd = orig_rd
    
    # Пересчитываем константы матрицы обратно для старых значений
    Ar = 1.0 - nu31 * (lam - 1.0)
    Aphi = 1.0 - nu32 * (lam - 1.0)
    k = kappa * np.sqrt(a11 / a22)
    Q = kappa * (kappa * Ar - Aphi) / a22 
    M_coeff = kappa / (2.0 * k * a22)
    Dr_const = Q / (1.0 - k**2)

    # --- 4. ОФОРМЛЕНИЕ ГРАФИКА ---
    plt.axhline(0, color='black', linewidth=1.5, linestyle='--', label='$F_z = 0$ (Равновесие)')
    # Отмечаем линию идеального цилиндра (kappa = 1.0), используя raw-строку (префикс r)
    plt.axvline(1.0, color='black', linewidth=0.8, linestyle='-.', alpha=0.6, label=r'Идеальный цилиндр ($\varkappa=1$)')

    titles = {
        'power': r'Зависимость продольной силы $F_z$ от мощности дисклинации $\varkappa$' + '\n' + r'для степенной плотности дислокаций $f(r) = A \cdot r^\beta$',
        'sin': r'Зависимость продольной силы $F_z$ от мощности дисклинации $\varkappa$' + '\n' + r'для синусоидальной плотности дислокаций $f(r) = B \cdot \sin(C \cdot r)$',
        'dirac': r'Зависимость продольной силы $F_z$ от мощности дисклинации $\varkappa$' + '\n' + r'для сосредоточенного дефекта $f(r) = D \cdot \delta(r - r_d)$'
    }
    
    plt.title(titles.get(f_type, 'Зависимость продольной силы'), fontsize=13, pad=15)
    plt.xlabel(r'Мощность дисклинации, $\varkappa$', fontsize=12)
    plt.ylabel(r'Суммарная продольная сила, $F_z$ [Н]', fontsize=12)
    plt.xlim(kappa_min, kappa_max)
    plt.legend(fontsize=11, loc='best')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

# # ГОТОВО
# plot_longitudinal_force('power', param_sets=[
#     {'A': 0.1, 'beta': -1.5},
#     {'A': 0.1, 'beta': -1.0},
#     {'A': 0.1, 'beta': -0.3},
#     {'A': 0.1, 'beta': 0},
#     {'A': 0.1, 'beta': 0.3},
#     {'A': 0.1, 'beta': 1.0},
#     {'A': 0.1, 'beta': 1.5},
# ]) 

# # ГОТОВО
# plot_longitudinal_force('power', param_sets=[
#     {'A': -0.75, 'beta': 0.3},
#     {'A': -0.5, 'beta': 0.3},
#     {'A': -0.1, 'beta': 0.3},
#     {'A': 0.0, 'beta': 0.3},
#     {'A': 0.1, 'beta': 0.3},
#     {'A': 0.5, 'beta': 0.3},
#     {'A': 0.75, 'beta': 0.3},
# ])  

# # ГОТОВО    
# plot_longitudinal_force('sin', param_sets=[
#     {'B': 0.1, 'C': -1.25},
#     {'B': 0.1, 'C': -0.6},
#     {'B': 0.1, 'C': 0.0},
#     {'B': 0.1, 'C': 0.6},
#     {'B': 0.1, 'C': 1.25},
#     {'B': 0.1, 'C': 2.5},
#     {'B': 0.1, 'C': 5.0}
# ])

# # ГОТОВО    
# plot_longitudinal_force('sin', param_sets=[
#     {'B': -2.0, 'C': 1.25},
#     {'B': -1.0, 'C': 1.25},
#     {'B': -0.5, 'C': 1.25},
#     {'B': 0.0, 'C': 1.25},
#     {'B': 0.5, 'C': 1.25},
#     {'B': 1.0, 'C': 1.25},
#     {'B': 2.0, 'C': 1.25}
# ])
    
# # ГОТОВО
# plot_longitudinal_force('dirac', param_sets=[
#     {'D': 2.0, 'rd': 0.5},   # На внутренней границе
#     {'D': 2.0, 'rd': 1.0},   # Ближе к внутренней
#     {'D': 2.0, 'rd': 1.75},  # Строго по центру стенки цилиндра
#     {'D': 2.0, 'rd': 2.5},   # Ближе к внешней
#     {'D': 2.0, 'rd': 3.0}    # На внешней границе
# ])

# # ГОТОВО
# plot_longitudinal_force('dirac', param_sets=[
#     {'D': -2.0, 'rd': 1.75},
#     {'D': -1.0, 'rd': 1.75},
#     {'D': 0.0,  'rd': 1.75}, # Эталон: пройдет через точку (1.0, 0.0)
#     {'D': 1.0,  'rd': 1.75},
#     {'D': 2.0,  'rd': 1.75}
# ])

# =====================================================================
# БЛОК 9: ПОСТРОЕНИЕ СЕМЕЙСТВА КРИВЫХ НАПРЯЖЕНИЙ Dr ПРИ ФИКСИРОВАННОМ KAPPA
# =====================================================================

def plot_radial_stress_family(f_type, param_sets, num_points=200):
    """
    Строит семейство кривых для безразмерного радиального напряжения Dr/E1
    в зависимости от относительного радиуса при фиксированном значении kappa.
    
    Параметры:
    f_type     - тип функции дислокаций ('power', 'sin', 'dirac')
    param_sets - список словарей с изменяемыми параметрами дефектов
    """
    # Объявляем глобальные переменные параметров для динамического обновления
    global A_pow, beta_pow, B_sin, C_sin, D_dirac, rd
    global kappa, lam, k, Q, M_coeff, Dr_const, Ar, Aphi
    
    # Сохраняем исходные глобальные значения, чтобы не испортить их
    orig_A_pow = A_pow
    orig_beta_pow = beta_pow
    orig_B_sin = B_sin
    orig_C_sin = C_sin
    orig_D_dirac = D_dirac
    orig_rd = rd

    # Гарантируем, что базовые константы рассчитаны под текущие глобальные kappa и lam
    Ar = 1.0 - nu31 * (lam - 1.0)
    Aphi = 1.0 - nu32 * (lam - 1.0)
    k = kappa * np.sqrt(a11 / a22)
    Q = kappa * (kappa * Ar - Aphi) / a22 
    M_coeff = kappa / (2.0 * k * a22)
    Dr_const = Q / (1.0 - k**2)

    plt.figure(figsize=(10, 6))

    # Перебираем наборы параметров
    for params in param_sets:
        if f_type == 'power':
            A_pow = params.get('A', orig_A_pow)
            beta_pow = params.get('beta', orig_beta_pow)
            line_label = rf'$A = {A_pow}$, $\beta = {beta_pow}$'
            
        elif f_type == 'sin':
            B_sin = params.get('B', orig_B_sin)
            C_sin = params.get('C', orig_C_sin)
            line_label = rf'$B = {B_sin}$, $C = {C_sin}$'
            
        elif f_type == 'dirac':
            D_dirac = params.get('D', orig_D_dirac)
            rd = params.get('rd', orig_rd)
            line_label = rf'$D = {D_dirac}$, $r_d = {rd}$'
            
        # Расчет радиальных напряжений для текущего набора параметров
        r_vals, Dr, _, _ = calculate_stresses(f_type, num_points=num_points)
        
        # Обезразмеривание (как в блоке 7)
        r_dimless = (r_vals - r0) / (r1 - r0)
        Dr_dimless = Dr / E1
        
        plt.plot(r_dimless, Dr_dimless, label=line_label, linewidth=2)

    # --- ВОССТАНОВЛЕНИЕ ИСХОДНОГО СОСТОЯНИЯ ---
    A_pow = orig_A_pow
    beta_pow = orig_beta_pow
    B_sin = orig_B_sin
    C_sin = orig_C_sin
    D_dirac = orig_D_dirac
    rd = orig_rd

    # --- ОФОРМЛЕНИЕ ГРАФИКА ---
    titles = {
        'power': r'Семейство распределений $D_r$ для степенной плотности $f(r) = A \cdot r^\beta$',
        'sin': r'Семейство распределений $D_r$ для синусоидальной плотности $f(r) = B \cdot \sin(C \cdot r)$',
        'dirac': r'Семейство распределений $D_r$ для дельта-функции $f(r) = D \cdot \delta(r - r_d)$'
    }
    
    main_title = titles.get(f_type, 'Распределение радиального напряжения')
    params_subtitle = rf'Фиксированные параметры: $\varkappa = {kappa}$, $\lambda = {lam}$, $p = {p}$'
    
    plt.title(f"{main_title}\n{params_subtitle}", fontsize=13, pad=15)
    plt.xlabel(r"Относительный радиус, $(r - r_0) / (r_1 - r_0)$", fontsize=12)
    plt.ylabel(r"Относительные радиальные напряжения, $D_r / E_1$", fontsize=12)
    plt.xlim(0, 1)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=11, loc='best')
    plt.tight_layout()
    plt.show()

# ГОТОВО
# plot_radial_stress_family('power', param_sets=[
#     {'A': -2.0, 'beta': 1.5},
#     {'A': -0.5, 'beta': 1.5},
#     {'A': 0.0,  'beta': 1.5}, # Чистый цилиндр без дислокаций
#     {'A': 0.5,  'beta': 1.5},
#     {'A': 2.0,  'beta': 1.5}
# ])

# # ГОТОВО
# plot_radial_stress_family('power', param_sets=[
#     {'A': 0.1, 'beta': -1.5},
#     {'A': 0.1, 'beta': -1.0},
#     {'A': 0.1, 'beta': 0},
#     {'A': 0.1, 'beta': 1.0},
#     {'A': 0.1, 'beta': 1.5},
# ])


# # ГОТОВО    
# plot_radial_stress_family('sin', param_sets=[
#     {'B': 0.1, 'C': -1.25},
#     {'B': 0.1, 'C': -0.6},
#     {'B': 0.1, 'C': 0.0},
#     {'B': 0.1, 'C': 0.6},
#     {'B': 0.1, 'C': 1.25},
#     {'B': 0.1, 'C': 2.5},
#     {'B': 0.1, 'C': 5.0}
# ])

# # ГОТОВО    
# plot_radial_stress_family('sin', param_sets=[
#     {'B': -2.0, 'C': 5.0},
#     {'B': -1.0, 'C': 5.0},
#     {'B': -0.5, 'C': 5.0},
#     {'B': 0.0, 'C': 5.0},
#     {'B': 0.5, 'C': 5.0},
#     {'B': 1.0, 'C': 5.0},
#     {'B': 2.0, 'C': 5.0}
# ])
    
# # ГОТОВО
# plot_radial_stress_family('dirac', param_sets=[
#     {'D': 2.0, 'rd': 0.5},   # На внутренней границе
#     {'D': 2.0, 'rd': 1.0},   # Ближе к внутренней
#     {'D': 2.0, 'rd': 1.75},  # Строго по центру стенки цилиндра
#     {'D': 2.0, 'rd': 2.5},   # Ближе к внешней
#     {'D': 2.0, 'rd': 3.0}    # На внешней границе
# ])

# # ГОТОВО
# plot_radial_stress_family('dirac', param_sets=[
#     {'D': -2.0, 'rd': 1.75},
#     {'D': -1.0, 'rd': 1.75},
#     {'D': 0.0,  'rd': 1.75}, # Эталон: пройдет через точку (1.0, 0.0)
#     {'D': 1.0,  'rd': 1.75},
#     {'D': 2.0,  'rd': 1.75}
# ])

# =====================================================================
# БЛОК 10: ПОСТРОЕНИЕ СЕМЕЙСТВА КРИВЫХ ОКРУЖНЫХ НАПРЯЖЕНИЙ Dphi
# =====================================================================

def plot_circumferential_stress_family(f_type, param_sets, num_points=200):
    """Строит семейство кривых для безразмерного окружного напряжения Dphi/E1"""
    global A_pow, beta_pow, B_sin, C_sin, D_dirac, rd
    global kappa, lam, k, Q, M_coeff, Dr_const, Ar, Aphi
    
    orig_A_pow, orig_beta_pow = A_pow, beta_pow
    orig_B_sin, orig_C_sin = B_sin, C_sin
    orig_D_dirac, orig_rd = D_dirac, rd

    # Пересчет констант под текущие глобальные kappa и lam
    Ar = 1.0 - nu31 * (lam - 1.0)
    Aphi = 1.0 - nu32 * (lam - 1.0)
    k = kappa * np.sqrt(a11 / a22)
    Q = kappa * (kappa * Ar - Aphi) / a22 
    M_coeff = kappa / (2.0 * k * a22)
    Dr_const = Q / (1.0 - k**2)

    plt.figure(figsize=(10, 6))

    for params in param_sets:
        if f_type == 'power':
            A_pow = params.get('A', orig_A_pow)
            beta_pow = params.get('beta', orig_beta_pow)
            line_label = rf'$A = {A_pow}$, $\beta = {beta_pow}$'
        elif f_type == 'sin':
            B_sin = params.get('B', orig_B_sin)
            C_sin = params.get('C', orig_C_sin)
            line_label = rf'$B = {B_sin}$, $C = {C_sin}$'
        elif f_type == 'dirac':
            D_dirac = params.get('D', orig_D_dirac)
            rd = params.get('rd', orig_rd)
            line_label = rf'$D = {D_dirac}$, $r_d = {rd}$'
            
        # ВАЖНО: Извлекаем Dphi (третий элемент из функции)
        r_vals, _, Dphi, _ = calculate_stresses(f_type, num_points=num_points)
        
        r_dimless = (r_vals - r0) / (r1 - r0)
        Dphi_dimless = Dphi / E1
        
        plt.plot(r_dimless, Dphi_dimless, label=line_label, linewidth=2)

    # Восстановление исходных значений
    A_pow, beta_pow = orig_A_pow, orig_beta_pow
    B_sin, C_sin = orig_B_sin, orig_C_sin
    D_dirac, rd = orig_D_dirac, orig_rd

    titles = {
        'power': r'Семейство распределений $D_\varphi$ для степенной плотности $f(r) = A \cdot r^\beta$',
        'sin': r'Семейство распределений $D_\varphi$ для синусоидальной плотности $f(r) = B \cdot \sin(C \cdot r)$',
        'dirac': r'Семейство распределений $D_\varphi$ для дельта-функции $f(r) = D \cdot \delta(r - r_d)$'
    }
    
    plt.title(f"{titles.get(f_type, 'Окружные напряжения')}\n" + 
              rf'Фиксированные параметры: $\varkappa = {kappa}$, $\lambda = {lam}$, $p = {p}$', 
              fontsize=13, pad=15)
    plt.xlabel(r"Относительный радиус, $(r - r_0) / (r_1 - r_0)$", fontsize=12)
    plt.ylabel(r"Относительные окружные напряжения, $D_\varphi / E_1$", fontsize=12)
    plt.xlim(0, 1)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=11, loc='best')
    plt.tight_layout()
    plt.show()


# # ГОТОВО
# plot_circumferential_stress_family('power', param_sets=[
#     {'A': -2.0, 'beta': 1.5},
#     {'A': -0.5, 'beta': 1.5},
#     {'A': 0.0,  'beta': 1.5}, # Чистый цилиндр без дислокаций
#     {'A': 0.5,  'beta': 1.5},
#     {'A': 2.0,  'beta': 1.5}
# ])

# # ГОТОВО
# plot_circumferential_stress_family('power', param_sets=[
#     {'A': 0.1, 'beta': -1.5},
#     {'A': 0.1, 'beta': -1.0},
#     {'A': 0.1, 'beta': 0},
#     {'A': 0.1, 'beta': 1.0},
#     {'A': 0.1, 'beta': 1.5},
# ])


# # ГОТОВО    
# plot_circumferential_stress_family('sin', param_sets=[
#     {'B': 0.1, 'C': -1.25},
#     {'B': 0.1, 'C': -0.6},
#     {'B': 0.1, 'C': 0.0},
#     {'B': 0.1, 'C': 0.6},
#     {'B': 0.1, 'C': 1.25},
#     {'B': 0.1, 'C': 2.5},
#     {'B': 0.1, 'C': 5.0}
# ])

# # ГОТОВО    
# plot_circumferential_stress_family('sin', param_sets=[
#     {'B': -2.0, 'C': 5.0},
#     {'B': -1.0, 'C': 5.0},
#     {'B': -0.5, 'C': 5.0},
#     {'B': 0.0, 'C': 5.0},
#     {'B': 0.5, 'C': 5.0},
#     {'B': 1.0, 'C': 5.0},
#     {'B': 2.0, 'C': 5.0}
# ])
    
# # ГОТОВО
# plot_circumferential_stress_family('dirac', param_sets=[
#     {'D': 2.0, 'rd': 0.5},   # На внутренней границе
#     {'D': 2.0, 'rd': 1.0},   # Ближе к внутренней
#     {'D': 2.0, 'rd': 1.75},  # Строго по центру стенки цилиндра
#     {'D': 2.0, 'rd': 2.5},   # Ближе к внешней
#     {'D': 2.0, 'rd': 3.0}    # На внешней границе
# ])

# # ГОТОВО
# plot_circumferential_stress_family('dirac', param_sets=[
#     {'D': -2.0, 'rd': 1.75},
#     {'D': -1.0, 'rd': 1.75},
#     {'D': 0.0,  'rd': 1.75}, # Эталон: пройдет через точку (1.0, 0.0)
#     {'D': 1.0,  'rd': 1.75},
#     {'D': 2.0,  'rd': 1.75}
# ])


# =====================================================================
# БЛОК 11: ПОСТРОЕНИЕ СЕМЕЙСТВА КРИВЫХ ОСЕВЫХ НАПРЯЖЕНИЙ Dz
# =====================================================================

def plot_axial_stress_family(f_type, param_sets, num_points=200):
    """Строит семейство кривых для безразмерного осевого напряжения Dz/E1"""
    global A_pow, beta_pow, B_sin, C_sin, D_dirac, rd
    global kappa, lam, k, Q, M_coeff, Dr_const, Ar, Aphi
    
    orig_A_pow, orig_beta_pow = A_pow, beta_pow
    orig_B_sin, orig_C_sin = B_sin, C_sin
    orig_D_dirac, orig_rd = D_dirac, rd

    Ar = 1.0 - nu31 * (lam - 1.0)
    Aphi = 1.0 - nu32 * (lam - 1.0)
    k = kappa * np.sqrt(a11 / a22)
    Q = kappa * (kappa * Ar - Aphi) / a22 
    M_coeff = kappa / (2.0 * k * a22)
    Dr_const = Q / (1.0 - k**2)

    plt.figure(figsize=(10, 6))

    for params in param_sets:
        if f_type == 'power':
            A_pow = params.get('A', orig_A_pow)
            beta_pow = params.get('beta', orig_beta_pow)
            line_label = rf'$A = {A_pow}$, $\beta = {beta_pow}$'
        elif f_type == 'sin':
            B_sin = params.get('B', orig_B_sin)
            C_sin = params.get('C', orig_C_sin)
            line_label = rf'$B = {B_sin}$, $C = {C_sin}$'
        elif f_type == 'dirac':
            D_dirac = params.get('D', orig_D_dirac)
            rd = params.get('rd', orig_rd)
            line_label = rf'$D = {D_dirac}$, $r_d = {rd}$'
            
        # ВАЖНО: Извлекаем Dz (четвертый элемент из функции)
        r_vals, _, _, Dz = calculate_stresses(f_type, num_points=num_points)
        
        r_dimless = (r_vals - r0) / (r1 - r0)
        Dz_dimless = Dz / E1
        
        plt.plot(r_dimless, Dz_dimless, label=line_label, linewidth=2)

    A_pow, beta_pow = orig_A_pow, orig_beta_pow
    B_sin, C_sin = orig_B_sin, orig_C_sin
    D_dirac, rd = orig_D_dirac, orig_rd

    titles = {
        'power': r'Семейство распределений $D_z$ для степенной плотности $f(r) = A \cdot r^\beta$',
        'sin': r'Семейство распределений $D_z$ для синусоидальной плотности $f(r) = B \cdot \sin(C \cdot r)$',
        'dirac': r'Семейство распределений $D_z$ для дельта-функции $f(r) = D \cdot \delta(r - r_d)$'
    }
    
    plt.title(f"{titles.get(f_type, 'Осевые напряжения')}\n" + 
              rf'Фиксированные параметры: $\varkappa = {kappa}$, $\lambda = {lam}$, $p = {p}$', 
              fontsize=13, pad=15)
    plt.xlabel(r"Относительный радиус, $(r - r_0) / (r_1 - r_0)$", fontsize=12)
    plt.ylabel(r"Относительные осевые напряжения, $D_z / E_1$", fontsize=12)
    plt.xlim(0, 1)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=11, loc='best')
    plt.tight_layout()
    plt.show()

# # ГОТОВО
# plot_axial_stress_family('power', param_sets=[
#     {'A': -2.0, 'beta': 1.5},
#     {'A': -0.5, 'beta': 1.5},
#     {'A': 0.0,  'beta': 1.5}, # Чистый цилиндр без дислокаций
#     {'A': 0.5,  'beta': 1.5},
#     {'A': 2.0,  'beta': 1.5}
# ])

# # ГОТОВО
# plot_axial_stress_family('power', param_sets=[
#     {'A': 0.1, 'beta': -1.5},
#     {'A': 0.1, 'beta': -1.0},
#     {'A': 0.1, 'beta': 0},
#     {'A': 0.1, 'beta': 1.0},
#     {'A': 0.1, 'beta': 1.5},
# ])


# # ГОТОВО    
# plot_axial_stress_family('sin', param_sets=[
#     {'B': 0.1, 'C': -1.25},
#     {'B': 0.1, 'C': -0.6},
#     {'B': 0.1, 'C': 0.0},
#     {'B': 0.1, 'C': 0.6},
#     {'B': 0.1, 'C': 1.25},
#     {'B': 0.1, 'C': 2.5},
#     {'B': 0.1, 'C': 5.0}
# ])

# # ГОТОВО    
# plot_axial_stress_family('sin', param_sets=[
#     {'B': -2.0, 'C': 5.0},
#     {'B': -1.0, 'C': 5.0},
#     {'B': -0.5, 'C': 5.0},
#     {'B': 0.0, 'C': 5.0},
#     {'B': 0.5, 'C': 5.0},
#     {'B': 1.0, 'C': 5.0},
#     {'B': 2.0, 'C': 5.0}
# ])
    
# # ГОТОВО
# plot_axial_stress_family('dirac', param_sets=[
#     {'D': 2.0, 'rd': 0.5},   # На внутренней границе
#     {'D': 2.0, 'rd': 1.0},   # Ближе к внутренней
#     {'D': 2.0, 'rd': 1.75},  # Строго по центру стенки цилиндра
#     {'D': 2.0, 'rd': 2.5},   # Ближе к внешней
#     {'D': 2.0, 'rd': 3.0}    # На внешней границе
# ])

# # ГОТОВО
# plot_axial_stress_family('dirac', param_sets=[
#     {'D': -2.0, 'rd': 1.75},
#     {'D': -1.0, 'rd': 1.75},
#     {'D': 0.0,  'rd': 1.75}, # Эталон: пройдет через точку (1.0, 0.0)
#     {'D': 1.0,  'rd': 1.75},
#     {'D': 2.0,  'rd': 1.75}
# ])