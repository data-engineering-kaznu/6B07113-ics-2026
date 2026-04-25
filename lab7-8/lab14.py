import numpy as np


# ── Среда CartPole (реализована с нуля на физических уравнениях) ───────────────

class CartPole:
    """
    Тележка с шестом. Задача — удерживать шест вертикально,
    двигая тележку влево (0) или вправо (1).

    Состояние: [позиция тележки, скорость тележки, угол шеста, угловая скорость]
    """

    GRAVITY       = 9.8
    MASS_CART     = 1.0
    MASS_POLE     = 0.1
    POLE_HALF_LEN = 0.5
    FORCE         = 10.0
    DT            = 0.02   # шаг времени (секунды)
    MAX_STEPS     = 500

    def reset(self):
        self.state = np.random.uniform(-0.05, 0.05, size=4)
        self.steps = 0
        return self.state.copy()

    def step(self, action):
        x, x_dot, theta, theta_dot = self.state

        force       = self.FORCE if action == 1 else -self.FORCE
        total_mass  = self.MASS_CART + self.MASS_POLE
        pole_mass_l = self.MASS_POLE * self.POLE_HALF_LEN
        cos_t       = np.cos(theta)
        sin_t       = np.sin(theta)

        # Уравнения движения (второй закон Ньютона)
        temp      = (force + pole_mass_l * theta_dot ** 2 * sin_t) / total_mass
        theta_acc = (self.GRAVITY * sin_t - cos_t * temp) / (
            self.POLE_HALF_LEN * (4 / 3 - self.MASS_POLE * cos_t ** 2 / total_mass)
        )
        x_acc = temp - pole_mass_l * theta_acc * cos_t / total_mass

        # Обновление состояния методом Эйлера
        x         += self.DT * x_dot
        x_dot     += self.DT * x_acc
        theta     += self.DT * theta_dot
        theta_dot += self.DT * theta_acc

        self.state  = np.array([x, x_dot, theta, theta_dot])
        self.steps += 1

        # Эпизод заканчивается если тележка выехала за края или шест упал
        done = (
            abs(x)     > 2.4   or
            abs(theta) > 0.2095 or  # ~12 градусов
            self.steps >= self.MAX_STEPS
        )

        return self.state.copy(), 1.0, done  # награда = +1 за каждый шаг


# ── Дискретизация состояний ────────────────────────────────────────────────────
# Переводим непрерывные числа в номера ячеек для Q-таблицы

BINS = [10, 10, 20, 20]  # количество ячеек по каждому из 4 параметров

BOUNDS_LOW  = np.array([-2.4, -4.0, -0.25, -4.0])
BOUNDS_HIGH = np.array([ 2.4,  4.0,  0.25,  4.0])


def discretize(state):
    state   = np.clip(state, BOUNDS_LOW, BOUNDS_HIGH)
    ratios  = (state - BOUNDS_LOW) / (BOUNDS_HIGH - BOUNDS_LOW)
    indices = (ratios * np.array(BINS)).astype(int)
    indices = np.clip(indices, 0, np.array(BINS) - 1)
    return tuple(indices)


# ── Q-таблица ──────────────────────────────────────────────────────────────────
# Форма: (6, 6, 12, 12, 2) — для каждой комбинации состояний и 2 действий

q_table = np.zeros(BINS + [2])  # форма: (10, 10, 20, 20, 2)


# ── Гиперпараметры ─────────────────────────────────────────────────────────────

EPISODES      = 10000  # сколько эпизодов обучаем
LEARNING_RATE = 0.15   # насколько быстро обновляем Q-значения (alpha)
DISCOUNT      = 0.99   # насколько ценим будущие награды (gamma)
EPSILON_START = 1.0    # начало: полное исследование
EPSILON_END   = 0.01   # конец: почти всегда лучшее действие
EPSILON_DECAY = 0.999  # насколько быстро уменьшается epsilon (минимум ~на 4600-м эпизоде)


# ── Обучение ───────────────────────────────────────────────────────────────────

env         = CartPole()
epsilon     = EPSILON_START
rewards_log = []
best_avg    = 0
best_q      = q_table.copy()  # сохраняем лучшую Q-таблицу

print("Обучение агента...")
for episode in range(EPISODES):
    state        = discretize(env.reset())
    total_reward = 0
    done         = False

    while not done:
        # Epsilon-greedy выбор действия
        if np.random.random() < epsilon:
            action = np.random.randint(2)       # случайное (исследование)
        else:
            action = np.argmax(q_table[state])  # лучшее по таблице (эксплуатация)

        next_obs, reward, done = env.step(action)
        next_state = discretize(next_obs)

        # Формула Q-learning
        q_table[state][action] += LEARNING_RATE * (
            reward + DISCOUNT * np.max(q_table[next_state]) - q_table[state][action]
        )

        state        = next_state
        total_reward += reward

    epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
    rewards_log.append(total_reward)

    if (episode + 1) % 500 == 0:
        avg = np.mean(rewards_log[-500:])
        print(f"  Эпизод {episode + 1:5d}/{EPISODES} | Средний результат: {avg:.1f} | Epsilon: {epsilon:.3f}")
        # Сохраняем Q-таблицу если это лучший результат за всё обучение
        if avg > best_avg:
            best_avg = avg
            best_q   = q_table.copy()

print(f"\nЛучший средний результат за обучение: {best_avg:.1f} шагов")
q_table = best_q  # используем лучшую таблицу для тестирования


# ── Тестирование обученного агента ─────────────────────────────────────────────

print("\nТестирование агента (10 эпизодов без случайных действий):")
test_results = []

for i in range(10):
    state = discretize(env.reset())
    steps = 0
    done  = False

    while not done:
        action            = np.argmax(q_table[state])
        next_obs, _, done = env.step(action)
        state             = discretize(next_obs)
        steps            += 1

    test_results.append(steps)
    print(f"  Эпизод {i + 1:2d}: {steps} шагов")

avg_steps = np.mean(test_results)
print(f"\nСредний результат на тесте: {avg_steps:.1f} шагов")
if avg_steps >= 200:
    print("Результат: цель достигнута (>= 200 шагов)")
else:
    print("Результат: цель не достигнута (< 200 шагов)")


