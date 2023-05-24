'''import simpy
import random

class Computer:
    def __init__(self, env, name, processing_time_mean, processing_time_std):
        self.env = env
        self.name = name
        self.processing_time_mean = processing_time_mean
        self.processing_time_std = processing_time_std
        self.queue = simpy.Resource(env, capacity=1)

    def process_task(self, task):
        processing_time = random.normalvariate(self.processing_time_mean, self.processing_time_std)
        yield self.env.timeout(processing_time)
        print(f"{self.env.now}: Task {task} completed on {self.name}")

def incoming_tasks(env, computers, p1, p2, p3, num_tasks):
    task_num = 0
    while task_num < num_tasks:
        yield env.timeout(random.expovariate(1.0/2.0))
        task_num += 1
        computer = random.choices(computers, weights=[p1, p2, p3])[0]
        with computer.queue.request() as request:
            yield request
            print(f"{env.now}: Task {task_num} started on {computer.name}")
            yield env.process(computer.process_task(task_num))

env = simpy.Environment()
computer1 = Computer(env, "Computer 1", 1, 2)
computer2 = Computer(env, "Computer 2", 3, 1)
computer3 = Computer(env, "Computer 3", 5, 2)
computers = [computer1, computer2, computer3]
env.process(incoming_tasks(env, computers, 0.4, 0.3, 0.3, 1000))
env.run()

queue_lengths = [len(computer.queue.queue) for computer in computers]
max_queue_length = max(queue_lengths)
print(f"Maximum queue length: {max_queue_length}")

utilizations = [1 - computer.queue.count/env.now for computer in computers]
print(f"Computer 1 utilization: {utilizations[0]}")
print(f"Computer 2 utilization: {utilizations[1]}")
print(f"Computer 3 utilization: {utilizations[2]}")

import heapq
import random

# Интервал времени обработки заданий на каждой ЭВМ
processing_times = [7, 3, 5]
processing_time_std = [4, 1, 2]
qq = 4
# Создаем очереди для каждой ЭВМ
queues = [[] for _ in range(3)]

# Функция, которая моделирует обработку задания на ЭВМ
def process(machine):
    # Генерируем время обработки задания на ЭВМ
    processing_time = random.normalvariate(processing_times[machine], processing_time_std[machine])
    # Обрабатываем задание на ЭВМ
    return processing_time

# Функция, которая моделирует поступление заданий в систему
def arrival(machines, queue, event_queue, queue_lengths):
    # Генерируем задание
    for i in range(qq):
        machine = random.choices(range(len(machines)), weights=[0.4, 0.3, 0.3])[0]
        # Помещаем задание в очередь
        queue[machine].append(i)
        # Добавляем событие в очередь событий
        heapq.heappush(event_queue, (machines[machine], machine, i))
        # Обновляем время обработки заданий на ЭВМ
        machines[machine] += 1
        # Сохраняем длину очередей
        queue_lengths[machine].append(len(queue[machine]))

# Запускаем моделирование
machines = [0 for _ in range(3)]
event_queue = []
queue_lengths = [[] for _ in range(3)]
arrival(machines, queues, event_queue, queue_lengths)
for i in range(qq):
    # Извлекаем следующее событие из очереди событий
    event = heapq.heappop(event_queue)
    machine = event[1]
    job = event[2]
    # Обрабатываем задание на ЭВМ
    processing_time = process(machine)
    # Если время обработки задания на ЭВМ меньше, чем время поступления следующего задания в очередь,
    # то задание обрабатывается на ЭВМ, иначе оно остается в очереди
    if processing_time < machines[machine]:
        machines[machine] -= processing_time
        queues[machine].pop(0)
    else:
        # Добавляем событие в очередь событий
        heapq.heappush(event_queue, (machines[machine] + processing_time, machine, job))
    # Задание выполнено
    machines[machine] = 0
    # Обновляем время обработки заданий на каждой ЭВМ
    for j in range(3):
        machines[j] = max(0, machines[j] - 1)
    # Сохраняем длину очередей
    for j in range(3):
        queue_lengths[j].append(len(queues[j]))
    # Выводим изменение очередей
    print("Шаг", i+1)
    for j in range(3):
        print("Очередь на ЭВМ", j+1, ":", queues[j])

# Рассчитываем максимальную длину каждой очереди
max_queue_lengths = [max(queue_lengths[j]) for j in range(3)]
print("Максимальная длина очереди на кажд
ой ЭВМ:", max_queue_lengths)

# Рассчитываем коэффициенты загрузки ЭВМ
total_processing_time = [processing_times[j] * (qq - max_queue_lengths[j]) for j in range(3)]
total_time = sum(total_processing_time)
utilization_factors = [processing_time / total_time for processing_time in total_processing_time]
print("Коэффициенты загрузки ЭВМ:", utilization_factors)'''

import heapq
import random

# Интервал времени обработки заданий на каждой ЭВМ
processing_times = [7, 3, 5]
processing_time_std = [4, 1, 2]

# Создаем очереди для каждой ЭВМ
queues = [[] for _ in range(3)]

# Функция, которая моделирует обработку задания на ЭВМ
def process(machine):
    # Генерируем время обработки задания на ЭВМ
    processing_time = random.normalvariate(processing_times[machine], processing_time_std[machine])
    # Обрабатываем задание на ЭВМ
    return processing_time

# Функция, которая моделирует поступление заданий в систему
def arrival(machines, queue, event_queue, queue_lengths):
    # Генерируем задание
    for i in range(200):
        machine = random.choices(range(len(machines)), weights=[0.4, 0.3, 0.3])[0]
        # Помещаем задание в очередь
        queue[machine].append(i)
        # Добавляем событие в очередь событий
        heapq.heappush(event_queue, (machines[machine], machine, i))
        # Обновляем время обработки заданий на ЭВМ
        machines[machine] += 1
        # Сохраняем длину очередей
        queue_lengths[machine].append(len(queue[machine]))

# Запускаем моделирование
machines = [0 for i in range(3)]
event_queue = []
queue_lengths = [[] for i in range(3)]
arrival(machines, queues, event_queue, queue_lengths)
busy_machines = [0 for i in range(3)]
total_busy_machines = [0 for i in range(3)]
while event_queue:
    # Извлекаем следующее событие из очереди событий
    event = heapq.heappop(event_queue)
    machine = event[1]
    job = event[2]
    # Обрабатываем задание на ЭВМ
    processing_time = process(machine)
    # Если время обработки задания на ЭВМ меньше, чем время поступления следующего задания в очередь,
    # то задание обрабатывается на ЭВМ, иначе оно остается в очереди
    if processing_time < machines[machine]:
        machines[machine] -= processing_time
        queues[machine].pop(0)
        busy_machines[machine] += processing_time
    else:
        # Добавляем событие в очередь событий
        heapq.heappush(event_queue, (machines[machine] + processing_time, machine, job))
    # Задание выполнено
    machines[machine] = 0
    # Обновляем время обработки заданий на каждой ЭВМ
    for j in range(3):
        machines[j] = max(0, machines[j] - 1)
        if machines[j] > 0:
            busy_machines[j] += 1
    # Сохраняем длину очередей
    for j in range(3):
        queue_lengths[j].append(len(queues[j]))
    # Сохраняем число занятых ЭВМ
    for j in range(3):
        if machines[j] > 0:
            total_busy_machines[j] += 1

# Рассчитываем максимальную длину каждой очереди
max_queue_lengths =[max(queue_lengths[j]) for j in range(3)]
print("Максимальная длина очереди на каждой ЭВМ:", max_queue_lengths)

#Рассчитываем коэффициенты загрузки ЭВМ
total_processing_time = [processing_times[j] * (200 - max_queue_lengths[j]) for j in range(3)]
total_time = sum(total_processing_time)
utilization_factors = [processing_time / total_time for processing_time in total_processing_time]
print("Коэффициенты загрузки ЭВМ:", utilization_factors)

#Рассчитываем среднее число заданий в очереди на каждой ЭВМ
avg_queue_lengths = [sum(queue_lengths[j]) / 200 for j in range(3)]
print("Среднее число заданий в очереди на каждой ЭВМ:", avg_queue_lengths)

#Рассчитываем среднее число занятых ЭВМ
avg_busy_machines = [total_busy_machines[j] / 200 for j in range(3)]
print("Среднее число занятых ЭВМ:", avg_busy_machines)

#Рассчитываем коэффициент простоя ЭВМ
idle_time = sum(machines)
total_time = sum(processing_times) * 200
idle_factor = idle_time / total_time
print("Коэффициент простоя ЭВМ:", idle_factor)