import concurrent
import itertools
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from math import ceil
from typing import Callable, List, Any

import sympy as sp
from sympy import zeros

QC = 384

QC_offsets = [QC * val for val in range(68)]
file = open("7x29.txt", "w")


def parse_matrix_from_file(file_path):
    """
    Вычитывает матрицу из файла
    """
    matrix = []
    with open(file_path, 'r') as f:
        for line in f:
            row = list(map(int, line.split()))
            matrix.append(row)
    return matrix


def sort_matrix_by_rows(mat):
    """

    :param mat: Входящая матрица
    :return: Входящая матрица отсортированная по числу нулевых(=-1) элементов в строке, по возрастанию
    """
    rows = [mat.row(i) for i in range(mat.rows)]
    rows_sorted = sorted(rows, key=lambda row: sum(1 for elem in row if elem != -1))
    sorted_matrix = sp.Matrix(mat.rows, mat.cols, lambda i, j: rows_sorted[i][j])
    return sorted_matrix


def create_sparse_matrix(matrix):
    """

    :param matrix: Матрица по которой хотим построить спарс
    :return: Спарс(матрица, которая хранит индексы не нулевых(-1) элементов)
    Пример:  1  2 -1            0 1
             3  4  3    спарс = 0 1 2
            -1 -1  1            0
    """
    n = matrix.cols
    sparse = [[] for _ in range(n)]

    for row_idx in range(matrix.rows):
        for col_idx in range(n):
            if matrix.row(row_idx)[col_idx] != -1:
                sparse[row_idx].append(col_idx)

    return sparse


def add_to_set(val, values):
    for i in list(values):
        values.remove(i)
        values.add((i + val) % QC)
    return values


def put_to_set(values_set: set, arg):
    for val in arg:
        if val in values_set:
            values_set.remove(val)
        else:
            values_set.add(val)


def add_to_list(val, values, values_count):
    for i in range(len(values) - values_count, len(values)):
        values[i] += val
        values[i] %= QC


def count_permanent(sparse, matrix, used_cols, values, step=0):
    """

    :param sparse: Спарс матрицы
    :param matrix: Матрица для которой считаем перманент
    :param used_cols: Массив уже использованных столбцов
    :param values: Массив в который сохраняем полученный многочлен(храним только степени)
    :param step: Шаг рекурсии
    :return: Число элементов добавленных на данном шаге
    """
    if step + 1 == matrix.rows:  # Проверка находимся ли мы на последней строке матрицы
        for index in sparse[step]:  # Проходимся по столбцам, значения которых != -1 в последней строке
            if used_cols[index] == 1:  # Если столбец уже был использован пропускаем его
                continue
            values.append(matrix.row(step)[index])  # Добавляем в массив
            return 1
        return 0
    count = 0
    for i in range(len(sparse[step])):
        sparse_index = sparse[step][i]
        if used_cols[sparse_index] == 1:
            continue
        used_cols[sparse_index] = 1
        values_add = count_permanent(sparse, matrix, used_cols, values, step + 1)
        add_to_list(matrix.row(step)[sparse_index], values, values_add)
        count += values_add
        used_cols[sparse_index] = 0

    return count


def remove_duplicates(values: list):
    element_counts = Counter(values)
    return [key for key, count in element_counts.items() if count % 2 != 0]


def solve(power_matrix):
    """
    Метод для поиска кодовых слов для подматрицы
    :param power_matrix: отсортированная подматрица показателей
    :return: возвращает мапу вида:
    (Длина кодового слова) -> {Кодовые слова данной длины(кодовые слова представлены индексами единиц)}
    например: 14 -> (768, 818, 855, 1001, 2075, 2081, 2101, 2232, 7400, 7426, 7607, 7644, 8869, 9000), ...
    """

    # Отсекаем ненужные варианты(в которых есть строка из 0)
    required_columns = list()

    for i in range(power_matrix.rows):
        non_zero_indices = set([j for j in range(power_matrix.cols) if power_matrix[i, j] != -1])
        if power_matrix.cols - len(non_zero_indices) >= power_matrix.rows:
            required_columns.append(non_zero_indices)

    elements = list(range(0, power_matrix.cols))
    all_combinations = itertools.combinations(elements, power_matrix.rows + 1)
    all_combinations_c = [comb for comb in all_combinations]

    for required_elements in required_columns:
        all_combinations_c = [comb for comb in all_combinations_c if set(comb).intersection(required_elements)]

    num_threads = 12  # Число используемых потоков

    chunk_size = len(all_combinations_c) // (
            num_threads * 4)  # Задаем размер блока, который будет обрабатывать поток в итерации
    chunks = [all_combinations_c[i:i + chunk_size] for i in
              range(0, len(all_combinations_c), chunk_size)]  # Разбиваем на блоки

    min_perm = float('inf')
    results = {}
    start_time = time.time()

    with ProcessPoolExecutor(max_workers=num_threads) as executor:
        futures = [
            executor.submit(process_combination_chunk, chunk, power_matrix, chunk_index + 1, len(chunks))
            # Вызываем функцию подсчета
            for chunk_index, chunk in enumerate(chunks)
        ]

        # Собираем результаты
        for future in futures:
            local_results, local_min_perm = future.result()
            min_perm = min(min_perm, local_min_perm)
            for key, value in local_results.items():
                if key in results:
                    results[key].extend(value)
                else:
                    results[key] = value

    print("summary time: " + str(time.time() - start_time))
    sorted_result = dict(sorted(results.items(), key=lambda item: item[0]))
    for name in sorted_result:
        file.write(str(name) + ": " + str(len(results[name])) + "\n")
    return sorted_result


def build_code_word(values, qc_number):
    for i in range(len(values)):
        values[i] = QC_offsets[qc_number] + (QC - 1 - values[i])


def process_combination_chunk(combinations_chunk, power_matrix, chunk_index, total_chunks):
    """

    :param combinations_chunk: Комбинации столбцов, которые будем обрабатывать
    :param power_matrix: Исходная матрица в экспоненциальной форме
    :param chunk_index: Номер разбиения(используется только для вывода информации)
    :param total_chunks: Общее число разбиений(используется только для вывода информации)
    :return: Полученные кодовые слова и длину минимального из них
    """
    start_time = time.time()
    power_matrix_copy = power_matrix.copy()
    used_rows = power_matrix_copy.cols * [0]
    min_perm = sys.maxsize
    index = 0

    local_results = {}

    # Проходимся по всем разбиениям
    for combination in combinations_chunk:
        index += 1
        sum_perm = 0
        sub_combinations = combinations(combination,
                                        len(combination) - 1)  # Выбираем под комбинации (изначально комбинации содержат число столбцов + 1), тут мы получаем разбиения длины числа столбцов

        code_word = []
        for sub_combination in sub_combinations:
            permanent = []
            sub_matrix = power_matrix_copy[:, sub_combination]  # Строим нужную подматрицу

            sparse = create_sparse_matrix(sub_matrix)  # Строим спарс

            count_permanent(sparse, sub_matrix, used_rows,
                            permanent)  # Считаем перманент, результат лежит в массиве permanent

            permanent = remove_duplicates(permanent)  # Удаляем повторы(так как по модулю 2)

            # Следующие 3 строки для сохранения кодового слова
            cyrc_number = tuple((set(combination) - set(sub_combination)))[0]  # Номер циркулянта
            build_code_word(permanent, cyrc_number)  # Строим кодовое слово
            code_word.extend(permanent.copy())  # Сохраняем

            sum_perm += len(permanent)  # Суммируем длину полученного перманента

        if sum_perm != 0:
            min_perm = min(min_perm, sum_perm)
            # Сохраняем полученные кодовые слова
            if sum_perm in local_results:
                local_results[sum_perm].append(code_word)
            else:
                local_results[sum_perm] = [code_word]
    print(str(chunk_index) + '/' + str(total_chunks) + ' time: ' + str(time.time() - start_time))
    return local_results, min_perm


def build_cycle_matrix(val):
    cycle_matrix = zeros(QC, QC)
    if val == -1:
        return cycle_matrix
    for index in range(cycle_matrix.rows):
        cycle_matrix[index, val] = 1
        val += 1
        val %= QC
    return cycle_matrix


def build_full_matrix(matr):
    full_matrix = sp.Matrix()
    for row_index in range(matr.rows):
        row_matrix = sp.Matrix()
        for val in matr.row(row_index):
            cycle_matrix = build_cycle_matrix(val)
            row_matrix = row_matrix.row_join(cycle_matrix)
        full_matrix = full_matrix.col_join(row_matrix)
    return full_matrix


def build_transposed_vector(polynom, size):
    code_word = zeros(size, 1)
    for val in polynom:
        code_word[val, 0] = 1
    return code_word


ROWS = 46


def low_grades(values):
    min_grade = min(values) % QC
    for i in range(len(values)):
        dif = (values[i] % QC) - min_grade
        if dif < 0:
            values[i] += QC - min_grade
        else:
            values[i] -= min_grade
    return values


def module_two(matr):
    return matr.applyfunc(lambda x: x % 2)


def is_matrix_zero(matr):
    """
    Проверяет состоит ли матрица только из нулей, возвращает true - если да, false - иначе
    """
    return all(x == 0 for x in matr)


def split_results(results, num_groups):
    """Разделить значения словаря на заданное количество групп."""
    all_values = []
    for value_list in results.values():
        all_values.extend(value_list)

    # Определяем размер каждой группы
    group_size = ceil(len(all_values) / num_groups)

    # Разделяем массив на группы
    return [all_values[i:i + group_size] for i in range(0, len(all_values), group_size)]


def process_group(group, transp_T, QC, block_cols, group_index, groups_count):
    """Обработать одну группу результатов."""
    print(str(group_index) + "/" + str(groups_count))
    group_set = set()
    for code_word in group:

        block_map = {}
        for num in code_word:
            block_key = num // QC
            block_map.setdefault(block_key, []).append(num % QC)

        all_values = []
        for col_index in range(transp_T.cols):
            offset = QC_offsets[col_index + block_cols]
            col_value = transp_T.col(col_index)
            block_col_values = {block_index: col_value[block_index] for block_index in block_map if
                                col_value[block_index] != -1}

            all_values.extend(
                (value - block_col_values[block_index]) % QC + offset
                for block_index, values in block_map.items()
                if block_index in block_col_values
                for value in values
            )

        value_counts = {}
        for value in all_values:
            value_counts[value] = value_counts.get(value, 0) + 1

        full_code_word = [value for value, count in value_counts.items() if count % 2 == 1]
        new_code_word = code_word + full_code_word

        if len(new_code_word) < 100:
            low = low_grades(new_code_word)
            low.sort()
            group_set.add(tuple(low))

    return group_set


if __name__ == '__main__':
    file_path = 'BG1_46x68_QC384.txt'
    matrix = parse_matrix_from_file(file_path)
    matr = sp.Matrix(matrix)

    block_rows = 7
    block_cols = 29

    sub_matrix = matr[range(0, block_rows), range(0, block_cols)]  # Подматрица размера block_rows X block_cols
    matrix_sympy = sort_matrix_by_rows(sub_matrix)

    full_matrix = build_full_matrix(matr)  # Полная матрица (переход от матрицы показателей)
    sub_full_matrix = build_full_matrix(sub_matrix)  # Полная подматрица

    T = matr[range(block_rows, ROWS), range(0, block_cols)]
    transp_T = T.T  # Транспонированная матрица T (с помощью которой достраиваем кодовые слова)

    results = solve(matrix_sympy)

    t = time.time()
    code_words_set = set()

    threads = 12

    #Все кодовые слова полученные в методе solve, объединяем в один массив и разбиваем на threads * 4 групп
    groups = split_results(results, threads * 4)

    #В 12 потоках поочередно достраиваем слова в группах
    with concurrent.futures.ProcessPoolExecutor(max_workers=threads) as executor:
        futures = [executor.submit(process_group, group, transp_T, QC, block_cols, index + 1, len(groups)) for
                   index, group in enumerate(groups)]

        for future in concurrent.futures.as_completed(futures):
            code_words_set.update(future.result())

    file.write("Cunt of words with length < 100: " + str(len(code_words_set)) + "\n")

    for word in code_words_set:
        file.write(str(word) + "\n")
    print("time: " + str(time.time() - t))

    file.close()
