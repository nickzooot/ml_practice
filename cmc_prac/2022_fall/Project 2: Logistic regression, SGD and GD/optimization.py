from oracles import BinaryLogistic
import sys
import time
import scipy as sc
import numpy as np

loss_types = {
    'binary_logistic': BinaryLogistic
}


class GDClassifier:
    """
    Реализация метода градиентного спуска для произвольного
    оракула, соответствующего спецификации оракулов из модуля oracles.py
    """

    def __init__(
            self, loss_function='binary_logistic', step_alpha=1, step_beta=0,
            tolerance=1e-10, max_iter=100, **kwargs
    ):
        """
        loss_function - строка, отвечающая за функцию потерь
        классификатора.
        Может принимать значения:
        - 'binary_logistic' - бинарная логистическая регрессия
        step_alpha - float, параметр выбора шага из текста задания
        step_beta- float, параметр выбора шага из текста задания
        tolerance - точность, по достижении которой, необходимо
        прекратить оптимизацию.
        Необходимо использовать критерий выхода по модулю разности
        соседних значений функции:
        если |f(x_{k+1}) - f(x_{k})| < tolerance: то выход
        max_iter - максимальное число итераций
        **kwargs - аргументы, необходимые для инициализации
        """
        if loss_function not in loss_types:
            raise TypeError("Invalid loss-function")

        if 'l2_coef' in kwargs:
            self.l2_coef = kwargs['l2_coef']
        else:
            self.l2_coef = 0
        if 'X_val' in kwargs:
            self.X_val = kwargs['X_val']
        else:
            self.X_val = None
        if 'y_val' in kwargs:
            self.y_val = kwargs['y_val']
        else:
            self.y_val = None
        self.loss_function = loss_function
        self.step_alpha = step_alpha
        self.step_beta = step_beta
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.w = None

    def fit(self, X, y, w_0=None, trace=False):
        """
        Обучение метода по выборке X с ответами y
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        y - одномерный numpy array
        w_0 - начальное приближение в методе
        trace - переменная типа bool
        Если trace = True, то метод должен вернуть словарь history,
        содержащий информацию
        о поведении метода. Длина словаря history =
        количество итераций + 1 (начальное приближение)
        history['time']: list of floats, содержит
        интервалы времени между двумя итерациями метода
        history['func']: list of floats, содержит
        значения функции на каждой итерации
        (0 для самой первой точки)
        """
        self.w = np.copy(w_0)
        f_cur = self.get_objective(X, y)
        history = dict()
        if trace:
            history['func'] = [f_cur]
            history['time'] = [0.0]
            if not (self.X_val is None or self.y_val is None):
                history['acc'] = [np.mean(self.predict(self.X_val)
                                          == self.y_val)]
        err = sys.maxsize
        niters = 1
        while err >= self.tolerance and niters <= self.max_iter:
            time_start = time.time()
            learning_rate = self.step_alpha / (niters **
                                               self.step_beta)
            grad = self.get_gradient(X, y)
            self.w = self.w - learning_rate * grad
            f_pred = f_cur
            f_cur = self.get_objective(X, y)
            err = abs(f_cur - f_pred)
            niters += 1
            time_end = time.time()
            if trace:
                history['func'].append(f_cur)
                history['time'].append(time_end - time_start)
                if not (self.X_val is None or self.y_val is None):
                    history['acc'].append(
                        np.mean(self.predict(self.X_val) ==
                                self.y_val))
        return history if trace else None

    def predict(self, X):
        """
        Получение меток ответов на выборке X
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        return: одномерный numpy array с предсказаниями
        """
        if self.w is None:
            raise ValueError("First do fitting!")
        return 2 * ((X.dot(self.w)) > 0) - 1

    def predict_proba(self, X):
        """
        Получение вероятностей принадлежности X к классу k
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        return: двумерной numpy array, [i, k] значение
        соответветствует вероятности
        принадлежности i-го объекта к классу k
        """
        p1 = sc.special.expit(X.dot(self.w))
        p0 = 1 - p1
        return np.array([p0, p1]).T

    def get_objective(self, X, y):
        """
        Получение значения целевой функции на выборке X с ответами y
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        y - одномерный numpy array
        return: float
        """
        loss_class = loss_types[self.loss_function](self.l2_coef)
        return loss_class.func(X, y, self.w)

    def get_gradient(self, X, y):
        """
        Получение значения градиента функции на выборке X с
        ответами y
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        y - одномерный numpy array
        return: numpy array, размерность зависит от задачи
        """
        loss_class = loss_types[self.loss_function](self.l2_coef)
        return loss_class.grad(X, y, self.w)

    def get_weights(self):
        """
        Получение значения весов функционала
        """
        return self.w


class SGDClassifier(GDClassifier):
    """
    Реализация метода стохастического градиентного спуска для
    произвольного
    оракула, соответствующего спецификации оракулов из модуля
    oracles.py
    """

    def __init__(
            self, loss_function='binary_logistic', batch_size=10000,
            step_alpha=1, step_beta=0,
            tolerance=1e-10, max_iter=100, random_seed=153, **kwargs
    ):
        """
        loss_function - строка, отвечающая за функцию потерь
        классификатора.
        Может принимать значения:
        - 'binary_logistic' - бинарная логистическая регрессия
        batch_size - размер подвыборки, по которой считается
        градиент
        step_alpha - float, параметр выбора шага из текста задания
        step_beta- float, параметр выбора шага из текста задания
        tolerance - точность, по достижении которой, необходимо
        прекратить оптимизацию
        Необходимо использовать критерий выхода по модулю разности
        соседних значений функции:
        если |f(x_{k+1}) - f(x_{k})| < tolerance: то выход
        max_iter - максимальное число итераций (эпох)
        random_seed - в начале метода fit необходимо вызвать
        np.random.seed(random_seed).Этот параметр нужен для
        воспроизводимости результатов на разных машинах.
        **kwargs - аргументы, необходимые для инициализации
        """
        super().__init__(
            loss_function=loss_function,
            step_alpha=step_alpha,
            step_beta=step_beta,
            tolerance=tolerance,
            max_iter=max_iter,
            **kwargs
        )
        self.batch_size = batch_size
        self.random_seed = random_seed
        self.loss_function = loss_function
        self.w = None

    def fit(self, X, y, w_0=None, trace=False, log_freq=1):
        """
        Обучение метода по выборке X с ответами y
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        y - одномерный numpy array
        w_0 - начальное приближение в методе
        Если trace = True, то метод должен вернуть словарь history,
        содержащий информацию
        о поведении метода. Если обновлять history после каждой
        итерации, метод перестанет
        превосходить в скорости метод GD. Поэтому, необходимо
        обновлять историю метода лишь
        после некоторого числа обработанных объектов в зависимости
        от приближённого номера эпохи.
        Приближённый номер эпохи:
            {количество объектов, обработанных методом SGD} /
            {количество объектов в выборке}
        log_freq - float от 0 до 1, параметр, отвечающий за частоту
        обновления.
        Обновление должно проиходить каждый раз, когда разница
        между двумя значениями приближённого номера эпохи
        будет превосходить log_freq.
        history['epoch_num']: list of floats, в каждом элементе
        списка будет записан приближённый номер эпохи:
        history['time']: list of floats, содержит интервалы времени
        между двумя соседними замерами
        history['func']: list of floats, содержит значения функции
        после текущего приближённого номера эпохи
        history['weights_diff']: list of floats, содержит квадрат
        нормы разности векторов весов с соседних замеров
        (0 для самой первой точки)
        """
        self.w = np.copy(w_0)
        f_cur = self.get_objective(X, y)
        history = dict()
        if trace:
            history['epoch_num'] = [0.0]
            history['time'] = [0.0]
            history['func'] = [f_cur]
            history['weights_diff'] = [0.0]
            if not (self.X_val is None or self.y_val is None):
                history['acc'] = [np.mean(self.predict(self.X_val)
                                          == self.y_val)]
        err = sys.maxsize
        rng = np.random.default_rng(seed=self.random_seed)
        X_size = X.shape[0]
        nprocessed_obj = 0
        w_epoch_prev = np.copy(self.w)
        time_start = time.time()
        is_acc = False
        for nepoch in range(1, self.max_iter + 1):
            rand_seq = rng.choice(X_size, size=X_size,
                                  replace=False)
            learning_rate = self.step_alpha / (nepoch **
                                               self.step_beta)
            for i in range(self.batch_size, X_size + 1,
                           self.batch_size):
                X_batch = X[rand_seq[i - self.batch_size:i]]
                y_batch = y[rand_seq[i - self.batch_size:i]]
                gradient = self.get_gradient(X_batch, y_batch)
                self.w -= learning_rate * gradient
                f_cur = self.get_objective(X, y)
                nprocessed_obj += self.batch_size
                approx_nepoch = nprocessed_obj / X_size
                if trace and (approx_nepoch -
                              history['epoch_num'][-1]) >= log_freq:
                    time_end = time.time()
                    err = abs(f_cur - history['func'][-1])
                    history['time'].append(time_end - time_start)
                    history['epoch_num'].append(approx_nepoch)
                    history['func'].append(f_cur)
                    if not (self.X_val is None or self.y_val is
                            None):
                        history['acc'].append((np.mean(self.predict(self.X_val)
                                                       == self.y_val)))
                    norm_sq = (self.w - w_epoch_prev) @ (self.w -
                                                         w_epoch_prev)
                    w_epoch_prev = np.copy(self.w)
                    history['weights_diff'].append(norm_sq)
                    time_start = time.time()
                if err < self.tolerance:
                    is_acc = True
                    break
            if is_acc:
                break
        return history if trace else None
