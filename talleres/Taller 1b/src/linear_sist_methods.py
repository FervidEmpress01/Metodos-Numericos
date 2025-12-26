# -*- coding: utf-8 -*-
"""
Python 3
05 / 07 / 2024
@author: z_tjona

"I find that I don't understand things unless I try to program them."
-Donald E. Knuth
"""

# ----------------------------- logging --------------------------
import logging
from sys import stdout
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s] %(message)s",
    stream=stdout,
    datefmt="%m-%d %H:%M:%S",
)
logging.info(datetime.now())

import numpy as np


# ####################################################################
def eliminacion_gaussiana(A: np.ndarray | list[list[float | int]]) -> dict:
    """Resuelve un sistema de ecuaciones lineales mediante el método de eliminación gaussiana.
    Retorna la solución y las estadísticas de operaciones.

    ## Parameters

    ``A``: matriz aumentada del sistema de ecuaciones lineales. Debe ser de tamaño n-by-(n+1), donde n es el número de incógnitas.

    ## Return

    ``result``: diccionario con:
        - 'x': vector solución.
        - 'stats': conteo de operaciones (cambios, mult_div, sumas_restas).

    """
    if not isinstance(A, np.ndarray):
        logging.debug("Convirtiendo A a numpy array.")
        A = np.array(A, dtype=float)
    assert A.shape[0] == A.shape[1] - 1, "La matriz A debe ser de tamaño n-by-(n+1)."
    n = A.shape[0]

    # Contadores de operaciones
    stats = {'cambios': 0, 'mult_div': 0, 'sumas_restas': 0}

    for i in range(0, n - 1):  # loop por columna

        # --- encontrar pivote
        p = None  # default, first element
        for pi in range(i, n):
            if abs(A[pi, i]) < 1e-12: # Usamos tolerancia en lugar de == 0 estricto
                # must be nonzero
                continue

            if p is None:
                # first nonzero element
                p = pi
                continue

            if abs(A[pi, i]) > abs(A[p, i]):
                p = pi

        if p is None:
            # no pivot found.
            raise ValueError("No existe solución única.")

        if p != i:
            # swap rows
            stats['cambios'] += 1
            logging.debug(f"Intercambiando filas {i} y {p}")
            _aux = A[i, :].copy()
            A[i, :] = A[p, :].copy()
            A[p, :] = _aux

        # --- Eliminación: loop por fila
        for j in range(i + 1, n):
            stats['mult_div'] += 1 # division para obtener m
            m = A[j, i] / A[i, i]
            A[j, i] = 0 # forzamos cero exacto
            
            # Operaciones para el resto de la fila: A[j, i+1:] = A[j, i+1:] - m * A[i, i+1:]
            # Elementos restantes en la fila: (n+1) columnas totales - (i+1) columnas ya procesadas
            n_cols_restantes = (n + 1) - (i + 1)
            
            stats['mult_div'] += n_cols_restantes # multiplicaciones
            stats['sumas_restas'] += n_cols_restantes  # restas
            
            A[j, i+1:] = A[j, i+1:] - m * A[i, i+1:]

        logging.info(f"\n{A}")

    if abs(A[n - 1, n - 1]) < 1e-12:
        raise ValueError("No existe solución única.")

        print(f"\n{A}")
    # --- Sustitución hacia atrás
    solucion = np.zeros(n)
    
    stats['mult_div'] += 1 # division final
    solucion[n - 1] = A[n - 1, n] / A[n - 1, n - 1]

    for i in range(n - 2, -1, -1):
        suma = 0
        for j in range(i + 1, n):
            stats['mult_div'] += 1 # multiplicacion coef * sol
            stats['sumas_restas'] += 1  # suma al acumulador
            suma += A[i, j] * solucion[j]
            
        stats['sumas_restas'] += 1  # resta (b - suma)
        stats['mult_div'] += 1 # division final
        solucion[i] = (A[i, n] - suma) / A[i, i]

    return {'x': solucion, 'stats': stats}


# ####################################################################
def gauss_jordan(A: np.ndarray | list[list[float | int]]) -> dict:
    """Resuelve un sistema de ecuaciones lineales mediante el método de Gauss-Jordan.

    ## Parameters

    ``A``: matriz aumentada del sistema de ecuaciones lineales. n-by-(n+1).

    ## Return

    ``result``: diccionario con:
        - 'x': vector solución.
        - 'stats': conteo de operaciones.
    """
    if not isinstance(A, np.ndarray):
        logging.debug("Convirtiendo A a numpy array.")
        A = np.array(A, dtype=float)
    assert A.shape[0] == A.shape[1] - 1, "La matriz A debe ser de tamaño n-by-(n+1)."
    n = A.shape[0]

    stats = {'cambios': 0, 'mult_div': 0, 'sumas_restas': 0}

    for i in range(n): # loop por columna y diagonal
        
        # --- encontrar pivote
        p = None
        for pi in range(i, n):
            if abs(A[pi, i]) > 1e-12:
                if p is None or abs(A[pi, i]) > abs(A[p, i]):
                    p = pi
        
        if p is None:
            raise ValueError("No existe solución única.")
            
        if p != i:
            # swap rows
            stats['cambios'] += 1
            logging.debug(f"Intercambiando filas {i} y {p}")
            _aux = A[i, :].copy()
            A[i, :] = A[p, :].copy()
            A[p, :] = _aux

        # --- Normalización de la fila pivote
        pivote = A[i, i]
        
        # Dividimos toda la fila por el pivote para obtener 1 en la diagonal
        # Operaciones: (n+1) divisiones (aunque algunas sean sobre 0, se cuentan por vectorización)
        stats['mult_div'] += (n + 1)
        A[i, :] = A[i, :] / pivote
        
        # --- Eliminación: loop por todas las filas (excepto la pivote)
        for j in range(n):
            if i != j:
                factor = A[j, i]
                
                # Fila_j = Fila_j - factor * Fila_i
                stats['mult_div'] += (n + 1) # multiplicaciones
                stats['sumas_restas'] += (n + 1)  # restas
                
                A[j, :] = A[j, :] - factor * A[i, :]
        
        logging.info(f"\n{A}")

    # La solución es la última columna
    solucion = A[:, -1]
    
    return {'x': solucion, 'stats': stats}


# ####################################################################
def descomposicion_LU(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Realiza la descomposición LU de una matriz cuadrada A.
    [IMPORTANTE] No se realiza pivoteo.

    ## Parameters

    ``A``: matriz cuadrada de tamaño n-by-n.

    ## Return

    ``L``: matriz triangular inferior.

    ``U``: matriz triangular superior. Se obtiene de la matriz ``A`` después de aplicar la eliminación gaussiana.
    """

    A = np.array(
        A, dtype=float
    )  # convertir en float, porque si no, puede convertir como entero

    assert A.shape[0] == A.shape[1], "La matriz A debe ser cuadrada."
    n = A.shape[0]

    L = np.zeros((n, n), dtype=float)

    for i in range(0, n):  # loop por columna

        # --- deterimnar pivote
        if A[i, i] == 0:
            raise ValueError("No existe solución única.")

        # --- Eliminación: loop por fila
        L[i, i] = 1
        for j in range(i + 1, n):
            m = A[j, i] / A[i, i]
            A[j, i:] = A[j, i:] - m * A[i, i:]

            L[j, i] = m

        logging.info(f"\n{A}")

    if A[n - 1, n - 1] == 0:
        raise ValueError("No existe solución única.")

    return L, A


# ####################################################################
def resolver_LU(L: np.ndarray, U: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Resuelve un sistema de ecuaciones lineales mediante la descomposición LU.

    ## Parameters

    ``L``: matriz triangular inferior.

    ``U``: matriz triangular superior.

    ``b``: vector de términos independientes.

    ## Return

    ``solucion``: vector con la solución del sistema de ecuaciones lineales.

    """

    n = L.shape[0]

    # --- Sustitución hacia adelante
    logging.info("Sustitución hacia adelante")

    y = np.zeros((n, 1), dtype=float)

    y[0] = b[0] / L[0, 0]

    for i in range(1, n):
        suma = 0
        for j in range(i):
            suma += L[i, j] * y[j]
        y[i] = (b[i] - suma) / L[i, i]

    logging.info(f"y = \n{y}")

    # --- Sustitución hacia atrás
    logging.info("Sustitución hacia atrás")
    sol = np.zeros((n, 1), dtype=float)

    sol[-1] = y[-1] / U[-1, -1]

    for i in range(n - 2, -1, -1):
        logging.info(f"i = {i}")
        suma = 0
        for j in range(i + 1, n):
            suma += U[i, j] * sol[j]
        logging.info(f"suma = {suma}")
        logging.info(f"U[i, i] = {U[i, i]}")
        logging.info(f"y[i] = {y[i]}")
        sol[i] = (y[i] - suma) / U[i, i]

    logging.debug(f"x = \n{sol}")
    return sol


# ####################################################################
def matriz_aumentada(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Construye la matriz aumentada de un sistema de ecuaciones lineales.

    ## Parameters

    ``A``: matriz de coeficientes.

    ``b``: vector de términos independientes.

    ## Return

    ``Ab``: matriz aumentada.

    """
    if not isinstance(A, np.ndarray):
        logging.debug("Convirtiendo A a numpy array.")
        A = np.array(A, dtype=float)
    if not isinstance(b, np.ndarray):
        b = np.array(b, dtype=float)
    assert A.shape[0] == b.shape[0], "Las dimensiones de A y b no coinciden."
    return np.hstack((A, b.reshape(-1, 1)))


# ####################################################################
def separar_m_aumentada(Ab: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Separa la matriz aumentada en la matriz de coeficientes y el vector de términos independientes.

    ## Parameters
    ``Ab``: matriz aumentada.

    ## Return
    ``A``: matriz de coeficientes.
    ``b``: vector de términos independientes.
    """
    logging.debug(f"Ab = \n{Ab}")
    if not isinstance(Ab, np.ndarray):
        logging.debug("Convirtiendo Ab a numpy array")
        Ab = np.array(Ab, dtype=float)
    return Ab[:, :-1], Ab[:, -1].reshape(-1, 1)