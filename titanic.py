from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def pedir_fichero(retry=False):
    try:
        ruta_csv = input("Introduce la ruta absoluta al fichero csv que quieres analizar: " if not retry else "Introduce una ruta absoluta válida: ")
        df = pd.read_csv(ruta_csv)
        return df
    except FileNotFoundError:
        print(f"{ruta_csv} no es la ruta a un fichero csv")
        return pedir_fichero(True)


def main():
    df = pedir_fichero()

    ruta_modelo = Path(str(Path(__file__).resolve().parent) + '/Resultados/modelo_final_titanic.joblib').resolve()

    modelo_titanic = joblib.load(ruta_modelo)

    predicciones = pd.Series(modelo_titanic.predict(df))
    df["survived"] = predicciones

    print(df)


def sumar_columnas(data):
    return data.sum(axis=1, keepdims=True)


def enumerar_sexo(data):
    return np.where(data == "male", 0, 1)


def categorizar_edad(data):
    return np.digitize(data, bins=[16, 32, 48, 64], right=True) + 1


if __name__ == '__main__':
    main()