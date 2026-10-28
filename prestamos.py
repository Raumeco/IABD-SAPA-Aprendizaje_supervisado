from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def pedir_datos():
    # ('int_rate', 'installment', 'fico',  'revol_bal', 'revol_util', 'inq_last_6mths', 'pub_rec')
    purpose = input("¿Cuál es el propósito del préstamo?\n1.- Consolidación de deuda\n2.- Tarjeta de crédito\n3.- Reforma hogar\n4.- Pequeño negocio\n5.- Gran compra\n6.- Estudiantil\n7.- Otro")
    return 
    


def main():
    df = pedir_datos()

    ruta_modelo = Path(str(Path(__file__).resolve().parent) + '/Resultados/modelo_final_prestamos.joblib').resolve()

    modelo = joblib.load(ruta_modelo)

    prediccion = modelo.predict(df)

    print(f"Según los datos provistos, con esas condiciones {'NO' if prediccion == 1 else 'SÍ'} será pagado al completo")


def sumar_columnas(data):
    return data.sum(axis=1, keepdims=True)


def enumerar_sexo(data):
    return np.where(data == "male", 0, 1)


def categorizar_edad(data):
    return np.digitize(data, bins=[16, 32, 48, 64], right=True) + 1


if __name__ == '__main__':
    main()