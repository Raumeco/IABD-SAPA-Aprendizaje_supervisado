from pathlib import Path

import joblib
import pandas as pd


def pedir_numero(mensaje, minimo, maximo, entero=False):
    while True:
        texto = input(f"{mensaje}\n > ").strip().replace(",", ".")
        try:
            valor = int(texto) if entero else float(texto)
        except ValueError:
            print("  Debes escribir un número" + (" entero." if entero else "."))
            continue

        if not minimo <= valor <= maximo:
            print(f"  El valor debe estar entre {minimo} y {maximo}.")
            continue
        return valor


def pedir_datos():
    volume = pedir_numero("Introduce la cilindrada del motor, en cm cúbicos (por ejemplo, 1600)", 500, 8000)
    weight = pedir_numero("Introduce el peso del coche, en kg (por ejemplo, 1200)", 500, 4000)

    return pd.DataFrame({
        "Volume": [volume],
        "Weight": [weight],
    })


def main():
    datos = pedir_datos()

    ruta_modelo = Path(__file__).resolve().parent / "Resultados" / "modelo_final_co2.joblib"
    modelo = joblib.load(ruta_modelo)

    prediccion = modelo.predict(datos)[0]
    print(f"\nSegún los datos provistos, las emisiones estimadas son de {prediccion:.1f} g/km de CO2.")


if __name__ == "__main__":
    main()