from pathlib import Path

import joblib
import pandas as pd

CATEGORIAS = [
    "debt_consolidation",
    "credit_card",
    "home_improvement",
    "small_business",
    "major_purchase",
    "educational",
    "all_other",
]


def pedir_proposito():
    while True:
        opcion = input(
            "Introduce el propósito del préstamo"
            "\n1.- Consolidación de deuda"
            "\n2.- Tarjeta de crédito"
            "\n3.- Reforma del hogar"
            "\n4.- Pequeño negocio"
            "\n5.- Gran compra"
            "\n6.- Estudios"
            "\n7.- Otro"
            "\n > "
        ).strip()

        if opcion.isdigit() and 1 <= int(opcion) <= len(CATEGORIAS):
            return CATEGORIAS[int(opcion) - 1]

        print(f"  Opción no válida. Elige un número del 1 al {len(CATEGORIAS)}.")


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
    purpose = pedir_proposito()
    int_rate = pedir_numero("Introduce la tasa de interés del préstamo, en % (por ejemplo, 12.5)", 1, 30)
    installment = pedir_numero("Introduce la cuota mensual del préstamo", 1, 10_000)
    fico = pedir_numero("Introduce el puntaje de crédito FICO del prestatario (300-850)", 300, 850, entero=True)
    revol_bal = pedir_numero(
        "Introduce el saldo rotativo del prestatario (monto no pagado al final del ciclo de facturación de la tarjeta)",
        0, 10_000_000,
    )
    revol_util = pedir_numero(
        "Introduce la utilización de la línea de crédito, en % (puede superar 100 si se pasó del límite)",
        0, 300,
    )
    inq_last_6mths = pedir_numero(
        "Introduce el número de consultas de acreedores en los últimos 6 meses", 0, 100, entero=True
    )
    pub_rec = pedir_numero(
        "Introduce el número de registros públicos despectivos del prestatario", 0, 100, entero=True
    )

    return pd.DataFrame({
        "purpose": [purpose],
        "int.rate": [int_rate / 100],  # el modelo espera una fracción (0.125), no un porcentaje
        "installment": [installment],
        "fico": [fico],
        "revol.bal": [revol_bal],
        "revol.util": [revol_util],
        "inq.last.6mths": [inq_last_6mths],
        "pub.rec": [pub_rec],
    })


def main():
    datos = pedir_datos()

    ruta_modelo = Path(__file__).resolve().parent / "Resultados" / "modelo_final_prestamos.joblib"
    modelo = joblib.load(ruta_modelo)

    prediccion = modelo.predict(datos)[0]
    resultado = "NO será pagado al completo" if prediccion == 1 else "SÍ será pagado al completo"
    print(f"\nSegún los datos provistos, el préstamo {resultado}.")


if __name__ == "__main__":
    main()