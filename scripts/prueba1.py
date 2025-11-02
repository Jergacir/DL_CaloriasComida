import os

datasets = {
    'Nutrition5k': 'datos/originales/nutrition5k',
    'Food-101': 'datos/originales/food101',
    'Food-11': 'datos/originales/food11',
    'ECUSTFD': 'datos/originales/ecustfd'
}

for nombre, ruta in datasets.items():
    existe = os.path.exists(ruta)
    print(f"{nombre}: {'✓ OK' if existe else '✗ FALTA'}")
