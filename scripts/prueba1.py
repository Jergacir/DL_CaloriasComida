# import os

# datasets = {
#     'Nutrition5k': 'datos/originales/nutrition5k',
#     'Food-101': 'datos/originales/food101',
#     'Food-11': 'datos/originales/food11',
#     'ECUSTFD': 'datos/originales/ecustfd'
# }

# for nombre, ruta in datasets.items():
#     existe = os.path.exists(ruta)
#     print(f"{nombre}: {'✓ OK' if existe else '✗ FALTA'}")

import torch

checkpoint = torch.load('modelos/modelo2_mejor.pth', map_location='cpu')

# Ver tipo
print(f"Tipo: {type(checkpoint)}")

# Si es diccionario, ver claves
if isinstance(checkpoint, dict):
    print(f"Claves: {checkpoint.keys()}")
else:
    print("Es un state_dict directo (OrderedDict)")
