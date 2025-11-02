# DL_CaloriasComida
estimacion-calorias-alimentos/
│
├── datos/
│   ├── sin_procesar/                    # Dataset original sin modificar
│   │   ├── ECUSTFD/
│   │   ├── Food-101/
│   │   └── ...
│   ├── procesados/                      # Datos preprocesados y transformados
│   │   ├── entrenamiento/
│   │   ├── validacion/
│   │   └── prueba/
│   ├── externos/                        # Fuentes de datos externas
│   │   └── bases_nutricionales/
│   └── anotaciones/                     # Archivos de etiquetas
│       ├── entrenamiento.txt
│       ├── validacion.txt
│       └── prueba.txt
│
├── notebooks/
│   ├── 01-analisis-exploratorio.ipynb
│   ├── 02-preprocesamiento-datos.ipynb
│   ├── 03-entrenamiento-modelo.ipynb
│   ├── 04-evaluacion-modelo.ipynb
│   └── 05-pruebas-inferencia.ipynb
│
├── src/                                  # Código fuente principal
│   ├── __init__.py
│   ├── datos/
│   │   ├── __init__.py
│   │   ├── cargador_datos.py           # Carga de datos
│   │   ├── preprocesamiento.py         # Preprocesamiento
│   │   └── aumento_datos.py            # Data augmentation
│   │
│   ├── modelos/
│   │   ├── __init__.py
│   │   ├── clasificador_cnn.py         # Modelo de clasificación
│   │   ├── mask_rcnn.py                # Modelo de segmentación
│   │   ├── estimador_volumen.py        # Estimación de volumen
│   │   └── arquitecturas/
│   │       ├── resnet.py
│   │       ├── mobilenet.py
│   │       └── inception.py
│   │
│   ├── entrenamiento/
│   │   ├── __init__.py
│   │   ├── entrenar.py                 # Script principal de entrenamiento
│   │   ├── entrenador.py               # Clase Entrenador
│   │   └── callbacks.py                # Callbacks personalizados
│   │
│   ├── evaluacion/
│   │   ├── __init__.py
│   │   ├── evaluar.py                  # Evaluación del modelo
│   │   └── metricas.py                 # Métricas personalizadas
│   │
│   ├── inferencia/
│   │   ├── __init__.py
│   │   ├── predecir.py                 # Predicciones
│   │   └── calculador_calorias.py      # Cálculo de calorías
│   │
│   └── utilidades/
│       ├── __init__.py
│       ├── configuracion.py            # Configuraciones
│       ├── funciones_auxiliares.py     # Funciones de ayuda
│       └── visualizacion.py            # Visualización de resultados
│
├── modelos/                              # Modelos entrenados guardados
│   ├── checkpoints/
│   │   ├── modelo_epoca_10.pth
│   │   └── modelo_epoca_20.pth
│   ├── mejor_modelo.pth
│   └── modelo_final.pth
│
├── configuraciones/
│   ├── config.yaml                      # Configuración principal
│   ├── config_modelo.yaml               # Configuración del modelo
│   ├── config_entrenamiento.yaml        # Configuración de entrenamiento
│   └── config_datos.yaml                # Configuración de datos
│
├── registros/
│   ├── registros_entrenamiento/
│   ├── tensorboard/
│   └── experimentos/
│
├── resultados/
│   ├── graficas/                        # Gráficas y visualizaciones
│   ├── predicciones/                    # Predicciones guardadas
│   └── reportes/                        # Reportes de evaluación
│
├── pruebas/
│   ├── __init__.py
│   ├── test_cargador_datos.py
│   ├── test_modelos.py
│   └── test_preprocesamiento.py
│
├── scripts/
│   ├── descargar_dataset.sh             # Script para descargar datasets
│   ├── preprocesar_datos.py             # Preprocesamiento por lotes
│   └── entrenar_modelo.sh               # Script de entrenamiento
│
├── documentacion/
│   ├── README.md
│   ├── arquitectura_modelo.md
│   └── documentacion_api.md
│
├── requirements.txt                      # Dependencias del proyecto
├── environment.yml                       # Entorno Conda
├── setup.py                             # Instalación del paquete
└── README.md
