# Convolutional-Neural-Networks-for-Digit-Recognition-

El objetivo del proyecto es implementar y comparar una red neuronal y dos CNN con arquitecturas distintas 
para el reconocimiento de dígitos, usando un dataset ampliamente conocido, mnist. 

# Requirements

- Python 3.10+
- NumPy, Matplotlib, Pytorch 
- scikit-learn, scikit-multilearn 

En este proyceto se utiliza  Pytorch v2.9.1

## Setup and execution

1. Creación del entorno virtual

Para garantizar un entorno aislado para las dependencias, crea un entorno virtual con el siguiente comando:

```python -m venv nombre_del_entorno```

o

```python3 -m venv nombre_del_entorno```

2. Activación del entorno

Una vez creado el entorno, actívalo con:

```source nombre_del_entorno/bin/activate```

3. Instalación de dependencias

Instala las dependencias necesarias desde requirements.txt usando:

```pip install -r requirements.txt```

4. Ejecución del código

Para ejecutar el codigo:

Cada modelo tiene su archivo propio en donde se encuentra el codigo para su entrenamiento, evaluacion 
y generación de graficas

Modelo 0 (Capa densa):

```python model_0.py```

o

```python3 model_0.py```


Modelo 1 (Primera CNN):


```python model_1.py```

o

```python3 model_1.py```

Modelo 2 (Segunda CNN):


```python model_2.py```

o

```python3 model_2.py```


## Structure

Este proyecto está organizado de la siguiente manera:

```
├── plot_scripts/            # Scripts de visualización
│   └── plots.py             # Funciones para graficar métricas y resultados
├── model_0.py               # Definición de arquitectura MLP - Modelo 0
├── model_1.py               # Primera CNN - Modelo 1
├── model_2.py               # Segunda CNN - Modelo 2
├── preprocess.py            # Preprocesamiento del dataset MNIST por medio de Pytorch
├── README_PROY.md           # Documentación del proyecto
├── requirements.txt         # Dependencias del entorno
└── train_and_test.py        # Script principal de entrenamiento y evaluación

```