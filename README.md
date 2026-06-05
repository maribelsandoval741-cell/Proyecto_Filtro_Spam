# DESARROLLO DE UN SISTEMA PREDICTIVO PARA LA DETECCIÓN DE SPAM USANDO REGRESIÓN LOGÍSTICA

Este repositorio contiene una solución de ingeniería de software enfocada en la detección y clasificación automatizada de correos electrónicos no deseados (Spam) utilizando el corpus histórico de **Enron-Spam**. 

El sistema está desarrollado con programación estructurada y modular en Python. Implementa un modelo de **Regresión Logística codificado desde cero** mediante álgebra lineal vectorizada en NumPy, lo que permite demostrar el sustento matemático interno del algoritmo (función sigmoide, función de costo Log-Loss y optimización por gradiente descendiente) sin depender de bibliotecas de Machine Learning de caja negra como Scikit-Learn.

## Requisitos del Sistema
Para compilar y ejecutar esta aplicación de manera local, es necesario contar con el siguiente entorno tecnológico:
* **Python 3.8** o superior instalado en el sistema operativo.
* Gestor de paquetes **pip** actualizado.

## Dependencias Requeridas
El entorno del software minimiza el acoplamiento de librerías externas, utilizando únicamente dos herramientas estándar para la manipulación operativa de los datos:
* **NumPy:** Utilizado como el motor matemático central para realizar operaciones matriciales vectorizadas, cálculo de combinaciones lineales y derivadas parciales.
* **Pandas:** Utilizado exclusivamente para la ingesta, limpieza inicial de valores nulos y renombrado adaptativo de las columnas de la base de datos CSV.

## Estructura del Proyecto
El repositorio organiza sus componentes de manera unificada y simplificada en la raíz del espacio de trabajo:

```text
filtro-spam-logistica/
│
├── email_text.csv         # Archivo de datos preprocesado (Dataset Enron)
└── filtro_spam.py         # Script unificado con la lógica matemática y pruebas