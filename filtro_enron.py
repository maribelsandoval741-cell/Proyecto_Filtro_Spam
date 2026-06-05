import os
import numpy as np
import pandas as pd

class RegresionLogisticaNativa:
    def __init__(self, learning_rate=0.1, epochs=500):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -250, 250)))

    def fit(self, X, y):
        muestras, caracteristicas = X.shape
        self.weights = np.zeros(caracteristicas)
        self.bias = 0

        for epoch in range(self.epochs):
            z = X.dot(self.weights) + self.bias
            y_pred = self.sigmoid(z)

            dw = (1 / muestras) * X.T.dot(y_pred - y)
            db = (1 / muestras) * np.sum(y_pred - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            if epoch % 100 == 0:
                costo = -(1 / muestras) * np.sum(
                    y * np.log(y_pred + 1e-15) +
                    (1 - y) * np.log(1 - y_pred + 1e-15)
                )
                print(f"Iteración {epoch} - Costo: {costo:.4f}")

    def predict_proba(self, X):
        z = X.dot(self.weights) + self.bias
        return self.sigmoid(z)

    def predict(self, X, threshold=0.5):
        return np.where(self.predict_proba(X) >= threshold, 1, 0)

class VectorizadorPalabrasClave:
    def __init__(self, max_features=2000):
        self.max_features = max_features
        self.vocabulario = []

    def fit(self, textos):
        palabras = " ".join(textos.astype(str)).lower().split()
        palabras = [p for p in palabras if len(p) > 3]
        frecuencias = pd.Series(palabras).value_counts()
        self.vocabulario = list(frecuencias.head(self.max_features).index)
        print("Tamaño del vocabulario:", len(self.vocabulario))

    def transform(self, textos):
        matriz = np.zeros((len(textos), len(self.vocabulario)))

        for i, texto in enumerate(textos):
            palabras_texto = str(texto).lower().split()

            for j, palabra in enumerate(self.vocabulario):
                matriz[i, j] = palabras_texto.count(palabra)

        return matriz

def cargar_datos_enron(ruta_archivo):
    if not os.path.exists(ruta_archivo):
        raise FileNotFoundError(f"No se encontró el archivo {ruta_archivo}")

    print("Cargando dataset...")
    df = pd.read_csv(ruta_archivo)

    if 'text' in df.columns:
        df.rename(columns={'text': 'texto'}, inplace=True)

    if 'Message' in df.columns:
        df.rename(columns={'Message': 'texto'}, inplace=True)

    if 'label_num' in df.columns:
        df.rename(columns={'label_num': 'label'}, inplace=True)

    df = df.dropna(subset=['texto']).reset_index(drop=True)

    spam = df[df['label'] == 1].shape[0]
    ham = df[df['label'] == 0].shape[0]

    print("Total registros:", len(df))
    print("Correos normales:", ham)
    print("Correos spam:", spam)

    return df

def entrenar_clasificador_spam(df):
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    X_texto = df["texto"]
    y = df["label"].values

    limite = int(len(df) * 0.8)

    X_train_txt = X_texto[:limite]
    X_test_txt = X_texto[limite:]

    y_train = y[:limite]
    y_test = y[limite:]

    vectorizador = VectorizadorPalabrasClave(max_features=1500)

    print("Generando vocabulario...")
    vectorizador.fit(X_train_txt)

    X_train = vectorizador.transform(X_train_txt)
    X_test = vectorizador.transform(X_test_txt)

    print("Entrenando modelo...")

    modelo = RegresionLogisticaNativa(
        learning_rate=0.1,
        epochs=300
    )

    modelo.fit(X_train, y_train)

    predicciones = modelo.predict(X_test)
    exactitud = np.mean(predicciones == y_test)

    print(f"Exactitud del modelo: {exactitud:.2%}")

    return modelo, vectorizador

def analizar_nuevo_correo(modelo, vectorizador, texto):
    X = vectorizador.transform(pd.Series([texto]))
    pred = modelo.predict(X)[0]
    prob = modelo.predict_proba(X)[0]

    print("\nCorreo:", texto)

    if pred == 1:
        print(f"Resultado: SPAM ({prob:.2%})")
    else:
        print(f"Resultado: NO SPAM ({prob:.2%})")

if __name__ == "__main__":
    try:
        df = cargar_datos_enron("email_text.csv")
        modelo, vectorizador = entrenar_clasificador_spam(df)

        print("\nPruebas del sistema")

        analizar_nuevo_correo(modelo, vectorizador,
            "CONGRATULATIONS! You won a cash prize click here now")

        analizar_nuevo_correo(modelo, vectorizador,
            "Hi, are we still meeting for lunch tomorrow at the office?")

        analizar_nuevo_correo(modelo, vectorizador,
            "URGENT: Your bank account has been blocked. Click here immediately.")

        analizar_nuevo_correo(modelo, vectorizador,
            "Hello, I attached the agenda for the finance meeting tomorrow.")

    except Exception as e:
        print("Error:", e)