# Práctica 3: Árboles de Decisión y Clasificación de Imágenes con Explicabilidad

**Integrantes:** Felipe Peralta, Samantha Suquilanda

---

## Descripción General

Esta práctica explora tres aspectos fundamentales del aprendizaje automático:

1. **Fundamentos teóricos**: Cálculo de ganancia de información en árboles de decisión
2. **Machine Learning clásico**: Clasificación de imágenes con SVM y técnicas de explicabilidad
3. **Deep Learning**: Clasificación de imágenes con CNN y explicabilidad mediante Grad-CAM

### Objetivos de Aprendizaje

- Comprender las medidas de impureza (Entropía, Gini, Error de Clasificación)
- Calcular la Ganancia de Información para seleccionar las mejores divisiones
- Implementar y comparar modelos SVM y CNN para clasificación de imágenes
- Aplicar técnicas de Explicabilidad de IA (XAI) para interpretar predicciones
- Utilizar Grad-CAM para visualizar regiones importantes en clasificación de imágenes

---

## Estructura del Proyecto

```bash
practica-3/
│
├── README.md                           # Este archivo
│
├── data/                               # Datasets
│   ├── fruits-360_100x100/            # Dataset principal Fruits-360
│   │   └── fruits-360/
│   │       ├── Training/              # Imágenes de entrenamiento
│   │       └── Test/                  # Imágenes de prueba
│   └── modelo-ml.ipynb                # Notebook de exploración inicial
│
├── docs/                               # Documentación y recursos teóricos
│   ├── arboles_decision.md            # Teoría de árboles de decisión
│   ├── maximizar_ganancia_informacion.md  # Teoría de ganancia de información
│   ├── ejemplo/
│   │   └── ejemplo_arbol_decision.ipynb
│
├── ejercicio-diapositivas/
│   └── ejercicio2.ipynb               # Ejercicio 1: Ganancia de Información
│
└── ejericio-modelos/                  # Ejercicios principales
    ├── modelo-ml-svm-explicable.ipynb # Ejercicio 2: SVM con explicabilidad
    └── modelo-ml-cnn-explicable.ipynb # Ejercicio 3: CNN con Grad-CAM
```

---

## Ejercicio 1: Maximizar Ganancia de Información

### Descripción

Este ejercicio implementa el cálculo de **Ganancia de Información (IG)** utilizando tres medidas de impureza diferentes para determinar cuál escenario de división es óptimo en un árbol de decisión.

### Conceptos Clave

#### Medidas de Impureza

1. **Entropía (Ih)**

   ```bash
   Ih = -Σ p(i) × log₂(p(i))
   ```

   - Mide el desorden o incertidumbre
   - Mayor entropía = mayor desorden

2. **Índice de Gini (Ig)**

   ```bash
   Ig = 1 - Σ p(i)²
   ```

   - Mide la probabilidad de clasificación errónea
   - Mayor Gini = mayor impureza

3. **Error de Clasificación (Ie)**

   ```bash
   Ie = 1 - max(p(i))
   ```

   - Mide el error usando la clase mayoritaria
   - Menos sensible a cambios en distribución

#### Ganancia de Información

```bash
IG = I(padre) - [N_izq/N_total × I(izq) + N_der/N_total × I(der)]
```

- **IG > 0**: La división reduce la impureza (buena división)
- **IG mayor**: Mejor división

### Problema Planteado

**Nodo Padre**: 30 ejemplos clase A, 100 ejemplos clase B

**Escenario A:**

- Hijo izquierdo: (10, 50)
- Hijo derecho: (20, 50)

**Escenario B:**

- Hijo izquierdo: (15, 40)
- Hijo derecho: (15, 60)

### Implementación

```python
def information_gain(parent, left, right, impurity_fn):
    n_p, n_l, n_r = sum(parent), sum(left), sum(right)
    ig = impurity_fn(parent) - (n_l / n_p) * impurity_fn(left) - (n_r / n_p) * impurity_fn(right)
    return ig
```

### Resultados Esperados

El ejercicio compara ambos escenarios usando las tres medidas de impureza y determina cuál división es óptima según cada criterio.

### Archivo

[ejercicio-diapositivas/ejercicio2.ipynb](ejercicio-diapositivas/ejercicio2.ipynb)

---

## Ejercicio 2: Modelo SVM Explicable

### Descripción ejercicio 2

Implementación de un clasificador **Support Vector Machine (SVM)** para clasificar imágenes del dataset Fruits-360, con técnicas de explicabilidad para interpretar las predicciones.

### Dataset

- **Nombre**: Fruits-360
- **Imágenes**: 100x100 píxeles, RGB
- **Clases**: 12 tipos de frutas
  - Apple Red 1, Banana 1, Limes 1, Strawberry 1
  - Cherimoya 1, Cucumber 1, Ginger 2, Mango 1
  - Nut 1, Peach 1, Papaya 1, Orange 1
- **División**: 80% entrenamiento, 20% validación

### Arquitectura SVM

```python
# Aplanar imágenes: (100, 100, 3) → (30000,)
X_train_flat = X_train.reshape(len(X_train), -1)

# Modelo SVM con kernel RBF
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
svm_model.fit(X_train_flat, y_train)
```

### Características Principales

1. **Preprocesamiento**:
   - Normalización de píxeles (0-1)
   - Aplanamiento de imágenes para SVM

2. **Modelo**:
   - Kernel RBF (Radial Basis Function)
   - Probabilidades habilitadas para análisis

3. **Evaluación**:
   - Accuracy
   - Classification Report (Precision, Recall, F1-Score)
   - Matriz de Confusión

4. **Explicabilidad**:
   - Visualización de vectores de soporte
   - Análisis de píxeles más influyentes
   - Mapas de calor de importancia

### Ventajas y Limitaciones

**Ventajas:**

- Efectivo en espacios de alta dimensión
- Robusto con datos bien separables
- Menor riesgo de overfitting con regularización

**Limitaciones:**

- No aprende características automáticamente
- Computacionalmente costoso con grandes datasets
- Difícil de interpretar en espacios de alta dimensión

### Archivo ejercicio 2

[ejericio-modelos/modelo-ml-svm-explicable.ipynb](ejericio-modelos/modelo-ml-svm-explicable.ipynb)

---

## Ejercicio 3: Modelo CNN Explicable con Grad-CAM

### Descripción ejericio 3

Implementación de una **Red Neuronal Convolucional (CNN)** para clasificación de imágenes con explicabilidad mediante **Grad-CAM** (Gradient-weighted Class Activation Mapping).

### Arquitectura CNN

```python
cnn_model = Sequential([
    # Capa Convolucional 1: Aprende bordes y colores simples
    Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)),
    MaxPooling2D(2, 2),
    
    # Capa Convolucional 2: Aprende formas más complejas
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    
    # Capa Convolucional 3: Características de alto nivel
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    
    # Capas Densas
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),  # Previene overfitting
    
    # Capa de Salida
    Dense(12, activation='softmax')  # 12 clases
])
```

### Hiperparámetros

- **Optimizador**: Adam
- **Función de pérdida**: Categorical Crossentropy
- **Épocas**: 10
- **Batch size**: 32
- **Dropout**: 0.5

### Grad-CAM: Explicabilidad Visual

Grad-CAM visualiza qué regiones de la imagen son importantes para la predicción del modelo.

#### ¿Cómo funciona Grad-CAM?

1. **Extrae activaciones** de la última capa convolucional
2. **Calcula gradientes** respecto a la clase predicha
3. **Pondera canales** según importancia
4. **Genera mapa de calor** mostrando regiones importantes

#### Implementación ejercicio 3

```python
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # Construir modelo de gradiente
    grad_model = tf.keras.models.Model(
        [model.inputs], 
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    
    # Calcular gradientes
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        class_channel = preds[:, pred_index]
    
    grads = tape.gradient(class_channel, last_conv_layer_output)
    
    # Global Average Pooling de gradientes
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Ponderar canales
    heatmap = last_conv_layer_output[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # Normalizar
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    
    return heatmap.numpy()
```

### Visualización de Explicabilidad

Para cada predicción, se genera:

1. **Imagen Original**: Input del modelo
2. **Heatmap Grad-CAM**: Regiones importantes (calientes = más importantes)
3. **Superposición**: Imagen + heatmap para interpretación visual

### Evaluación

1. **Curvas de Aprendizaje**:
   - Accuracy vs Épocas
   - Loss vs Épocas
   - Comparación Training/Validation

2. **Matriz de Confusión**:
   - Visualización de predicciones correctas e incorrectas
   - Identificación de clases confundidas

3. **Casos de Prueba**:
   - Predicción + explicación visual para casos específicos

### Exportación del Modelo

El notebook incluye una función para exportar el modelo entrenado:

```python
def exportar_modelo(model, nombre_modelo='modelo_cnn_frutas', formato='h5'):
    """
    Exporta el modelo en múltiples formatos:
    - H5: Modelo completo
    - SavedModel: Formato TensorFlow
    - JSON: Arquitectura
    - Pesos: Archivo separado
    - Classes: Lista de clases
    """
    # ... implementación
```

### Ventajas CNN sobre SVM

1. **Aprendizaje de características**: Automático y jerárquico
2. **Mejor rendimiento**: En tareas de visión por computador
3. **Explicabilidad visual**: Grad-CAM muestra qué "ve" el modelo
4. **Escalabilidad**: Mejor con grandes datasets

### Archivo ejercicio 3

[ejericio-modelos/modelo-ml-cnn-explicable.ipynb](ejericio-modelos/modelo-ml-cnn-explicable.ipynb)

---

## Referencias

### Papers y Artículos

1. **Grad-CAM**: Selvaraju et al. (2017). "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"
2. **SVM**: Cortes & Vapnik (1995). "Support-vector networks"
3. **CNN**: LeCun et al. (1998). "Gradient-based learning applied to document recognition"

### Libros

- Raschka, S., & Mirjalili, V. (2019). *Python Machine Learning* (3rd ed.). Packt Publishing
- Géron, A. (2019). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* (2nd ed.). O'Reilly Media

### Recursos Online

- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Keras Documentation](https://keras.io/)
- [Fruits-360 Dataset](https://www.kaggle.com/moltean/fruits)

### Documentación del Proyecto

- [Árboles de Decisión](docs/arboles_decision.md)
- [Maximizar Ganancia de Información](docs/maximizar_ganancia_informacion.md)
- [Guía Práctica Ejercicio 1](docs/GuiadePractica%20y%20Recursos%20ExplicacionPredicciones/README-1.md)
- [Guía Práctica Ejercicios 2 y 3](docs/GuiadePractica%20y%20Recursos%20ExplicacionPredicciones/README-2.md)

---

## Autores

**Felipe Peralta** & **Samantha Suquilanda**

Universidad Politecnica Salesiana - Aprendizaje Automático - 7mo Semestre
