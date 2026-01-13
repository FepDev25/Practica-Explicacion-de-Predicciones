# Práctica 3: Árboles de Decisión y Técnicas de Explicabilidad en Machine Learning

**Integrantes:** Felipe Peralta, Samantha Suquilanda

---

## Descripción General

Esta práctica explora cuatro aspectos fundamentales del aprendizaje automático y la explicabilidad de modelos:

1. **Fundamentos teóricos**: Cálculo de ganancia de información en árboles de decisión
2. **Visión por Computador**: Clasificación de imágenes con SVM y CNN (con Grad-CAM)
3. **Clasificación Tabular - ML Clásico**: Predicción de intención de compra con Random Forest (con LIME)
4. **Clasificación Tabular - Deep Learning**: Predicción con Redes Neuronales (con Gradientes Integrados)

### Objetivos de Aprendizaje

- Comprender las medidas de impureza (Entropía, Gini, Error de Clasificación)
- Calcular la Ganancia de Información para seleccionar las mejores divisiones
- Implementar y comparar modelos para clasificación de imágenes (SVM vs CNN)
- Aplicar técnicas de Explicabilidad de IA (XAI): **Grad-CAM**, **LIME**, **Integrated Gradients**
- Trabajar con datos tabulares desbalanceados y técnicas de balanceo
- Comparar enfoques de Machine Learning clásico vs Deep Learning
- Implementar tracking de experimentos con **MLflow** para reproducibilidad y gestión de modelos

---

## Estructura del Proyecto

```bash
practica-3/
│
├── README.md                                      # Este archivo
│
├── data/                                          # Datasets
│   ├── fruits-360_100x100/                       # Dataset Fruits-360 (imágenes)
│   │   └── fruits-360/
│   │       ├── Training/                         # Imágenes de entrenamiento
│   │       └── Test/                             # Imágenes de prueba
│   └── online+shoppers+purchasing+intention+dataset/  # Dataset Online Shoppers (tabular)
│       └── online_shoppers_intention.csv
│
├── docs/                                          # Material de apoyo (extra)
│   ├── arboles_decision.md
│   ├── maximizar_ganancia_informacion.md
│   └── GuiadePractica y Recursos ExplicacionPredicciones/
│       ├── README-1.md
│       ├── README-2.md
│       └── PreprocesamientoRandomForestConPipeline.ipynb
│
├── ejercicio-diapositivas/
│   └── ejercicio2.ipynb                          # Ejercicio 1: Ganancia de Información
│
├── ejericio-modelos/                             # Ejercicios principales (2, 3 y 4)
│   ├── modelo-ml-svm-explicable.ipynb            # Ejercicio 2A: SVM para imágenes
│   ├── modelo-ml-cnn-explicable.ipynb            # Ejercicio 2B: CNN para imágenes + Grad-CAM
│   ├── online-shoppers-explicable-rf.ipynb       # Ejercicio 3: Random Forest + LIME
│   ├── online-shoppers-explicable-gi.ipynb       # Ejercicio 4: Red Neuronal + Integrated Gradients
│   └── exportaciones/                            # resultados de experimentos con MLflow
│       ├── cnn-frutas-explicable/                # Exportaciones del modelo CNN
│       │   ├── mlruns/                           # Tracking MLflow
│       │   ├── models/                           # Modelos guardados
│       │   └── plots/                            # Gráficas generadas
│       ├── svm-frutas-explicable/                # Exportaciones del modelo SVM
│       │   ├── mlruns/
│       │   ├── models/
│       │   └── plots/
│       ├── nn-shoppers-gi/                       # Exportaciones Red Neuronal + IG
│       │   ├── mlruns/
│       │   ├── models/
│       │   ├── plots/
│       │   └── explanations/
│       └── rf-shoppers-lime/                     # Exportaciones Random Forest + LIME
│           ├── mlruns/
│           ├── models/
│           ├── plots/
│           ├── explanations/
│           └── trees/
└──
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

### Conclusión

El ejercicio compara ambos escenarios usando las tres medidas de impureza y determina que **el mejor split NO es el que divide más "parejo", sino el que produce nodos más puros**.

### Archivo

[ejercicio-diapositivas/ejercicio2.ipynb](ejercicio-diapositivas/ejercicio2.ipynb)

---

## Ejercicio 2: Clasificación de Imágenes - SVM y CNN

### Descripción ejercicio 2

Implementación y comparación de dos enfoques para clasificación de imágenes del dataset **Fruits-360**:

- **SVM (Support Vector Machine)**: Enfoque clásico de ML
- **CNN (Convolutional Neural Network)**: Enfoque de Deep Learning

Incluye técnicas de explicabilidad para interpretar las predicciones, especialmente **Grad-CAM** para CNN.

### Dataset Fruits-360

- **Imágenes**: 100x100 píxeles, RGB
- **Clases seleccionadas**: 12 tipos de frutas
  - Apple Red 1, Banana 1, Limes 1, Strawberry 1
  - Cherimoya 1, Cucumber 1, Ginger 2, Mango 1
  - Nut 1, Peach 1, Papaya 1, Orange 1
- **División**: 80% entrenamiento, 20% validación

### 2A. Modelo SVM

#### Arquitectura

```python
# Aplanar imágenes: (100, 100, 3) → (30000,)
X_train_flat = X_train.reshape(len(X_train), -1)

# Modelo SVM con kernel RBF
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
svm_model.fit(X_train_flat, y_train)
```

#### Características

- **Preprocesamiento**: Normalización (0-1) y aplanamiento
- **Kernel**: RBF (Radial Basis Function)
- **Evaluación**: Accuracy, Classification Report, Matriz de Confusión
- **Explicabilidad**: Visualización de vectores de soporte, análisis de píxeles influyentes

#### Ventajas y Limitaciones

**Ventajas:**

- Efectivo en espacios de alta dimensión
- Robusto con datos bien separables
- Menor riesgo de overfitting con regularización adecuada

**Limitaciones:**

- No aprende características automáticamente
- Computacionalmente costoso con grandes datasets
- Difícil de interpretar en espacios de alta dimensión

#### Tracking con MLflow

Todos los experimentos se registran automáticamente con MLflow:

- **Parámetros**: kernel, C, gamma, train/test split
- **Métricas**: accuracy, precision, recall, F1-score
- **Artefactos**: modelo guardado (pickle), gráficas de confusión, explicaciones
- **Ubicación**: `exportaciones/svm-frutas-explicable/mlruns/`

### 2B. Modelo CNN con Grad-CAM

#### Arquitectura CNN

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

#### Hiperparámetros

- **Optimizador**: Adam
- **Función de pérdida**: Categorical Crossentropy
- **Épocas**: 10
- **Batch size**: 32
- **Dropout**: 0.5

#### Grad-CAM: Explicabilidad Visual

**Grad-CAM** (Gradient-weighted Class Activation Mapping) visualiza qué regiones de la imagen son importantes para la predicción.

**¿Cómo funciona?**

1. Extrae activaciones de la última capa convolucional
2. Calcula gradientes respecto a la clase predicha
3. Pondera canales según importancia
4. Genera mapa de calor mostrando regiones importantes

**Visualización:**

- Regiones calientes (rojas) = Más importantes para la predicción
- Regiones frías (azules) = Menos relevantes

#### Ventajas CNN sobre SVM

1. **Aprendizaje de características**: Automático y jerárquico
2. **Mejor rendimiento**: En tareas de visión por computador
3. **Explicabilidad visual**: Grad-CAM muestra qué "ve" el modelo
4. **Escalabilidad**: Mejor con grandes datasets

#### Tracking con MLflow

Integración completa con MLflow para gestión de experimentos:

- **Parámetros**: arquitectura (capas, filtros), épocas, batch_size, learning_rate, dropout
- **Métricas**: train/val accuracy, train/val loss, test accuracy, precision, recall, F1
- **Artefactos**: 
  - Modelo completo (.h5)
  - Arquitectura (JSON)
  - Pesos (.h5)
  - Clases (JSON)
  - Curvas de entrenamiento
  - Matrices de confusión
  - Mapas Grad-CAM
- **Ubicación**: `exportaciones/cnn-frutas-explicable/mlruns/`

### Archivos

- **SVM**: [ejericio-modelos/modelo-ml-svm-explicable.ipynb](ejericio-modelos/modelo-ml-svm-explicable.ipynb)
- **CNN + Grad-CAM**: [ejericio-modelos/modelo-ml-cnn-explicable.ipynb](ejericio-modelos/modelo-ml-cnn-explicable.ipynb)

---

## Ejercicio 3: Predicción de Compras Online - Random Forest

### Descripción ejercicio 3

Predicción de **intención de compra** en una tienda online usando **Random Forest** con explicabilidad mediante **LIME** (Local Interpretable Model-agnostic Explanations).

### Dataset Online Shoppers Purchasing Intention

- **Fuente**: UCI Machine Learning Repository
- **Tipo**: Datos tabulares de sesiones web
- **Objetivo**: Predecir si un usuario completará una compra (`Revenue`: True/False)
- **Características**:
  - Numéricas: Duración de visitas, tasa de rebote, tasa de salida, valor de páginas
  - Categóricas: Mes, tipo de visitante, fin de semana, región, etc.
- **Desbalanceo**: ~85% No Compra, ~15% Compra

### Flujo del Análisis

1. **Exploración de Datos (EDA)**:
   - Análisis de distribución de la variable objetivo
   - Correlaciones entre variables numéricas
   - Análisis temporal (ventas por mes)
   - Análisis categórico (tipo de visitante)

2. **Preprocesamiento**:
   - Eliminación de duplicados
   - One-Hot Encoding para variables categóricas
   - Escalado con StandardScaler
   - Manejo de desbalanceo con `class_weight='balanced'`

3. **Modelo Random Forest**:

   ```python
   rf_model = RandomForestClassifier(
       n_estimators=100, 
       class_weight='balanced', 
       random_state=42,
       n_jobs=-1
   )
   ```

4. **Explicabilidad Global**: Importancia de características nativa de Random Forest
   - Muestra qué variables son más importantes globalmente
   - Basado en Mean Decrease in Impurity

5. **Explicabilidad Local con LIME**:
   - Explica predicciones individuales
   - Muestra qué características influyeron en cada predicción
   - Visualización de contribuciones positivas/negativas

### LIME: Local Interpretable Model-agnostic Explanations

**¿Qué es LIME?**

LIME explica predicciones individuales de cualquier modelo de ML entrenando un modelo interpretable localmente alrededor de la instancia que se quiere explicar.

**Ventajas:**

- Model-agnostic (funciona con cualquier modelo)
- Explicaciones locales específicas por instancia
- Intuitivo y fácil de visualizar
- Identifica características más influyentes

**Implementación:**

```python
explainer_lime = lime.lime_tabular.LimeTabularExplainer(
    training_data=np.array(X_train),
    feature_names=X.columns,
    class_names=['No Compra', 'Compra'],
    mode='classification'
)

exp = explainer_lime.explain_instance(
    data_row=cliente_data, 
    predict_fn=model.predict_proba, 
    num_features=10
)
```

### Visualizaciones Incluidas

1. **Matriz de Confusión**: Rendimiento del modelo
2. **Importancia Global**: Ranking de características
3. **Explicaciones LIME**: Para clientes individuales
4. **Árbol de Decisión Individual**: Visualización de un árbol del bosque

### Tracking con MLflow

Sistema completo de tracking para Random Forest:

- **Parámetros**: n_estimators, class_weight, max_features, bootstrap, xai_method, lime_num_features
- **Métricas**: accuracy, precision, recall, F1-score, AUC
- **Artefactos**:
  - Modelo (pickle)
  - Scaler (pickle)
  - Feature names (JSON)
  - Feature importances (CSV)
  - Matriz de confusión
  - Gráfica de importancia global
  - Explicaciones LIME individuales (múltiples clientes)
  - Visualizaciones de árboles individuales
- **Ubicación**: `exportaciones/rf-shoppers-lime/mlruns/`

**Comando para visualizar experimentos:**
```bash
cd ejericio-modelos
mlflow ui --backend-store-uri file:///$(pwd)/exportaciones/rf-shoppers-lime/mlruns
```

### Archivo ejercicio 3

[ejericio-modelos/online-shoppers-explicable-rf.ipynb](ejericio-modelos/online-shoppers-explicable-rf.ipynb)

---

## Ejercicio 4: Predicción de Compras Online - Red Neuronal

### Descripción ejercicio 4

Predicción de **intención de compra** usando una **Red Neuronal Densa** (Deep Learning) con explicabilidad mediante **Gradientes Integrados** (Integrated Gradients).

### Arquitectura de Red Neuronal

```python
model = Sequential([
    # Capa de entrada
    Dense(64, activation='relu', input_shape=(X_train_tensor.shape[1],)),
    Dropout(0.2),  # Regularización
    
    Dense(32, activation='relu'),
    Dropout(0.2),
    
    Dense(16, activation='relu'),
    
    # Capa de salida: Clasificación binaria
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
```

### Características del Entrenamiento

- **Épocas**: 20
- **Batch size**: 32
- **Validation split**: 20%
- **Class weighting**: Balanceo automático para manejar desbalanceo
- **Umbral de decisión**: 0.7 (ajustado para priorizar precision en clase positiva)

### Gradientes Integrados: Explicabilidad para Redes Neuronales

**¿Qué son los Gradientes Integrados?**

Técnica de atribución de características que calcula la importancia de cada entrada integrando gradientes a lo largo de una trayectoria desde una línea base (baseline) hasta la entrada real.

**¿Cómo funciona?**

1. **Baseline**: Define un punto de referencia (ej: todos ceros o media del dataset)
2. **Trayectoria**: Crea interpolaciones lineales desde baseline hasta la entrada
3. **Gradientes**: Calcula gradientes en cada punto de la trayectoria
4. **Integración**: Promedia los gradientes multiplicados por la diferencia con baseline

**Ventajas:**

- Específico para redes neuronales
- Satisface axiomas de sensibilidad e implementación invarianza
- Atribuciones suaves y estables
- No requiere entrenar modelos adicionales

**Implementación:**

```python
@tf.function
def integrated_gradients(inputs, model, baseline=None, num_steps=50):
    if baseline is None:
        baseline = tf.zeros_like(inputs)
    
    # Interpolar entre baseline e input
    alphas = tf.linspace(0.0, 1.0, num_steps + 1)
    interpolated_inputs = baseline + alphas[:, tf.newaxis, tf.newaxis] * (inputs - baseline)
    
    # Calcular gradientes
    with tf.GradientTape() as tape:
        tape.watch(interpolated_inputs)
        predictions = model(interpolated_inputs)
    
    gradients = tape.gradient(predictions, interpolated_inputs)
    
    # Integrar
    avg_gradients = tf.reduce_mean(gradients, axis=0)
    integrated_grads = (inputs - baseline) * avg_gradients
    
    return integrated_grads
```

### Visualizaciones

1. **Curvas de Entrenamiento**: Accuracy y Loss durante entrenamiento
2. **Matriz de Confusión**: Evaluación del modelo
3. **Atribuciones por Característica**: Visualización de Integrated Gradients para casos individuales
4. **Comparación Real vs Predicho**: Análisis de aciertos y errores

### Tracking con MLflow

Tracking detallado para Deep Learning con TensorFlow:

- **Parámetros**: arquitectura (capas, neuronas), épocas, batch_size, dropout, learning_rate, threshold, xai_method, ig_steps
- **Métricas**: 
  - Por época: train/val accuracy, train/val loss
  - Finales: test accuracy, precision, recall, F1-score, AUC
- **Artefactos**:
  - Modelo completo (.h5)
  - Scaler (pickle)
  - Feature names (JSON)
  - Curvas de entrenamiento (accuracy/loss)
  - Matriz de confusión
  - Explicaciones con Integrated Gradients (múltiples clientes)
  - Visualizaciones de atribuciones
- **Ubicación**: `exportaciones/nn-shoppers-gi/mlruns/`

**Comando para visualizar experimentos:**
```bash
cd ejericio-modelos
mlflow ui --backend-store-uri file:///$(pwd)/exportaciones/nn-shoppers-gi/mlruns
```

### Comparación: Random Forest vs Red Neuronal

| Aspecto | Random Forest (Ej. 3) | Red Neuronal (Ej. 4) |
| --------- | --------------------- | --------------------- |
| **Tipo** | ML Clásico (Ensemble) | Deep Learning |
| **Interpretabilidad Nativa** | Alta (árboles) | Baja (caja negra) |
| **Técnica XAI** | LIME | Integrated Gradients |
| **Explicaciones** | Locales, model-agnostic | Locales, específicas NN |
| **Entrenamiento** | Más rápido | Más lento (épocas) |
| **Hiperparámetros** | Menos críticos | Más sensibles |
| **Manejo de no-linealidad** | Excelente | Excelente |
| **MLflow Tracking** | Completo | Completo con callbacks |

### Archivo ejercicio 4

[ejericio-modelos/online-shoppers-explicable-gi.ipynb](ejericio-modelos/online-shoppers-explicable-gi.ipynb)

---

### Material Extra (Opcional)

El directorio `docs/` contiene material de apoyo adicional:

- **Teoría**: `arboles_decision.md`, `maximizar_ganancia_informacion.md`
- **Guías**: `docs/GuiadePractica y Recursos ExplicacionPredicciones/`
  - README-1.md: Guía detallada ejercicio ganancia información
  - README-2.md: Guía detallada modelos SVM y CNN
  - PreprocesamientoRandomForestConPipeline.ipynb: Ejemplo adicional

---

## Comparación de Técnicas de Explicabilidad

| Técnica | Tipo Modelo | Nivel | Ventajas | Desventajas | Usado en |
| --------- | ------------- | ------- | ---------- | ------------- | ---------- |
| **Grad-CAM** | CNN | Local (imágenes) | Visual, intuitivo, específico para visión | Solo para CNNs | Ejercicio 2B |
| **LIME** | Agnóstico | Local (tabular) | Funciona con cualquier modelo, interpretable | Inestable, requiere muestreo | Ejercicio 3 |
| **Integrated Gradients** | Redes Neuronales | Local (tabular) | Matemáticamente fundamentado, estable | Solo para modelos diferenciables | Ejercicio 4 |
| **Feature Importance** | Random Forest | Global | Nativo del modelo, rápido | No explica casos individuales | Ejercicio 3 |

### Cuándo usar cada técnica

- **Grad-CAM**: Clasificación de imágenes con CNNs
- **LIME**: Cuando necesitas explicar cualquier modelo de caja negra
- **Integrated Gradients**: Redes neuronales que requieren explicaciones precisas
- **Feature Importance**: Análisis global rápido en modelos basados en árboles

---

## Gestión de Experimentos con MLflow

### ¿Por qué MLflow?

MLflow es una plataforma open-source para gestionar el ciclo de vida completo de machine learning. En esta práctica se utiliza para:

1. **Reproducibilidad**: Registrar todos los parámetros e hiperparámetros
2. **Comparación**: Comparar diferentes experimentos fácilmente
3. **Tracking**: Seguimiento de métricas durante el entrenamiento
4. **Versionado**: Gestionar diferentes versiones de modelos
5. **Artefactos**: Almacenar modelos, gráficas y explicaciones organizadamente

### Estructura de Exportaciones

Cada modelo tiene su propia carpeta en `exportaciones/` con la siguiente estructura:

```
exportaciones/{modelo-nombre}/
├── mlruns/              # Base de datos MLflow
│   ├── 0/              # Experimento por defecto
│   └── {experiment_id}/
│       └── {run_id}/
│           ├── meta.yaml
│           ├── metrics/
│           ├── params/
│           ├── tags/
│           └── artifacts/
├── models/              # Modelos guardados
├── plots/               # Gráficas generadas
├── explanations/        # Explicaciones XAI (opcional)
└── trees/               # Visualizaciones árboles (solo RF)
```

### Modelos con MLflow Integrado

| Modelo | Experimento | Framework | Ubicación |
|--------|-------------|-----------|------------|
| **CNN** | CNN-Frutas-Explicable | mlflow.keras | `exportaciones/cnn-frutas-explicable/` |
| **SVM** | SVM-Frutas-Explicable | mlflow.sklearn | `exportaciones/svm-frutas-explicable/` |
| **Random Forest** | RF-Shoppers-LIME | mlflow.sklearn | `exportaciones/rf-shoppers-lime/` |
| **Red Neuronal** | NN-Shoppers-IG | mlflow.tensorflow | `exportaciones/nn-shoppers-gi/` |

### Comandos Útiles

**Visualizar experimentos en UI:**
```bash
# Para un modelo específico
cd ejericio-modelos
mlflow ui --backend-store-uri file:///$(pwd)/exportaciones/cnn-frutas-explicable/mlruns

# Acceder en navegador
http://localhost:5000
```

**Cargar modelo guardado:**
```python
import mlflow

# Cargar modelo de MLflow
model_uri = "file:///path/to/mlruns/0/{run_id}/artifacts/model"
loaded_model = mlflow.keras.load_model(model_uri)  # Para CNN/NN
# o
loaded_model = mlflow.sklearn.load_model(model_uri)  # Para SVM/RF
```

**Comparar experimentos:**
```python
import mlflow

mlflow.set_tracking_uri("file:///path/to/mlruns")
experiment = mlflow.get_experiment_by_name("CNN-Frutas-Explicable")
runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

# Ver métricas de todos los runs
print(runs[['metrics.test_accuracy', 'metrics.test_f1', 'params.epochs']])
```

### Beneficios Implementados

1. **Organización**: Cada modelo tiene su carpeta independiente
2. **Trazabilidad**: Todos los artefactos están versionados
3. **Comparación**: Fácil comparación entre diferentes configuraciones
4. **Portabilidad**: Los modelos pueden cargarse en cualquier máquina
5. **Documentación**: Los parámetros y métricas se registran automáticamente

---

## Resumen de Datasets

### 1. Fruits-360 (Ejercicio 2)

- **Tipo**: Imágenes
- **Tamaño**: 100x100 píxeles, RGB
- **Clases**: 12 tipos de frutas
- **Aplicación**: Visión por Computador

### 2. Online Shoppers Purchasing Intention (Ejercicios 3 y 4)

- **Tipo**: Tabular
- **Instancias**: ~12,000 sesiones
- **Features**: ~18 características (numéricas y categóricas)
- **Objetivo**: Predicción binaria (compra sí/no)
- **Desafío**: Dataset desbalanceado

---

## Referencias

### Papers Fundamentales

1. **Grad-CAM**: Selvaraju et al. (2017). "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"
2. **LIME**: Ribeiro et al. (2016). "Why Should I Trust You?: Explaining the Predictions of Any Classifier"
3. **Integrated Gradients**: Sundararajan et al. (2017). "Axiomatic Attribution for Deep Networks"
4. **Random Forest**: Breiman (2001). "Random Forests"
5. **SVM**: Cortes & Vapnik (1995). "Support-vector networks"
6. **CNN**: LeCun et al. (1998). "Gradient-based learning applied to document recognition"

### Libros Recomendados

- Raschka, S., & Mirjalili, V. (2019). *Python Machine Learning* (3rd ed.). Packt Publishing
- Géron, A. (2019). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* (2nd ed.). O'Reilly Media
- Molnar, C. (2022). *Interpretable Machine Learning*. <https://christophm.github.io/interpretable-ml-book/>

### Recursos Online

- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Keras Documentation](https://keras.io/)
- [MLflow Documentation](https://mlflow.org/)
- [LIME Documentation](https://lime-ml.readthedocs.io/)
- [Fruits-360 Dataset](https://www.kaggle.com/moltean/fruits)
- [Online Shoppers Dataset](https://archive.ics.uci.edu/ml/datasets/Online+Shoppers+Purchasing+Intention+Dataset)

### Herramientas de XAI y MLOps

- **LIME**: <https://github.com/marcotcr/lime>
- **SHAP**: <https://github.com/slundberg/shap>
- **Grad-CAM**: Implementado en TensorFlow/Keras
- **Integrated Gradients**: TensorFlow official
- **MLflow**: <https://mlflow.org/> - Gestión de experimentos y modelos

---

## Autores

**Felipe Peralta** & **Samantha Suquilanda**

Universidad Politécnica Salesiana - Aprendizaje Automático - 7mo Semestre

---

## Nota Final

Este proyecto demuestra la importancia de la **explicabilidad en IA (XAI)**. No basta con que un modelo tenga buena precisión; es fundamental entender **por qué** toma sus decisiones, especialmente en aplicaciones críticas como:

- 🏥 Medicina (diagnósticos)
- 💰 Finanzas (aprobación de créditos)
- ⚖️ Justicia (evaluación de riesgos)
- 🛒 E-commerce (recomendaciones)

Las técnicas de explicabilidad cubiertas en esta práctica son herramientas esenciales para construir **IA confiable y transparente**.

---

**Fecha**: Enero 2026
