# Arboles de Decisión

- Los árboles de decisión son modelos de clasificación muy atractivos, especialmente cuando se busca interpretabilidad en el modelo.
- Un árbol de decisión funciona descomponiendo los datos mediante una serie de preguntas binarias o decisiones.
- Basándose en las características del conjunto de entrenamiento, el modelo aprende estas preguntas para inferir las etiquetas de clase. Este concepto se aplica incluso si las características son números reales (como en el conjunto de datos Iris), donde se puede definir un valor de corte (cut-off) y preguntar, por ejemplo, si el ancho del pétalo es mayor o igual a cierto valor.

## El Proceso de Aprendizaje (Splitting)

1. Inicio y División: El algoritmo comienza en la raíz del árbol. El proceso clave es dividir los datos de forma iterativa en cada nodo.
2. Pureza de las Hojas: Este procedimiento de división se repite hasta que las hojas son "puras", lo que significa que todos los ejemplos de entrenamiento en ese nodo pertenecen a la misma clase.
3. Maximización de la Ganancia de Información (IG): Para decidir qué característica usar para la división en cada nodo, el algoritmo se enfoca en maximizar la Ganancia de Información (IG). La característica que produce la IG más grande es la elegida.

- La Ganancia de Información ($IG$) es la diferencia entre la impureza del nodo padre ($D_p$) y la suma de las impurezas de los nodos hijos ($D_j$). Una menor impureza en los nodos hijos resulta en una Ganancia de Información mayor. La fórmula general es:

$$IG(D_p, f) = I(D_p) − \sum_{j=1}^{m} \frac{N_j}{N_p} I(D_j)$$

## Criterios de Impureza (Medidas I)

La medida $I$ en la fórmula anterior se refiere a la impureza del nodo. Los tres criterios de división o impureza más comunes en los árboles de decisión binarios (que implementan la mayoría de las librerías, incluyendo scikit-learn) son:

1. Entropía ($I_H$):
    - El criterio de entropía intenta maximizar la información mutua en el árbol.
    - La entropía es cero (0) si todos los ejemplos en un nodo pertenecen a la misma clase.
    - La entropía es máxima si la distribución de clases es uniforme (por ejemplo, 1 en un entorno de clase binaria donde la probabilidad de ambas clases es 0.5).

2. Impureza Gini ($I_G$):
    - La impureza Gini tiene como objetivo minimizar la probabilidad de una clasificación errónea.
    - Al igual que la entropía, es máxima si las clases están perfectamente mezcladas.
    - En la práctica, la impureza Gini y la entropía suelen producir resultados muy similares, por lo que a menudo no se justifica gastar mucho tiempo en compararlas.

3. Error de Clasificación ($I_E$):
    - Esta medida se calcula como $1 - \max\{p(i|t)\}$, donde $p(i|t)$ es la proporción de ejemplos de la clase $i$ en el nodo $t$.
    - No se recomienda para hacer crecer el árbol de decisión porque es menos sensible a los cambios en las probabilidades de clase de los nodos, aunque puede ser un criterio útil para la poda.

## Desafíos y Extensiones

1. Overfitting y Poda: Los árboles de decisión pueden construir fronteras de decisión complejas al dividir el espacio de características en rectángulos. Sin embargo, si el árbol se vuelve muy profundo, puede llevar fácilmente al sobreajuste (overfitting), donde el modelo captura bien los patrones en los datos de entrenamiento, pero falla al generalizar a datos no vistos. Para mitigar esto, generalmente se poda el árbol estableciendo un límite para la profundidad máxima (`max_depth`),.
2. Random Forests (Bosques Aleatorios): Un *random forest* es un ensamble de árboles de decisión. Esta técnica promedia múltiples árboles de decisión (profundos) que individualmente sufren de alta varianza para construir un modelo más robusto, menos susceptible al sobreajuste y con mejor rendimiento de generalización. Generalmente, no es necesario podar el *random forest*.
3. Escalado de Características: Una ventaja de los árboles de decisión (a diferencia de otros algoritmos como el perceptrón o la regresión logística) es que el escalado de características (feature scaling) no es un requisito necesario para el algoritmo.
