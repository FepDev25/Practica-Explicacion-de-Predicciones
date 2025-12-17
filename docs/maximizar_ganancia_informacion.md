# Maximizar la Ganancia de Información (IG)

- El tema de Maximizar la Ganancia de Información (IG) es el objetivo central utilizado en los algoritmos de aprendizaje de árboles de decisión (Decision Tree Learning) para determinar la mejor manera de dividir los datos en cada nodo.

## Objetivo y Definición de la Ganancia de Información

- El objetivo fundamental es maximizar la IG en cada división para seleccionar las características más informativas.

1. Proceso Iterativo: El algoritmo comienza en la raíz del árbol y divide los datos en la característica que resulta en la mayor Ganancia de Información (IG). Este proceso se repite en cada nodo hijo hasta que las hojas son "puras", lo que significa que todos los ejemplos de entrenamiento en ese nodo pertenecen a la misma clase.
2. Fórmula de la IG: La Ganancia de Información es simplemente la diferencia entre la impureza del nodo padre y la suma de las impurezas de los nodos hijos. Cuanto menor sea la impureza de los nodos hijos, mayor será la ganancia de información.
    - La fórmula general para la IG, donde $f$ es la característica para realizar la división, $D_p$ es el conjunto de datos del nodo padre, y $D_j$ es el conjunto de datos del $j$-ésimo nodo hijo, es:
        $$IG(D_p, f) = I(D_p) − \sum_{j=1}^{m} \frac{N_j}{N_p} I(D_j)$$
    - Para los árboles de decisión binarios (que son los que implementan la mayoría de las librerías, incluyendo scikit-learn), la división se realiza en dos nodos hijos ($D_{left}$ y $D_{right}$).

## Medidas de Impureza (Splitting Criteria)

- La $I$ en la fórmula anterior representa la medida de impureza del nodo. Existen tres criterios de división comunes utilizados en los árboles de decisión binarios: la impureza Gini ($I_G$), la entropía ($I_H$) y el error de clasificación ($I_E$).

### 1. Entropía ($I_H$)

- La entropía intenta maximizar la información mutua en el árbol.
- La entropía es cero (0) si todos los ejemplos en un nodo pertenecen a la misma clase.
- La entropía es máxima si la distribución de clases es uniforme. Por ejemplo, en un entorno de clase binaria, la entropía es 1 si las clases se distribuyen uniformemente (ej., $p=0.5$).

### 2. Impureza Gini ($I_G$)

- La impureza Gini se enfoca en minimizar la probabilidad de una clasificación errónea.
- Similar a la entropía, la impureza Gini es máxima si las clases están perfectamente mezcladas.
- En la práctica, la impureza Gini y la entropía suelen producir resultados muy similares, por lo que a menudo no vale la pena dedicar mucho tiempo a compararlas en lugar de experimentar con diferentes límites de poda (*pruning*).

### 3. Error de Clasificación ($I_E$)

- Aunque es una medida de impureza, el error de clasificación no se recomienda para hacer crecer un árbol de decisión porque es menos sensible a los cambios en las probabilidades de clase de los nodos. Sin embargo, puede ser un criterio útil para la poda. Los criterios de Gini y Entropía tienden a favorecer una división que es genuinamente más pura, mientras que el error de clasificación podría dar el mismo resultado de IG para divisiones con diferentes niveles de pureza.
