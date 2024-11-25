
# Identificación de Perfiles Educativos en el ámbito de la Educación Superior mediante Clustering

Este proyecto implementa técnicas de aprendizaje no supervisado para identificar perfiles característicos de estudiantes universitarios en Ecuador. Utiliza algoritmos de clustering, como K-modes y clustering jerárquico, aplicados a un conjunto de datos con información demográfica y académica. El análisis permite identificar patrones y problemáticas clave en el sistema de educación superior, proporcionando una base sólida para la formulación de políticas públicas y estrategias educativas.



## Resumen del proyecto
El objetivo principal de este proyecto es analizar datos educativos utilizando algoritmos de clustering.
Identificar clusters que reflejen características comunes entre los estudiantes.
Generar perfiles que puedan ser útiles para intervenciones específicas en el ámbito educativo.

## Características principales

Preprocesamiento de datos:
Limpieza de datos, eliminación de valores inconsistentes.
Codificación de variables categóricas utilizando One-Hot Encoding.
Reducción de dimensionalidad basada en análisis de correlación (Cramér's V).
Aplicación de algoritmos de clustering:

K-modes: Ideal para datos categóricos.
Clustering jerárquico: Utilizando Hamming.
Evaluación de los clusters:

Coeficiente de Silhouette.
Índice de Calinski-Harabasz.
Índice de correlación cophenética (CCC) para clustering jerárquico.
Resultados:

Identificación de clusters que reflejan patrones demográficos y académicos.
Visualizaciones que resumen la distribución de estudiantes y la oferta académica.
## Requisitos

Para ejecutar el notebook, necesitas las siguientes herramientas y librerías:

Entorno
Python 3.7+
Jupyter Notebook o Google Colab.
Librerías necesarias
pandas
numpy
matplotlib
seaborn
scipy
sklearn
## Cómo usar el proyecto

Este es el link para clonar el proyecto
git clone https://github.com/LuisStatsOps/Clustering_educacion.git

Configura los datos:

Coloca el archivo de datos en formato .xlsx en la raíz del proyecto.
Asegúrate de que los datos tengan las columnas requeridas, como se especifica en el notebook.
Ejecuta el notebook:

Abre el archivo notebook - Colab.ipynb en Jupyter Notebook o Google Colab.
Sigue las celdas paso a paso para realizar el análisis.
Interpreta los resultados:

Revisa las visualizaciones generadas y los clusters formados.
Consulta el apéndice del artículo científico para más detalles sobre las características de los clusters.


## Autor

Luis Alberto Baca Guerrero
Universidad San Francisco de Quito
Máster en Ciencia de Datos
