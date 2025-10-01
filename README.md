## Flujo de trabajo comparativo (HPC) para calcular correlación de pares de imagenes

#### 1. Metadata

**Trabajo final del curso de posgrado: Introducción a la programación HPC con Python y sus aplicaciones al procesamiento de imagenes - Instituto M. Gulich (UNC/CONAE), Argentina.**

-   Autor: [Agustín E. Tiberi](agustintiberi.com) - tiberiagustin\@gmail.com / aetiberi\@unse.edu.ar

-   [*Lab. de Sistemas de Información Geográfica, Universidad Nacional de Santiago del Estero, Argentina* ](https://github.com/siglabfcfunse)

-   Versión: 1.0

#### 2. Introducción

Este flujo de trabajo está diseñado para:

-   Desempeñar correlaciones de Pearson en pares de imágenes satelitales (correlación de Pearson).

-   Evaluar la aplicación de diferentes métodos de paralelización durante el cómputo de las correlaciones. Se recomienda hacerlo con 10 o más pares de imágenes para poder observar diferencias de desempeño.

#### 3. Métricas

**3.1 Los métodos de paralelización a evaluar:**

-   V1: secuencial con Numpy

-   V2: multiprocessing

-   V3: Numba JIT

-   V4: PyTorch

**3.2 El archivo** `hpc_pairs.py` **genera un .csv con las columnas llamado `results.csv`**:

-   variant (variante de paralelización: V1, V2, V3 o V4)

-   n (numero de procesos)

-   total_sec (tiempo total del proceso)

-   mean_sec (en caso que tengas más de un n)

-   r_mean: R medio (en caso que tengas más de un n)

-   rmse_mean: RMSE medio (en caso que tengas más de un n)

El CSV de ejemplo se encuenta en la carpeta `results_example`.

**3.3 El archivo** `analyze_results.py` **genera dos gráficos:**

-   `results_times.png` (tiempos medios)

-   `results_speedup.png` (speedup vs V1)

#### 4. Instrucciones

Será necesario como data input un CSV por cada par de imágenes, conteniendo cuatro columnas:

-   Coordenadas X

-   Coordenadas Y

-   Variable 1, Variable 2 (En el ejemplo: "FAI" de MODIS y "CHLA" de Sentinel-2)

Para hacerlo, pre-preocesar las imagenes para obtener misma resolución y grilla. Eliminar pixeles faltantes (NAs). Ver ejemplo de CSV en: `data_example/test_datax.csv`

En el archivo `hpc_pairs.py`, editar en la linea 16 el nombre de las columnas a correlacionar (tus variables).

**Una vez listo, correr los siguientes comandos:**

``` python
# Correr este comando para cada variante de parelización y para cada par de imagenes. Reemplazar la X.

python hpc_pairs.py --pattern "test_dataX.csv" --variant vX

# Ejemplo:

python hpc_pairs.py --pattern "test_data8.csv" --variant v2

# Cada vez que corres, el comando guardará las métricas en un "results.csv" 
# Luego correr el siguiente comando. Generará los dos gráficos de resultados.

python analyze_results.py
```

#### 5. Licencia

```         
© Copyright 2025 Agustin E. Tiberi - Lab. de Sistemas de Información Geográfica, Universidad Nacional de Santiago del Estero

El contenido de este sitio está bajo una licencia CC BY-NC-SA 4.0
https://creativecommons.org/licenses/by-nc-sa/4.0/deed.es
```