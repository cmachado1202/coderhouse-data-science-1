# coderhouse-data-science-1
Preentregas de Data Science 1 — Carolina Machado

## Continuidad del proyecto

Se conserva el notebook de la preentrega 2 sin modificar y se continúa con **Bank Marketing (UCI), bank-full.csv**: 45.211 filas y 17 columnas originales. El objetivo exploratorio es estudiar factores asociados a la contratación de un depósito antes del contacto. No se entrenó ni desplegó un modelo.

- `DataScience1_Preentrega3_CarolinaMachado.ipynb`: notebook con código y salidas ejecutadas, tipos, nulos, describe, tres filtros booleanos y tres agregaciones.
- `data/bank-full.csv.gz`: copia comprimida sin alteraciones del CSV original, separado por punto y coma. El notebook lo descomprime automáticamente en `data/bank-full.csv` si todavía no existe.
- `data/bank-names.txt`: diccionario original de esta versión.
- `data/bank_saneado.csv`: base derivada que genera el notebook, sin duration, con indicadores descriptivos y categorías de contacto previo; no se duplica en Git.
- `generar_reportes.py`: reconstruye el notebook y los informes PDF de estructura inicial y EDA visual; el EDA contiene cinco gráficos con interpretación.

## Reproducir

Python 3.11 o superior. Crear un entorno virtual e instalar `pip install -r requirements.txt`. Desde la raíz del repositorio, ejecutar `python generar_reportes.py`. El original se descomprime localmente, se genera el CSV saneado y el EDA lee ese archivo saneado. Los PDF se guardan en `informes/` y los gráficos intermedios en `tmp/graficos/`. No necesita descargar el dataset, claves, servicios pagos ni datos privados.

## Decisiones del análisis

No se detectaron nulos técnicos ni duplicados exactos. Se documenta `unknown` como categoría informativamente faltante y `pdays=-1` como ausencia de contacto previo; se conservan saldos negativos y extremos válidos. Los tres filtros de dominio pueden excluir cero filas: no se simulan errores inexistentes. Se elimina `duration` de la base predictiva por conocerse después de la llamada. `day` y `month` no alcanzan para reconstruir una fecha completa porque falta el año por registro.

Se presentan asociaciones descriptivas. La muestra corresponde a una institución y período históricos; no prueba causalidad ni justifica decisiones automáticas sobre personas. Un futuro modelo requiere definir el instante de predicción, separar entrenamiento y evaluación y revisar sesgos.

## Fuente y licencia

Moro, S., Rita, P. y Cortez, P. (2014). **Bank Marketing** [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5K306 — https://archive.ics.uci.edu/dataset/222/bank+marketing.

Fuente del paquete original: https://archive.ics.uci.edu/static/public/222/bank+marketing.zip. Licencia CC BY 4.0, con atribución. La base `bank_saneado.csv` es una transformación documentada del original, no una fuente independiente. No se usa `bank-additional-full.csv`.
