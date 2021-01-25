Profeta: procedimiento de pronóstico automático
Construir Pypi_Version Conda_Version

Prophet es un procedimiento para pronosticar datos de series de tiempo basado en un modelo aditivo donde las tendencias no lineales se ajustan a la estacionalidad anual, semanal y diaria, más los efectos de las vacaciones. Funciona mejor con series de tiempo que tienen fuertes efectos estacionales y varias temporadas de datos históricos. Prophet es robusto ante los datos faltantes y los cambios de tendencia, y normalmente maneja bien los valores atípicos.

Prophet es un software de código abierto lanzado por el equipo Core Data Science de Facebook . Está disponible para descargar en CRAN y PyPI .

Links importantes
Inicio: https://facebook.github.io/prophet/
Documentación HTML: https://facebook.github.io/prophet/docs/quick_start.html
Rastreador de problemas: https://github.com/facebook/prophet/issues
Repositorio de código fuente: https://github.com/facebook/prophet
Contribuyendo: https://facebook.github.io/prophet/docs/contributing.html
Paquete Prophet R: https://cran.r-project.org/package=prophet
Paquete Prophet Python: https://pypi.python.org/pypi/fbprophet/
Publicar publicación de blog: https://research.fb.com/prophet-forecasting-at-scale/
Documento del profeta: Sean J. Taylor, Benjamin Letham (2018) Pronóstico a escala. The American Statistician 72 (1): 37-45 ( https://peerj.com/preprints/3190.pdf ).
Instalación en R
Prophet es un paquete CRAN para que puedas usarlo install.packages.

install.packages ( ' profeta ' )
Después de la instalación, ¡puede comenzar!

Ventanas
En Windows, R requiere un compilador, por lo que deberá seguir las instrucciones proporcionadas por rstan. El paso clave es instalar Rtools antes de intentar instalar el paquete.

Si tiene una configuración personalizada del compilador Stan, instálela desde la fuente en lugar del binario CRAN.

Instalación en Python
Prophet está en PyPI, por lo que puede usarlo pippara instalarlo. Desde v0.6 en adelante, Python 2 ya no es compatible.

# Instale pystan con pip antes de usar pip para instalar fbprophet
pip instalar pystan

pip instalar fbprophet
La dependencia predeterminada que tiene Prophet es pystan. PyStan tiene sus propias instrucciones de instalación . Instale pystan con pip antes de usar pip para instalar fbprophet.

También puede elegir un backend estándar alternativo (más experimental) llamado cmdstanpy. Requiere la interfaz de línea de comandos CmdStan y tendrá que especificar la variable de entorno que STAN_BACKENDapunta a ella, por ejemplo:

# bash
$ CMDSTAN=/tmp/cmdstan-2.22.1 STAN_BACKEND=CMDSTANPY pip install fbprophet
Tenga en cuenta que la CMDSTANvariable está directamente relacionada con el cmdstanpymódulo y puede omitirse si sus binarios CmdStan están en su $PATH.

También es posible instalar Prophet con dos backends:

# bash
$ CMDSTAN=/tmp/cmdstan-2.22.1 STAN_BACKEND=PYSTAN,CMDSTANPY pip install fbprophet
Después de la instalación, ¡puede comenzar!

Si actualiza la versión de PyStan instalada en su sistema, es posible que deba reinstalar fbprophet ( consulte aquí ).

Anaconda
Úselo conda install gccpara configurar gcc. La manera más fácil de instalar profeta es a través de Conda-fragua: conda install -c conda-forge fbprophet.

Ventanas
En Windows, PyStan requiere un compilador, por lo que deberá seguir las instrucciones . La forma más sencilla de instalar Prophet en Windows es en Anaconda.

Linux
Asegúrese de que estén instalados los compiladores (gcc, g ++, build-essential) y las herramientas de desarrollo de Python (python-dev, python3-dev). En los sistemas Red Hat, instale los paquetes gcc64 y gcc64-c ++. Si está utilizando una máquina virtual, tenga en cuenta que necesitará al menos 4 GB de memoria para instalar fbprophet y al menos 2 GB de memoria para usar fbprophet.

Registro de cambios
Versión 0.6 (2020.03.03)
Corrija errores relacionados con cambios ascendentes en holidaysy pandaspaquetes.
Compile el modelo durante el primer uso, no durante la instalación (para cumplir con la política de CRAN)
cmdstanpy backend ahora disponible en Python
Python 2 ya no es compatible
Versión 0.5 (2019.05.14)
Estacionalidades condicionales
Estimaciones de validación cruzada mejoradas
Trazar gráficamente en Python
Corrección de errores
Versión 0.4 (2018.12.18)
Funcionalidad de vacaciones agregada
Corrección de errores
Versión 0.3 (2018.06.01)
Estacionalidad multiplicativa
Métricas y visualizaciones de errores de validación cruzada
Parámetro para establecer el rango de puntos de cambio potenciales
Modelo de Stan unificado para ambos tipos de tendencias
Incertidumbre de tendencia futura mejorada para datos subdiarios
Corrección de errores
Versión 0.2.1 (2017.11.08)
Corrección de errores
Versión 0.2 (2017.09.02)
Pronóstico con datos subdiarios
Estacionalidad diaria y estacionalidades personalizadas
Regresores adicionales
Acceso a muestras predictivas posteriores
Función de validación cruzada
Mínimos saturantes
Corrección de errores
Versión 0.1.1 (2017.04.17)
Corrección de errores
Nuevas opciones para detectar la estacionalidad anual y semanal (ahora la predeterminada)
Versión 0.1 (2017.02.23)
Versión inicial
Licencia
Prophet tiene la licencia MIT .
