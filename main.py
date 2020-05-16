import pandas as pd
from matplotlib import pyplot as plt
import math
import random

'''
    Vamos a encerrar los coeficientes de la función en un objeto
'''
class Elemento:
    # Constructor del elemento a optimizar
    def __init__(self, alpha, mu, betha):
        self.alpha = alpha
        self.mu = mu
        self.betha = betha
        self.calificacion = 0.0

'''
    Algoritmo Genético para intentar dar aproximación a los coeficientes alfa, beta, mu de la función de campana
    @param tamaño de la población
    @param Número de generaciones que iterará
    @param coeficiente de mutación de los elementos de la población
    @param método cross over que se utilizará
    @param datos a predecir
'''
class AlgoritmoGenetico:
    # Constructor del algoritmo genético
    def __init__(self, tampoblacion, generaciones, coeficiente_mutacion, metodo_cross_over, datos_a_predecir):
        self.tampoblacion = tampoblacion
        self.generaciones = generaciones
        self.coeficiente_mutacion = coeficiente_mutacion
        self.datos_a_predecir = datos_a_predecir
        self.metodo_cross_over = metodo_cross_over
        self.poblacion  = []
        self.mejorPostor = None
        self.inicializa_poblacion()
    # Inicializa población
    def inicializa_poblacion(self):
        # Vamos a crear la población de elementos
        for i in range(0, self.tampoblacion):
            # Incluimos un elemento con los valores alpha, mu y betha
            self.poblacion.append(Elemento(random.randrange(1500, 2500), random.randrange(60, 80), random.randrange(100, 600)))
        # Iteramos por cada elemento de la población para asignarle una medida fit
        self.mejorPostor = self.poblacion[0]
        for elemento in self.poblacion:
            elemento.calificacion = self.funcion_objetivo(elemento)
            if(elemento.calificacion < self.mejorPostor.calificacion):
                self.mejorPostor = elemento

    '''
        Función para aproximar la medida
        @param elemento a optimizar con los valores de la función
        @param x (dataFrame)
        @return predicción de acuerdo a los parámetros
    '''
    def campana_datos(self, elemento, x):
        if(elemento.betha == 0):
            elemento.betha = 1
        return elemento.alpha * math.pow(math.e,-(math.pow((x - elemento.mu),2)/elemento.betha))

    '''
        Queremos que se minimice la función de error cuadrático medio
        @param Elemento de la población
        @return valor de la función de error cuadrático medio para el elemento de la población
    '''
    def funcion_objetivo(self, elemento):
        suma = 0 # Suma del error cuadrático medio
        dia = 12 #Iniciamos en el día 12 pues las medidas con las que contamos inician de 12 hasta 77
        for medida in self.datos_a_predecir:
            suma = suma + (math.pow((int(medida) - self.campana_datos(elemento, dia)), 2)/len(self.datos_a_predecir))
            dia = dia + 1 #Incrementamos el día en uno
        return suma
    # Método de selección de dos elementos de la población
    def selecciona(self):
        indice1 = int(random.random() * len(self.poblacion))
        indice2 = int(random.random() * len(self.poblacion))
        promedio_errores = (self.poblacion[indice1].calificacion + self.poblacion[indice2].calificacion)/2;
        madre = self.poblacion[indice1]
        padre = self.poblacion[indice2]
        while(madre.calificacion < promedio_errores):
            indice1 = int(random.random() * len(self.poblacion))
            madre = self.poblacion[indice1]
        while(padre.calificacion < promedio_errores):
            indice2 = int(random.random() * len(self.poblacion))
            padre = self.poblacion[indice2]
        return padre, madre

    # Método de selección de los elementos
    '''
        Sección con los métodos crossOver posibles
    '''
    #Metodo para seleccionar el metodo de cruzamiento con base en el parámetro inicil del algoritmo
    def seleccionarMetodoCruza(self,padre, madre):
        metodosCruzamiento = {
            "Cross_Over": self.cross_Over,
            "Cross_over_probabilistico": self.cross_over_probabilistico,
        }
        func = metodosCruzamiento.get(self.metodo_cross_over)
        return func(padre, madre)
    # Cross over alternando
    def cross_Over(self, madre, padre):
        val = { "alfaM": madre.alpha, "alfaP": padre.alpha,
                "betaM": madre.betha, "betaP": padre.betha,
                "muM": madre.mu, "muP": padre.mu
        }
        bart = Elemento(val.get("alfaP"), val.get("muM"), val.get("betaP"))
        lisa = Elemento(val.get("alfaM"), val.get("muP"), val.get("betaM"))
        return bart, lisa
    # Cross over probabilístico
    def cross_over_probabilistico(self, madre, padre):
        val = { "alfaM": madre.alpha, "alfaP": padre.alpha,
                "betaM": madre.betha, "betaP": padre.betha,
                "muM": madre.mu, "muP": padre.mu
        }
        bart = Elemento(val.get(random.choice(["alfaM", "alfaP"])) , val.get(random.choice(["betaM", "betaP"])), val.get(random.choice(["muM", "muP"])))
        lisa = Elemento(val.get(random.choice(["alfaM", "alfaP"])) , val.get(random.choice(["betaM", "betaP"])), val.get(random.choice(["muM", "muP"])))
        return bart, lisa
    # Mutacion
    def mutacion(self, hijoA):
        # Para la mutación generaremos una epsilon que corresponde al radio de una vecindad, y posteriormente le sumaremos/restaremos el valor
        # Mutación del coeficiente alpha
        val = random.random()
        if(val < self.coeficiente_mutacion):
            # Generamos una vecindad
            valAleatorioAcotado = (hijoA.alpha/2) * random.uniform(0.01, self.coeficiente_mutacion)
            hijoA.alpha = hijoA.alpha - valAleatorioAcotado
        # Mutación del coeficiente mu
        val = random.random()
        if(val < self.coeficiente_mutacion):
            # Generamos una vecindad
            valAleatorioAcotado = (hijoA.mu/2) * random.uniform(0.01, self.coeficiente_mutacion)
            hijoA.mu = hijoA.mu - valAleatorioAcotado
        # Mutación del coeficiente betha
        val = random.random()
        if(val < self.coeficiente_mutacion):
            # Generamos una vecindad
            valAleatorioAcotado = (hijoA.betha/2) * random.uniform(0.01, self.coeficiente_mutacion)
            hijoA.betha = hijoA.betha - valAleatorioAcotado
        return hijoA
    '''
        Método incial del algoritmo que ejecuta absolutamente todo
    '''
    def ejecuta(self):
        # Vamos a empezar a iterar por cada generación
        for i in range(0, self.generaciones):
            nuevaPoblacion = []
            nuevaPoblacion.append(self.mejorPostor)
            # Aquí ya tenemos todos los individuos de la población calificados
            for j in range(0, int(self.tampoblacion/2)):
                marge, homero = self.selecciona()
                bart, lisa    = self.seleccionarMetodoCruza(marge, homero)
                bart  = self.mutacion(bart)
                lisa = self.mutacion(lisa)
                nuevaPoblacion.append(bart)
                nuevaPoblacion.append(lisa)
            # Redefinimos la población
            self.poblacion = nuevaPoblacion
            # Iteramos por cada elemento de la población
            for elemento in self.poblacion:
                elemento.calificacion = self.funcion_objetivo(elemento)
                if(elemento.calificacion < self.mejorPostor.calificacion):
                    self.mejorPostor = elemento
        #self.resultados()
    '''
        Método que reporta los resultados del algoritmo
    '''
    def resultados(self):
        dia = 12
        for elemento in self.poblacion:
            print("Coeficiente Alfa: "+str(elemento.alpha)+" , Coeficiente Betha: "+str(elemento.betha)+" , Coefficiente Mu: "+str(elemento.mu))
            print("Error cuadrático: "+str(self.funcion_objetivo(elemento)) + " , f(x) := "+str(self.campana_datos(elemento, dia))+"\n")
            print("=====================================================================================")
            dia = dia + 1
        res = []
        dia = 12
        print("El mejor postor: ")
        print("Alpha: "+str(self.mejorPostor.alpha)+ " , Mu: "+str(self.mejorPostor.mu)+" , Betha: "+str(self.mejorPostor.betha)+" , Error Cuadrático Medio: "+str(self.funcion_objetivo(self.mejorPostor)) )
        # Vamos a graficar los resultados con los valores obtenidos de alpha, betha y mu
        for data in self.datos_a_predecir:
            res.append(self.campana_datos(self.mejorPostor, dia))
            dia = dia  + 1
        plt.plot(self.datos_a_predecir)
        plt.plot(res)
        plt.show()

# Tenemos la columna de Día y de Nuevos Casos
data = pd.read_csv("DatosDiego.csv")

mejor = AlgoritmoGenetico(159, 70, random.random(),"Cross_Over", data["Nuevos Casos"])

for i in range(0, 30):
    ag = AlgoritmoGenetico(159, 70, random.random(),"Cross_Over", data["Nuevos Casos"])
    ag.ejecuta()
    if(ag.mejorPostor.calificacion < mejor.mejorPostor.calificacion):
        mejor = ag
    print("Alpha: " + str(ag.mejorPostor.alpha) +" , Betha: "+ str(ag.mejorPostor.betha) + " , Mu: "+str(ag.mejorPostor.mu) + " Error: "+str(ag.funcion_objetivo(ag.mejorPostor)))

mejor.resultados()
