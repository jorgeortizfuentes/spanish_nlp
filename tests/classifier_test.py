from spanish_nlp import classifiers

sc = classifiers.SpanishClassifier(model_name="hate_speech", device='cpu')
t1 =  "LAS RATAS QUE ESTÁN EN EL CONGRESO DEBERÍAN SER EXTERMINADAS"
t2 = "El presidente convocó a una reunión a los representantes de los partidos políticos"
p1 = sc.predict(t1)
p2 = sc.predict(t2)

print("Text 1: ", t1)
print("Prediction 1: ", p1)
print("Text 2: ", t2)
print("Prediction 2: ", p2)


# Install package pip in this folder pip (forcing): pip install --force-reinstall -e .