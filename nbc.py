from numpy import std, mean, prod
from scipy.stats import norm
from math import isclose

train = {
        "pronostico": [
            "soleado", "soleado", "nublado", "lluvioso", "lluvioso", "lluvioso", "nublado", "soleado",
            "soleado", "lluvioso", "soleado", "nublado", "nublado", "lluvioso"
            ],
        "temperatura": [
            36, 28, 30, 20, 2, 5, 11, 22, 9, 17, 19, 22, 27, 21
            ],
        "humedad": [
            "alta", "alta", "alta", "alta", "normal", "normal", "normal", "alta", "normal", "normal",
            "normal", "alta", "normal", "alta"
            ],
        "viento": [
            "leve", "fuerte", "leve", "leve", "leve", "fuerte", "fuerte", "leve", "leve", "leve",
            "fuerte", "fuerte", "leve", "fuerte"
            ],
        "asado": [
            "no", "no", "si", "si", "si", "no", "si", "no", "si", "si", "si", "si", "si", "no"
            ]
        }

# lista de características
def get_features(dataset:dict, label:str) -> list:
    features = []
    for feature in dataset:
        if feature != label:
            features.append(feature)
    return features

# P(H), probabilidad de la clase (Naive)
def prob_class(dataset:dict, label:str, target:str) -> float:
    return dataset[label].count(target) / float(len(train[label]))

# P(E_n|H); prob. de cierta característica dado que ocurre la clase
def prob_feature_class(dataset:dict, label:str, target:str, feature:str, feature_target:str) -> float:
    target_indexes = [i for i, x in enumerate(dataset[label]) if x == target]
    partition = []
    for index in target_indexes:
        partition.append(dataset[feature][index])
    if type(partition[0]) in (int, float):  # dato numérico
        return norm.cdf(feature_target, mean(partition), std(partition))
    else:   # dato nominal
        return partition.count(feature_target) / float(len(partition))

# P(E|H); prob de todas las características dado que ocurre la clase
def prob_features_class(dataset:dict, label:str, target:str, sample:list) -> float:
    peh = []
    features = get_features(dataset, label)
    for i in range(len(sample)):
        peh.append(prob_feature_class(dataset, label, target, features[i], sample[i]))
    return prod(peh)

# P(E); probabilidad de todas las características
def prob_features(dataset:dict, label:str, sample:list) -> float:
    pe = 0.0
    for clas in set(dataset[label]):
        pe += prob_features_class(dataset, label, clas, sample) * prob_class(dataset, label, clas)
    return pe

# P(H|E); probabilidad de la clase dadas ciertas características (Naive-Bayes)
def prob_class_features(dataset:dict, label:str, sample:list) -> dict:
    # TODO: prob_features_class se llama dos veces, optimizar.
    phes = {}
    pe = prob_features(dataset, label, sample)
    for clas in set(dataset[label]):
        peh = prob_features_class(dataset, label, clas, sample)
        ph = prob_class(dataset, label, clas)
        phes[clas] = peh * ph / pe
    return phes

# TESTS
#assert isclose(prob_class_features(train, "asado", ["soleado", 19, "normal", "leve"])['si'], 0.85, abs_tol=1e-2)
print(prob_class_features(train, "asado", ["soleado", 19, "normal", "leve"]))
print(prob_class_features(train, "asado", ["lluvioso", 34, "alta", "leve"]))
print(prob_class_features(train, "asado", ["nublado", 14, "normal", "fuerte"]))


# print("SI")
# print("PRON: {}".format(prob_feature_class(train, "asado", "si", "pronostico", "soleado")))
# print("TEMP: {}".format(prob_feature_class(train, "asado", "si", "temperatura", 19)))
# print("HUME: {}".format(prob_feature_class(train, "asado", "si", "humedad", "normal")))
# print("VIEN: {}".format(prob_feature_class(train, "asado", "si", "viento", "leve")))
# print("\nNO")
# print("PRON: {}".format(prob_feature_class(train, "asado", "no", "pronostico", "soleado")))
# print("TEMP: {}".format(prob_feature_class(train, "asado", "no", "temperatura", 19)))
# print("HUME: {}".format(prob_feature_class(train, "asado", "no", "humedad", "normal")))
# print("VIEN: {}".format(prob_feature_class(train, "asado", "no", "viento", "leve")))
