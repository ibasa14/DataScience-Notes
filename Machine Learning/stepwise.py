
def stepwise(data, response):
    '''
    data: conjunto completo de datos
    response: la variable dependiente
    '''
    # Creamos un set (conjunto con valores unicos) con todas las caracteríticas a evaluar y quitamos la variable dependiente
    remaining = set(data.columns)
    remaining.remove(response)
    selected = []
    current_score, best_new_score = float('inf'), float('inf')
    
    while True:
        #bandera para gestionar la salida del bucle
        changed = False
        #forward step
        scores_with_candidates = []
        # Recorremos cada uno de los valores que aún existen en remaining
        # Para cada uno de ellos, entrenamos el modelo y guardamos la metrica que se desee (en este caso aic) 
        for candidate in remaining:
            formula = "{} ~ {}".format(response,
                                           ' + '.join(selected + [candidate]))
            y, x = dmatrices(formula, data)
            score = OLS(y, x).fit().aic
            scores_with_candidates.append((score, candidate))
        # Ordenamos de mayor a menor
        scores_with_candidates.sort(reverse = True)
        # Nos quedamos con la mejor opción, es decir la que tiene el valor más bajo
        best_new_score_forward, best_candidate = scores_with_candidates.pop()

        #backward step
        if len(selected) > 1:
            scores_with_candidates = []
            for candidate in selected:
                variables_aux = selected.copy()
                variables_aux.remove(candidate)
                formula = "{} ~ {}".format(response, ' + '.join(variables_aux))
                y, x = dmatrices(formula, data)
                score = OLS(y, x).fit().aic
                scores_with_candidates.append((score, candidate))
            # Ordenamos de menor a mayor
            scores_with_candidates.sort()
            # Nos quedamos con la peor opcion, es decir la que tiene el valor más alto
            worst_score, worst_candidate = scores_with_candidates.pop()

            # Primero comprobamos que el peor valor no es peor que lo que tenemos actualmente
            # damos prioridad a quitar variables antes que añadir nuevas
            if current_score > worst_score:
                selected.remove(worst_candidate)
                remaining.append(worst_candidate)
                current_score = worst_score
                continue
        # En caso de que el nuevo valor sea inferior que el almacenado actualmente, lo guardamos como seleccionado y
        # actualizamo el mejor valor
        if current_score > best_new_score_forward:
            remaining.remove(best_candidate)
            selected.append(best_candidate)
            current_score = best_new_score_forward
            continue
        # En caso de que no se añada ni se quite ninguna variable salimos del bucle
        break
    return selected
