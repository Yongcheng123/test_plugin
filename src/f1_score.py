def compareKeys(key1, key2):
    if key1[0] == key2[0]:
        return 0
    if key1[0] > key2[0]:
        return 1
    if key1[0] < key2[0]:
        return -1

def worker1_1(params):
    thread_id, p = params
    element_key, label, prediction  = p
    n_of_element = len(prediction)
    element, totPos, truePos, falsNeg = 0, 0, 0, 0
    for i in range(n_of_element):
        try:
            if prediction[i] == element_key:
                if label[i] == element_key:
                    truePos += 1
                    element += 1
                totPos += 1
            else:
                if label[i] == element_key:
                    falsNeg += 1
                    element += 1
        except:
            break  
    data = (element, totPos, truePos, falsNeg)
    return element_key, data

def worker1(params):
    element_key, lab_pred = params
    label, prediction = lab_pred
    n_of_element = len(prediction)
    element, totPos, truePos, falsNeg = 0, 0, 0, 0
    for i in range(n_of_element):
        if prediction[i] == element_key:
            if label[i] == element_key:
                truePos += 1
                element += 1
            totPos += 1
        else:
            if label[i] == element_key:
                falsNeg += 1
                element += 1
        
    data = (element, totPos, truePos, falsNeg)
    return element_key, data

def worker2(params):
    thread_id, params = params
    totalPositive, truePositive, falseNegative, element_key = params
    precision, recall, f1 = 0.0, 0.0, 0.0
    if totalPositive != 0.0:
        precision = float(truePositive)/float(totalPositive)
    if falseNegative != 0.0 or truePositive != 0.0:
        recall = float(truePositive)/float(truePositive + falseNegative)
    sum = precision + recall
    if sum != 0.0:
        f1 = float(2 * precision * recall)/float(sum)
    data = f1, precision, recall
    return element_key, data

def f1Scores(prediction, label):
    from multiprocessing import Pool
    from functools import cmp_to_key
    prediction = prediction.cpu().numpy()
    label = label.cpu().numpy()

    pool = Pool(processes=3)
    pool_returns = pool.map(worker1, enumerate(
        [(label.copy(), prediction.copy()), (label.copy(), prediction.copy()), 
         (label.copy(), prediction.copy())]))

    pool_returns.sort(key=cmp_to_key(compareKeys))
    background, totalBackgroundPositive, backgroundTruePositive, \
        backgroundFalseNegative = pool_returns[0][1]
    helix, totalHelixPositive, helixTruePositive, helixFalseNegative = \
        pool_returns[1][1]
    sheet, totalSheetPositive, sheetTruePositive, sheetFalseNegative = \
        pool_returns[2][1]
    pool.close()
    
    pool = Pool(processes=3)
    args = [(totalBackgroundPositive, backgroundTruePositive, 
             backgroundFalseNegative, 0),
            (totalHelixPositive, helixTruePositive, helixFalseNegative, 1),
            (totalSheetPositive, sheetTruePositive, sheetFalseNegative, 2)]
    pool_returns = pool.map(worker2, enumerate(args))
    pool_returns.sort(key=cmp_to_key(compareKeys))
    backgroundF1, precisionBackground, recallBackground = pool_returns[0][1]
    helixF1, precisionHelix, recallHelix = pool_returns[1][1]
    betasheetF1, precisionBetaSheet, recallBetasheet = pool_returns[2][1]
    pool.close()

    distri_of_classes = str(background)   + ", " + str(helix) + ", " \
            + str(sheet) 

    return helixF1, betasheetF1, backgroundF1, distri_of_classes

