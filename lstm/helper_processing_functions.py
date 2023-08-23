def min_max_scaler(series, minimum, maximum):
    return (series - minimum)/(maximum - minimum)

def min_max_inverse_scaler(series, minimum, maximum):
    return (series*(maximum - minimum)) + minimum

def create_dataset(dataset, l, lookback):
    X, y = [], []
    for i in range(len(dataset) - lookback):
        feature = [[float(m)] for m in dataset[i:i+lookback]]
#         feature = [[m,n] for m,n in zip(dataset[i:i + lookback], l[i:i+lookback])]
        target = l[i+1:i+lookback+1]
        target = [[float(m)] for m in l[i+1:i+lookback+1]]
        X.append(feature)
        y.append(target)
    return torch.tensor(X), torch.tensor(y)