import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets

class NaiveBayes:
    def fit(self, X, y):
        # Naive Bayes modelini moslash
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)
        
        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self._var = np.zeros((n_classes, n_features), dtype=np.float64)
        self._periors = np.zeros(n_classes, dtype=np.float64)
        
        # Har bir xususiyat uchun o'rtacha, dispersiya va klass priorlarini hisoblash
        for idx, c in enumerate(self._classes):
            X_c = X[y == c]
            self._mean[idx, :] = X_c.mean(axis=0)
            self._var[idx, :] = X_c.var(axis=0)
            self._periors[idx] = X_c.shape[0] / float(n_samples)
            
    def predict(self, X):
        # Naive Bayes modeli orqali bashoratlarni aniqlash
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)
    
    def _predict(self, x):
        # Bitta xususiyat uchun klassni bashorat qilish
        posteriors = []
        
        for idx, c in enumerate(self._classes):
            prior = np.log(self._periors[idx])
            posterior = np.sum(np.log(self._pdf(idx, x)))
            posterior = posterior + prior
            posteriors.append(posterior)
            
        return self._classes[np.argmax(posteriors)]
    
    def _pdf(self, class_idx, x):
        # Bitta xususiyat uchun klassga tegishli shartli qonuniy guruchni hisoblash
        mean = self._mean[class_idx]
        var = self._var[class_idx]
        numerator = np.exp(-((x - mean)**2) / (2 * var)) 
        denominator = np.sqrt(2 * np.pi * var) 
        return numerator / denominator

if __name__ == "__main__":
    def accuracy(y_true, y_pred):
        # Klassifikatsiya aniqliligi hisoblash
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy
    
    # Sintetik ma'lumot generatsiya qilish
    X, y = datasets.make_classification(
        n_samples=1000, n_features=10, n_classes=2, random_state=123
    )
    
    # Ma'lumotlarni o'qish va test qilish uchun bo'lib bo'lish
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123
    )
    
    # Naive Bayes modelini yaratish va o'qitish
    model = NaiveBayes()
    model.fit(X_train, y_train)
    
    # Bashoratlarni aniqlash va aniqlilikni baholash
    predictions = model.predict(X_test)
    
    print("Naive Bayes klassifikatsiya aniqliligi:", accuracy(y_test, predictions))
