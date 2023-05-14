from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV

class SVMClassifier:

    def __init__(self, X, y, fn, test_size=0.2, num_features=7):
        self.X = X
        self.y = y
        self.fn = fn
        self.test_size = test_size
        self.num_features = num_features
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None
        self.certainty = None
        self.accuracy = None
        self.svm = SVC(probability=True)
        self.best_parameters = None
        
    def preprocess_data(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=self.test_size, random_state=None, shuffle=True)
        self.scale_features()
        self.select_features()
        
    def scale_features(self):
        scaler = MinMaxScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)

    def select_features(self):
        selector = SelectKBest(f_classif, k=self.num_features)
        self.X_train = selector.fit_transform(self.X_train, self.y_train)
        self.X_test = selector.transform(self.X_test)
        self.fn = [self.fn[i] for i, selected in enumerate(selector.get_support()) if selected]
    
    def set_parameters(self):
        param_grid = {'kernel': ['linear', 'rbf', 'poly'], 'C': [0.1, 0.5, 1, 2, 5, 10, 20], 'gamma': ['scale', 'auto']}
        grid_search = GridSearchCV(self.svm, param_grid, cv=2) # creates a grid search object with 2-fold cross-validation
        grid_search.fit(self.X_train, self.y_train)
        self.best_parameters = grid_search.best_params_
        self.svm.set_params(**self.best_parameters)
        
    def get_feature_names(self):
        return self.fn
    
    def get_best_parameters(self):
        return self.best_parameters
    
    def train(self):
        self.svm.fit(self.X_train, self.y_train)
        
    def test(self):
        self.y_pred = self.svm.predict(self.X_test)
        self.certainty = self.svm.predict_proba(self.X_test)
        self.accuracy = accuracy_score(self.y_test, self.y_pred)

    def run(self):
        self.preprocess_data()
        self.set_parameters()
        self.train()
        self.test()
        return self.y_pred, self.y_test, self.certainty, self.accuracy