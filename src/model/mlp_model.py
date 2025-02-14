from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

def train_mlp(X_train, X_test, y_train, y_test):
    """
    Trains a Multilayer Perceptron classifier on the training data and evaluates it on the test data.
    """
    # Standardize features by removing the mean and scaling to unit variance
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize the MLPClassifier
    mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, activation='relu', solver='adam', random_state=42)
    
    # Train the model
    mlp.fit(X_train_scaled, y_train)
    
    # Predict on the test set
    y_pred = mlp.predict(X_test_scaled)
    
    print("Accuracy MLP: ", accuracy_score(y_test, y_pred))
    print("Classification Report MLP:\n", classification_report(y_test, y_pred))
    
    return mlp