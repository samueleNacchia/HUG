import numpy as np

class SmartDepressionClassifier:
    def __init__(self, pipeline, thresholds):
        self.pipeline = pipeline
        self.thresholds = thresholds

    def predict(self, X):
        probs = self.pipeline.predict_proba(X)[:, 1]
        final_thresholds = np.full(len(X), 0.5)

        # Uso la versione ottimizzata che abbiamo discusso
        X_str = X.astype(str)
        for feature, mapping in self.thresholds.items():
            if feature in X_str.columns:
                for val, custom_t in mapping.items():
                    mask = (X_str[feature] == str(val))
                    final_thresholds[mask] = custom_t
        return (probs >= final_thresholds).astype(int)

    def predict_proba(self, X):
        return self.pipeline.predict_proba(X)