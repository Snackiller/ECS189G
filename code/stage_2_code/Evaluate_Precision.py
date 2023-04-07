from code.base_class.evaluate import evaluate
from sklearn.metrics import precision_score


class Evaluate_Precision(evaluate):
    data = None

    def evaluate(self):
        return precision_score(self.data['true_y'], self.data['pred_y'], average="weighted", zero_division=0)

