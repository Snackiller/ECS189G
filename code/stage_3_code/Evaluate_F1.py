from code.base_class.evaluate import evaluate
from sklearn.metrics import f1_score


class Evaluate_F1(evaluate):
    data = None

    def evaluate(self):
        return f1_score(self.data['true_y'], self.data['pred_y'], average="weighted")