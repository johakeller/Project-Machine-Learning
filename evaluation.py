import pickle
import csv
import io
import os
import torch
from captum.attr import LayerIntegratedGradients
from captum.attr import visualization as viz
from sklearn.metrics import roc_auc_score

from params import *
from models import *


def write_results(output, file):
    """
    Helper function that writes the output of the training/testing loop into a .txt file in the output_folder.

    Args:
        output (str): Message
        file (str): File path
    """
    os.makedirs(OUTPUT, exist_ok=True)

    with open(os.path.join(OUTPUT, f'{file}.txt'), "a") as file:
        file.write(output)


def shannon_entropy(predictions):
    """
    Calculates the Shannon-based entropy, given a single or multiple predictions (probabilities) of the model for one class. The Shannon-based entropy
    is the measure of uncertainty in the given predictions. If there are multiple predictions given, it returns the average Shannon-based entropy 
    over these predictions.

    Args: 
        predictions (numpy.ndarray): Predicted probabilities of the model for one class

    Returns:
        float: Per class average Shannon-based entropy value of the predictions
    """
    sum = 0
    entropy = 0.0
    if not isinstance(predictions, np.ndarray):
        # Convert single prediction to numpy array
        predictions = np.array([predictions])
    for prediction in predictions:
        sum += 1
        # if the prediction is 0.0, the entropy is also 0.0
        if prediction == 0:
            continue
        else:
            entropy += -(prediction*np.log(prediction))
    if sum == 0:
        return 0
    return entropy/sum

# TASK SHEET MS3
def predict(input, model=None):
    """
    TASK SHEET MS3:
    Predicts labels for the input data and provides confidence scores for these labels by using th Shannon-based entropy.
    It works with pure text comments and with a sigmoid(model output). 

    Args:
        input (torch.Tensor or numpy.ndarray): Input for prediction (either already predictions or text) 
        model (torch.nn.Module): Model used to predict (default: None)

    Returns:
        Tuple: Tuple containing
            numpy.ndarray: Predictions for the classes
            dict: Confidence scores for the classes
    """
    if model is None:
        predictions = input
    else:
        predictions = torch.sigmoid(
            model.forward(input)).cpu().detach().numpy()

    # calculate confidence per label
    confidence_scores = {}
    count = 0
    entropy_sum = 0
    for i in range(predictions.shape[1]):
        confidence = 1.0 - shannon_entropy(predictions[:, i])
        entropy_sum += shannon_entropy(predictions[:, i])
        count += 1
        # key: e.g. toxic_confidence
        label_name = ORDER_LABELS[i] + "_confidence"
        confidence_scores[label_name] = confidence
    confidence_scores["average_confidence"] = 1 - (entropy_sum/count)
    return predictions, confidence_scores


def calc_metrics(labels, predictions, loss, len_dataset, epoch=0):
    """
    Calculates various metrics based on passed true labels and predictions. 

    Args:
        labels (torch.Tensor): True labels for the data
        predictions (torch.Tensor): Predictions of the model for the same data
        loss (float): summed loss per epoch
        len_dataset (int): Length of the input dataset
        epoch (int): Current epoch number (optional, default: 0)

    Returns:
        dict: dictionary containing all the computed metrics, which are: epoch, abg_loss, roc_auc, accuracy, TPR, FPR, TNR, FNR, toxic (ROC-AUC), severe_toxic (ROC-AUC), 
        obscene (ROC-AUC), threat (ROC-AUC), insult (ROC-AUC), identity_hate (ROC-AUC), toxic_confidence (confidence score), severe_toxic_confidence (confidence score), 
        obscene_confidence (confidence score), threat_confidence (confidence score), insult_confidence (confidence score), identity_hate_confidence (confidence score)
        mlcm.cm: 2D multi-label confusion matrix
    """
    T, TN, TP, FP, FN, P, N = 0, 0, 0, 0, 0, 0, 0
    total = 0

    # COMPUTE METRICS
    sigmoid = torch.nn.Sigmoid()
    preds_sigmoid = sigmoid(predictions)
    preds_th = torch.ge(preds_sigmoid, THRESHOLD).int()

    T += (preds_th == labels).sum().item()
    TP += ((preds_th == 1) & (labels == 1)).sum().item()
    FP += ((preds_th == 1) & (labels == 0)).sum().item()
    TN += ((preds_th == 0) & (labels == 0)).sum().item()
    FN += ((preds_th == 0) & (labels == 1)).sum().item()
    P += (labels == 1).sum().item()
    N += (labels == 0).sum().item()

    # sump up total number of labels in batch
    total += labels.nelement()

    labels = labels.cpu().numpy()
    predictions = predictions.cpu().numpy()
    preds_sigmoid = preds_sigmoid.cpu().numpy()
    preds_th = preds_th.cpu().numpy()

    metrics = {
        'epoch': epoch+1,
        'avg_loss': loss / len_dataset,
        'roc_auc': roc_auc_score(labels, predictions, average='macro', multi_class='ovr'),
        'accuracy': T / total,
        'TPR': TP/P,
        'FPR': FP/(FP+TN),
        'TNR': TN/N,
        'FNR': FN/(FN+TP),
        'toxic': roc_auc_score(np.array(labels)[:, 0], np.array(predictions)[:, 0], average='macro', multi_class='ovr'),
        'severe_toxic': roc_auc_score(np.array(labels)[:, 1], np.array(predictions)[:, 1], average='macro', multi_class='ovr'),
        'obscene': roc_auc_score(np.array(labels)[:, 2], np.array(predictions)[:, 2], average='macro', multi_class='ovr'),
        'threat': roc_auc_score(np.array(labels)[:, 3], np.array(predictions)[:, 3], average='macro', multi_class='ovr'),
        'insult': roc_auc_score(np.array(labels)[:, 4], np.array(predictions)[:, 4], average='macro', multi_class='ovr'),
        'identity_hate': roc_auc_score(np.array(labels)[:, 5], np.array(predictions)[:, 5], average='macro', multi_class='ovr')
    }

    # calculate confidence per label and append it
    _, confidence_scores = predict(preds_sigmoid)
    metrics.update(confidence_scores)
    return metrics


class Explainer:
    """
    Class to interpret model predictions and generating visualizations on the base of the Integrated Gradient method.

    Attributes:
        model (torch.nn. Module): Used model to perform predictions
        model_input (torch.nn.Module): The embedding layer of the model

    Methods:
        model_output: Obtains an output for a given model and input
        summarize_attributions: Collects the attributions for interpretation
        construct_input_and_baseline: Builds input tensors and baseline tensors for a given text
        explain: Generates visualizations of the Integrated Gradient method per sample
        explain_samples: Runs the explanation loop for several samples of a given source file
    """

    def __init__(self):
        """
        Initializes the Explainer with a trained model from a file. 
        """

        self.model = Model()

        class CPU_Unpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if module == 'torch.storage' and name == '_load_from_bytes':
                    return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
                else:
                    return super().find_class(module, name)

        # Load the model using CPU_Unpickler
        with open('./output_folder/model.pkl', 'rb') as file:
            self.model = CPU_Unpickler(file).load()
        self.model.eval()
        self.model.to(DEVICE)

        self.model_input = self.model.base_model.embedding

    def model_output(self, *inputs):
        """
        Retrieves the model output for an input.

        Args:
            *inputs: argument list of input tensors

        Returns:
            torch.Tensor: Model output
        """
        out = self.model(*inputs)[0]
        return out.unsqueeze(0)  # torch.sigmoid(out)

    def summarize_attributions(self, attributions):
        """
        Summarizes the attributions for the interpretation.

        Args:
            attributions (torch.Tensor): Attributions obtained from Integrated Gradients

        Returns:
            torch.Tensor: Summarized attributions
        """

        attributions = attributions.sum(dim=-1).squeeze(0)
        attributions = attributions / torch.norm(attributions)

        return attributions

    def construct_input_and_baseline(self, text):
        """
        Creates an tokenized input and a tokenized baseline from the provided comment text. 

        Args:
            text (str): Input text

        Returns:
            Tuple: Tuple containing
                torch.Tensor: Input
                torch.Tensor: Baseline
                List: Token list
        """
        max_length = 510
        baseline_token_id = TOKENIZER.pad_token_id
        sep_token_id = TOKENIZER.sep_token_id
        cls_token_id = TOKENIZER.cls_token_id

        # Encode text with max_length set to 512
        text_ids = TOKENIZER.encode(
            text, truncation=True, add_special_tokens=False)
        # print(text_ids)
        token_list_sum = [cls_token_id] + text_ids + [sep_token_id]
        token_list_1 = TOKENIZER.convert_ids_to_tokens(token_list_sum)
        # print(token_list_1)

        # Pad or truncate to exactly 512 tokens
        pad_length = max_length - len(text_ids)
        if pad_length > 0:
            text_ids += [baseline_token_id] * pad_length
        else:
            text_ids = text_ids[:max_length]

        # Construct input_ids and baseline_input_ids
        input_ids = [cls_token_id] + text_ids + [sep_token_id]
        baseline_input_ids = [cls_token_id] + \
            [baseline_token_id] * len(text_ids) + [sep_token_id]
        token_list_sum = [cls_token_id] + token_list_1 + [sep_token_id]

        # Convert to tensor
        input_ids_tensor = torch.tensor([input_ids], device='cpu')
        baseline_input_ids_tensor = torch.tensor(
            [baseline_input_ids], device='cpu')

        # Convert input_ids to tokens
        # print(input_ids)
        token_list = TOKENIZER.convert_ids_to_tokens(input_ids)
        # print(token_list)

        return input_ids_tensor, baseline_input_ids_tensor, token_list_1

    # TASK SHEET MS3
    def explain(self, text, **labels_and_values):
        """
        TASK SHEET MS3:
        Generates visualization, explaining the model's predictions by overlaying Integrated Gradients and the input tex. 

        Args:
            text (str): Input text
            **labels_and_values: Keyword arguments for the labels and their values
        """
        lig = LayerIntegratedGradients(self.model_output, self.model_input)
        label_to_index_mapping = {'toxic': 0, 'severe_toxic': 1,
                                  'obscene': 2, 'threat': 3, 'insult': 4, 'identity_hate': 5}

        visualizations = []
        print(f'Comment text: {text}')
              
        for label, value in labels_and_values.items():
            print(f'Label: {label}, value: {value}')
            target_class_index = label_to_index_mapping[label]
            print(f'Target class index: {target_class_index}')  # Using the mapping to get the index
            input_ids, baseline_input_ids, all_tokens = self.construct_input_and_baseline(
                text)
            input_ids = input_ids.to(DEVICE)
            baseline_input_ids = baseline_input_ids.to(DEVICE)
            attributions, delta = lig.attribute(inputs=input_ids,
                                                target=target_class_index,  # target_class_index,
                                                baselines=baseline_input_ids,
                                                return_convergence_delta=True,
                                                internal_batch_size=1)
            attributions_sum = self.summarize_attributions(attributions)
            print(f'Attribution: {attributions_sum}')

            score_vis = viz.VisualizationDataRecord(
                word_attributions=attributions_sum,
                pred_prob=torch.sigmoid(self.model(input_ids)[
                                        0][target_class_index]),
                pred_class=torch.sigmoid(self.model(input_ids)[
                                         0][target_class_index]).round(),
                true_class=value,
                attr_class=text,
                attr_score=attributions_sum.sum(),
                raw_input_ids=all_tokens,
                convergence_score=delta
            )

            visualizations.append(score_vis)

        print(viz.visualize_text(visualizations))


    def explain_samples(self, source_file):
        """
        Runs the explanation loop over a source .csv file and performs explain() on the individual samples. 

        Args:
            source_file (str): Path to source file
        """
        with open(source_file, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            i = 0
            for row in reader:
                i += 1
                file_name = f'sample{i}'
                text = row[0].strip('"')
                self.explain(text, toxic=row[1], severe_toxic=row[2],
                             obscene=row[3], threat=row[4], insult=row[5], identity_hate=row[6])
