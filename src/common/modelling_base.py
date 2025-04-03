import pathlib
import pickle
import time
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from captum.attr import IntegratedGradients, NoiseTunnel
from captum.attr import visualization as viz
from matplotlib import pyplot
from matplotlib.colors import LinearSegmentedColormap
from scipy.spatial import distance
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data_set import CustomImageDataset
from .diagnosis_mapping import BENIGN_DIAGNOSIS, DIAGNOSIS_TO_TARGET
from .path_constants import MODEL_PATH_BINARY, MODEL_PATH_MULTI, SALIENCY_MAP_PATH
from .result_to_csv import write_csv_row
from .utils import display_images, load_indices, save_indices

MODELNAME_MULTI = 'efficient_net_v2_s_multi'
MODELNAME_BINARY = 'efficient_net_v2_s_binary'
MODEL_PATHS = {
    MODELNAME_MULTI: MODEL_PATH_MULTI,
    MODELNAME_BINARY: MODEL_PATH_BINARY
}

class ModelHandler:
    def __init__(self, dataset, indices=None, inference_model=None, embedding_model=None, ood_model=None, model_name=MODELNAME_MULTI):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.inference_model: InferenceModel = inference_model
        self.embedding_model: EmbeddingModel = embedding_model
        self.ood_model: OODModel = ood_model
        self.dataset: CustomImageDataset = dataset
        self.indices: dict = indices # Should contain 3 subarrays train_indices, val_indices, test_indices in that order
        self.model_name: str = model_name

    def build_inference_model(self, model_name=None):
        if model_name is None:
            model_name = self.model_name

        if model_name == MODELNAME_MULTI:
            model = create_efficientnet()
            self.inference_model = InferenceModelMulti(model)
        elif model_name == MODELNAME_BINARY:
            model = create_efficientnet(multiclass=False)
            self.inference_model = InferenceModelBinary(model)
        else:
            raise Exception("Unknown model name")

    def build_embedding_model(self):
        print("Set embedding model")
        self.embedding_model = EmbeddingModel(self.inference_model.model)

    def store_models(self):
        # Create the directory including parents if needed, and don't raise an error if it already exists
        if self.model_name in (MODELNAME_MULTI, MODELNAME_BINARY):
            current_model_directory_path = MODEL_PATHS[self.model_name] / time.strftime("%Y_%m_%d-%H_%M_%S")
            current_model_directory_path.mkdir(parents=True, exist_ok=True)
        else:
            raise Exception("Unknown model name")

        # Create directory for model infos
        current_model_info_directory_path = current_model_directory_path / 'info'
        current_model_info_directory_path.mkdir(parents=True, exist_ok=True)

        # Save the original model
        torch.save(self.inference_model.model, current_model_directory_path.resolve() / (self.model_name + '.pth'))
        print("Saved model")
        
        if self.indices is not None:
            save_indices(current_model_directory_path.resolve() / 'indices.npz', *self.indices)

        if len(self.inference_model.loss_history) != 0:
            # Unpack the training and validation losses into separate lists
            training_losses, validation_losses = zip(*self.inference_model.loss_history)

            # Epochs are simply a range from 1 to the length of the loss history
            epochs = range(1, len(self.inference_model.loss_history) + 1)

            # Create the plot
            plt.figure(figsize=(10, 6))
            plt.plot(epochs, training_losses, label='Training Loss', marker='o')
            plt.plot(epochs, validation_losses, label='Validation Loss', marker='o')

            # Adding titles and labels
            plt.title('Training and Validation Losses Over Epochs')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.xticks(epochs)

            plt.legend()
            plt.savefig(current_model_info_directory_path.resolve() / 'training_validation_losses.png')
            plt.close()  # Close the figure to free up memory

            print("Saved loss history image")

        # Dump PCA model
        if self.embedding_model.pca is not None:
            with open(current_model_directory_path.resolve() / 'pca_model.pkl', 'wb') as file:
                pickle.dump(self.embedding_model.pca, file)
                print("Dumped PCA model")

        # Dump the embedding
        if self.embedding_model.embeddings is not None:
            np.save(current_model_directory_path.resolve() / 'embeddings.npy', self.embedding_model.embeddings)
            print("Saved embeddings")

        # Dump the reduced embedding
        if self.embedding_model.reduced_embeddings is not None:
            np.save(current_model_directory_path.resolve() / 'reduced_embeddings.npy',
                    self.embedding_model.reduced_embeddings)
            print("Saved reduced embeddings")

        # Dump the normalized embedding
        if self.embedding_model.normalized_embeddings is not None:
            np.save(current_model_directory_path.resolve() / 'normalized_embeddings.npy',
                    self.embedding_model.normalized_embeddings)
            print("Saved normalized embeddings")

        # Dump the predictions
        if self.inference_model.predictions is not None:
            np.save(current_model_directory_path.resolve() / 'predictions.npy', self.inference_model.predictions)
            print("Saved predictions")

        # Dump the prediction probabilities
        if self.inference_model.probs is not None:
            np.save(current_model_directory_path.resolve() / 'probs.npy', self.inference_model.probs)
            print("Saved prediction probabilities")

        # Dump the sorted OOD distances (truncated)
        if self.ood_model.sorted_distances is not None:
            np.save(current_model_directory_path.resolve() / 'sorted_distances_trunc.npy', self.ood_model.sorted_distances)
            print("Saved truncated sorted pairwise embedding distances")

        # Dump the sorted normalized OOD distances (truncated)
        if self.ood_model.sorted_normalized_distances is not None:
            np.save(current_model_directory_path.resolve() / 'sorted_normalized_distances_trunc.npy', self.ood_model.sorted_normalized_distances)
            print("Saved truncated sorted pairwise embedding normalized distances")

    @staticmethod
    def load_models(dataset, time_stamp_path=None, model_name=MODELNAME_MULTI):
        if model_name in (MODELNAME_MULTI, MODELNAME_BINARY):
            _MODEL_PATH = MODEL_PATHS[model_name]
        else:
            raise Exception("Unknown model name")

        if time_stamp_path is None:
            model_sub_dirs = [dir for dir in _MODEL_PATH.iterdir() if dir.is_dir()]
            most_recent_model_dir = sorted(model_sub_dirs, key=lambda x: x.name)[-1]
            time_stamp_path = most_recent_model_dir.name

            if model_name is None:
                pth_files = list(most_recent_model_dir.glob("*.pth"))
                if len(pth_files) == 1:
                    model_name = pth_files[0].name[:-4]  # Remove .pth postfix
                elif len(pth_files) == 0:
                    raise FileNotFoundError("No .pth file found in the directory.")
                else:
                    raise Exception("There is more than one .pth file in the directory.")

        indices = None
        if pathlib.Path(f"{_MODEL_PATH.resolve()}/{time_stamp_path}/indices.npz").exists():
            indices = load_indices(f"{_MODEL_PATH.resolve()}/{time_stamp_path}/indices.npz")
            print("Loaded indices")
        else:
            print("No saved indices found")

        embeddings = None
        if pathlib.Path(f"{_MODEL_PATH.resolve()}/{time_stamp_path}/embeddings.npy").exists():
            embeddings = np.load(f"{_MODEL_PATH.resolve()}/{time_stamp_path}/embeddings.npy")
            print("Loaded embeddings")
        else:
            print("No saved embeddings found")

        reduced_embeddings = None
        if pathlib.Path(f"{_MODEL_PATH.resolve()}/{time_stamp_path}/reduced_embeddings.npy").exists():
            reduced_embeddings = np.load(f"{_MODEL_PATH.resolve()}/{time_stamp_path}/reduced_embeddings.npy")
            print("Loaded reduced embeddings")
        else:
            print("No saved reduced embeddings found")

        normalized_embeddings = None
        if pathlib.Path(f"{_MODEL_PATH.resolve()}/{time_stamp_path}/normalized_embeddings.npy").exists():
            normalized_embeddings = np.load(f"{_MODEL_PATH.resolve()}/{time_stamp_path}/normalized_embeddings.npy")
            print("Loaded normalized embeddings")
        else:
            print("No saved normalized embeddings found")

        pca = None
        if pathlib.Path(f"{_MODEL_PATH.resolve()}/{time_stamp_path}/pca_model.pkl").exists():
            with open(f"{_MODEL_PATH.resolve()}/{time_stamp_path}/pca_model.pkl", 'rb') as file:
                pca: PCA = pickle.load(file)
                print("Loaded PCA model")
        else:
            print("No saved PCA model found")

        predictions = None
        if pathlib.Path(f"{_MODEL_PATH.resolve()}/{time_stamp_path}/predictions.npy").exists():
            predictions = np.load(f"{_MODEL_PATH.resolve()}/{time_stamp_path}/predictions.npy")
            print("Loaded predictions")
        else:
            print("No saved predictions found")

        probs = None
        if pathlib.Path(f"{_MODEL_PATH.resolve()}/{time_stamp_path}/probs.npy").exists():
            probs = np.load(f"{_MODEL_PATH.resolve()}/{time_stamp_path}/probs.npy")
            print("Loaded prediction probabilities")
        else:
            print("No saved prediction probabilities found")

        map_location = None if torch.cuda.is_available() else torch.device('cpu')
        model = torch.load(f"{_MODEL_PATH.resolve()}/{time_stamp_path}/{model_name}.pth", map_location=map_location)
        print(f"Loaded {model_name} from {_MODEL_PATH.name}/{time_stamp_path}")
        
        if model_name == MODELNAME_MULTI:
            inference_model = InferenceModelMulti(model,
                                                  probs=probs,
                                                  predictions=predictions)
        elif model_name == MODELNAME_BINARY:
            inference_model = InferenceModelBinary(model,
                                                   probs=probs,
                                                   predictions=predictions)
        else:
            raise Exception("Unknown model name")
        print("Created inference model")
        
        embedding_model = EmbeddingModel(model, pca=pca,
                                        embeddings=embeddings,
                                        reduced_embeddings=reduced_embeddings,
                                        normalized_embeddings=normalized_embeddings)
        print("Created embedding model")

        sorted_distances_truncated = None
        if pathlib.Path(f"{_MODEL_PATH.resolve()}/{time_stamp_path}/sorted_distances_trunc.npy").exists():
            sorted_distances_truncated = np.load(f"{_MODEL_PATH.resolve()}/{time_stamp_path}/sorted_distances_trunc.npy")
            print("Loaded truncated sorted pairwise embedding distances")
        else:
            print("No saved truncated sorted pairwise embedding distances found")

        sorted_normalized_distances_truncated = None
        if pathlib.Path(f"{_MODEL_PATH.resolve()}/{time_stamp_path}/sorted_normalized_distances_trunc.npy").exists():
            sorted_normalized_distances_truncated = np.load(f"{_MODEL_PATH.resolve()}/{time_stamp_path}/sorted_normalized_distances_trunc.npy")
            print("Loaded truncated sorted pairwise embedding normalized distances")
        else:
            print("No saved truncated sorted pairwise embedding distances normalized found")

        ood_model = OODModel(
            sorted_distances=sorted_distances_truncated,
            sorted_normalized_distances=sorted_normalized_distances_truncated,
            embeddings=embedding_model.reduced_embeddings, 
            normalized_embeddings=embedding_model.normalized_embeddings,
            probs=probs, 
            predictions=predictions)
        print("Created OOD model")

        return ModelHandler(model_name=model_name,
                            inference_model=inference_model,
                            embedding_model=embedding_model,
                            ood_model=ood_model,
                            dataset=dataset,
                            indices=indices)

    def get_melanoma_score(self, image_name='', image=None) -> int:
        if image_name != '' and image is None:
            image_idx = self.dataset.get_index_by_image_name(image_name)
            prediction_prob = self.inference_model.probs[image_idx]
        elif image_name == '' and image is not None:
            _, prediction_prob = self.inference_model.predict_image(dataset=self.dataset,
                                                                    image=image)
        else:
            raise Exception("Need to provide image name xor image")

        if self.model_name == MODELNAME_BINARY:
            return int(prediction_prob * 10)
        elif self.model_name == MODELNAME_MULTI:
            return int(prediction_prob[1] * 10)
        else:
            raise Exception("Unknown model name")

    def get_reliability_score(self, image_name, image=None) -> int:
        image_idx = self.dataset.get_index_by_image_name(image_name)
        if image is None:
            embeddings = self.embedding_model.reduced_embeddings
            return self.ood_model.get_ood_score(image_embedding=embeddings[image_idx],
                                                embeddings=embeddings)
        else:
            # If image is provided, update embedding at that index
            transformed_image = self.dataset.transform_image(image)
            image_embedding = self.embedding_model.compute_reduced_embedding(transformed_image)
            embeddings = self.embedding_model.reduced_embeddings.copy()
            embeddings[image_idx, :] = image_embedding
            return self.ood_model.get_ood_score(image_embedding=image_embedding,
                                                embeddings=embeddings)

    def get_reliability_score_knn_prediction_variance(self, image_name, train_indices, dataset, image=None) -> int:
        embeddings = self.embedding_model.reduced_embeddings[train_indices]
        image_idx = self.dataset.get_index_by_image_name(image_name)
        if image is None:
            image_embedding = self.embedding_model.reduced_embeddings[image_idx]
            return self.ood_model.get_ood_score_prediction_variance(image_embedding=image_embedding, embeddings=embeddings, train_indices=train_indices)
        else:
            prediction_x, prob = self.inference_model.predict_image(
                dataset=dataset, image=image
            )
            image_embedding = self.embedding_model.compute_reduced_embedding(image)
            return self.ood_model.get_ood_score_prediction_variance(image_embedding=image_embedding, embeddings=embeddings, train_indices=train_indices, prob=prob), prediction_x, prob

    def get_reliability_score_normalized_knn_prediction_variance(self, image_name, train_indices, dataset, image=None) -> int:
        normalized_embeddings = self.embedding_model.normalized_embeddings[train_indices]
        image_idx = self.dataset.get_index_by_image_name(image_name)
        if image is None:
            normalized_image_embedding = self.embedding_model.normalized_embeddings[image_idx]
            return self.ood_model.get_ood_score_prediction_variance(image_embedding=normalized_image_embedding,
                                                embeddings=normalized_embeddings, train_indices=train_indices)
        else:
            prediction_x, prob = self.inference_model.predict_image(
                dataset=dataset, image=image
            )
            image_embedding = self.embedding_model.compute_reduced_embedding(image)
            normalized_image_embedding = image_embedding / np.linalg.norm(image_embedding)
            return self.ood_model.get_ood_score_prediction_variance(image_embedding=normalized_image_embedding,
                                                embeddings=normalized_embeddings, train_indices=train_indices, prob=prob), prediction_x, prob

    def get_reliability_score_normalized_knn_distance(self, image_name, train_indices, image=None) -> int:
        normalized_embeddings = self.embedding_model.normalized_embeddings[train_indices]
        image_idx = self.dataset.get_index_by_image_name(image_name)
        if image is None:
            assert train_indices is not None
            normalized_image_embedding = self.embedding_model.normalized_embeddings[image_idx]
            return self.ood_model.get_ood_score_normalized_distance(image_embedding=normalized_image_embedding,
                                                embeddings=normalized_embeddings)
        else:
            image_embedding = self.embedding_model.compute_reduced_embedding(image)
            normalized_image_embedding = image_embedding / np.linalg.norm(image_embedding)
            return self.ood_model.get_ood_score_normalized_distance(normalized_image_embedding=normalized_image_embedding,
                                                normalized_embeddings=normalized_embeddings)

    def get_most_similar_images(self, image_name, image=None, use_ground_truth=False, save_plot=True) -> tuple[str, str]:
        predictions = None
        if not use_ground_truth:
            predictions = self.inference_model.predictions

        if image is not None:
            image = self.dataset.transform_image(image)

        image_names = self.embedding_model.get_most_similar_by_image_name(dataset=self.dataset,
                                                                          image=image,
                                                                          image_name=image_name,
                                                                          predictions=predictions)
        name_similar_malig, name_similar_benign = image_names

        if save_plot:
            display_images([image_name, *image_names], dataset=self.dataset, display=False, save=True)
        print(f"Computed images most similar to {image_name}")

        return name_similar_malig, name_similar_benign

    def get_saliency_map(self, image, target=1, single=False) -> pyplot.plot:
        transformed_image = self.dataset.transform_image(image)
        return self.inference_model.compute_saliency_map(transformed_image, target, single=single)


# TODO: Incorporate ABC to make properly abstract
class InferenceModel:
    def __init__(self, model: nn.Module, probs=None, predictions=None):
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.integrated_gradients = IntegratedGradients(self.model)
        self.noise_tunnel = NoiseTunnel(self.integrated_gradients)
        self.loss_history = []  # List of tuples (train_loss, val_loss) for each epoch after training
        self.probs = probs
        if predictions is None and probs is not None:
            predictions = (probs > 0.5).astype(int)
        self.predictions = predictions

    def train_model(self, train_loader, val_loader, lr=0.02, num_epochs=20, criterion_weight=None):
        raise NotImplementedError("Subclass must implement abstract method")

    def validate_model(self, val_loader, criterion):
        raise NotImplementedError("Subclass must implement abstract method")

    def evaluate_model(self, test_loader, write_to_csv=False):
        raise NotImplementedError("Subclass must implement abstract method")

    def compute_predictions(self, input_data_loader: DataLoader, set_attribute=False):
        raise NotImplementedError("Subclass must implement abstract method")

    def predict_image(self, dataset, image):
        raise NotImplementedError("Subclass must implement abstract method")

    def predict_image_by_name(self, dataset, image_name):
        raise NotImplementedError("Subclass must implement abstract method")

    def compute_saliency_map(self, image, target, single=False):
        self.model.eval()

        image = torch.Tensor(image[None, :, :, :]).to(self.device)
        target = torch.tensor([target], dtype=torch.long).to(self.device)

        # if target is not None:
        #     target = torch.Tensor(np.array([target]).reshape((1, 1)))
        #     target = target.to(self.device)

        # Compute attributions using Integrated Gradients
        attributions_ig = self.integrated_gradients.attribute(image,
                                                            #   target=target,
                                                              n_steps=15)

        default_cmap = LinearSegmentedColormap.from_list('custom blue',
                                                         [(0, '#ffffff'),
                                                          (0.25, '#000000'),
                                                          (1, '#000000')], N=256)

        # plt.clf()
        if single:
            plot_ret = viz.visualize_image_attr(
                attr=np.transpose(attributions_ig.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                original_image=np.transpose(image.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                method='blended_heat_map',
                cmap=pyplot.colormaps['YlOrRd'],
                show_colorbar=True,
                # sign='positive',
                # outlier_perc=1,
                alpha_overlay=0.8,
                use_pyplot=False)
        else:
            plot_ret = viz.visualize_image_attr_multiple(
                attr=np.transpose(attributions_ig.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                original_image=np.transpose(image.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                methods=["original_image", "heat_map"],
                signs=["all", "positive"],
                cmap=pyplot.colormaps['YlOrRd'],
                show_colorbar=True,
                use_pyplot=False)

        plot_ret[0].savefig(SALIENCY_MAP_PATH)
        print("Saved saliency map")


class InferenceModelMulti(InferenceModel):
    def __init__(self, model: nn.Module, probs=None, predictions=None):
        super().__init__(model, probs, predictions)
    
    def train_model(self, train_loader, val_loader, lr=0.02, num_epochs=20, criterion_weight=None):
        print(f"Training on device: {self.device}.")

        criterion = nn.CrossEntropyLoss()
        criterion.to(self.device)

        optimizer = Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

        early_stopping_patience = num_epochs
        early_stopping_counter = 0

        best_val_loss = float("inf")
        best_model_weights = deepcopy(self.model.state_dict())
        self.loss_history = []

        for epoch in range(num_epochs):
            self.model.train()  # Set model to training mode
            running_loss = 0.0
            for data in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit='batch'):
                # Get the inputs; data is a dict of [inputs, labels] + adjust label dimensions
                inputs, labels = data['image'].to(self.device), data['target'].to(self.device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward + backward + optimize
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # Print statistics
                running_loss += loss.item()

            # Print loss per epoch
            avg_train_loss = running_loss / len(train_loader)
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_train_loss:.4f}')

            # Validation phase
            avg_val_loss = self.validate_model(val_loader, criterion)

            # Add loss to history for later analysis
            self.loss_history.append((avg_train_loss, avg_val_loss))

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_weights = deepcopy(self.model.state_dict())
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= early_stopping_patience:
                    print(f"Early stopping triggered in epoch {epoch}")
                    break

            # Adjust learning rate
            scheduler.step()

        # Load the best model weights
        self.model.load_state_dict(best_model_weights)

    def validate_model(self, val_loader, criterion):
        self.model.eval()  # Set model to evaluation mode
        val_loss = 0.0
        all_labels = []
        all_predictions = []
        with torch.no_grad():
            for data in val_loader:
                inputs, labels = data['image'].to(self.device), data['target'].to(self.device)
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted_labels = torch.max(outputs, 1)
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted_labels.cpu().numpy())

        avg_loss = val_loss / len(val_loader)
        
        accuracy = 100 * (np.array(all_predictions) == np.array(all_labels)).mean()
        f1 = f1_score(np.array(all_labels), np.array(all_predictions), average='weighted')

        print(f'Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%, F1 Score: {f1:.2f}')

        return avg_loss

    def evaluate_model(self, test_loader, write_to_csv=False):
        self.model.eval()
        if write_to_csv:
            test_loader.dataset.set_eval_mode(True)
        
        all_labels = []
        all_predictions = []
        with torch.no_grad():
            for data_index, data in enumerate(test_loader):
                print(f"evaluate {data_index}/{len(test_loader)}")
                images = data['image'].to(self.device)
                labels = data['target'].to(self.device)
                outputs = self.model(images)
                _, predicted_labels = torch.max(outputs, 1)
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted_labels.cpu().numpy())

                if write_to_csv:
                    ages = data['meta_data']['age_approx'].to(self.device).float()
                    names = data['meta_data']['image_name']
                    sex = data['meta_data']['sex']
                    for i in range(len(labels)):
                        write_csv_row([names[i], ages[i].item(), sex[i], labels[i].item(), predicted_labels[i].item()])

        # Calculating metrics
        print(classification_report(all_labels, all_predictions, digits=4))
        accuracy = accuracy_score(all_labels, all_predictions)
        print(f'Test Accuracy: {accuracy:.2f}%')
        return accuracy

    def compute_predictions(self, input_data_loader: DataLoader, set_attribute=False):
        self.model.eval()
        predictions = []
        probs = []
        with torch.no_grad():
            for data in input_data_loader:
                inputs = data['image'].to(self.device)
                outputs = self.model(inputs)
                _, predicted_labels = torch.max(outputs, 1)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                predictions.extend(predicted_labels.cpu().numpy())
                probs.extend(probabilities.cpu().numpy())
                
        if set_attribute:
            self.probs = probs
            self.predictions = predictions

        return predictions, probs

    def predict_image(self, dataset, image):
        self.model.eval()
        transformed_image = dataset.transform_image(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(transformed_image)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
        return predicted_class, probabilities.squeeze().tolist()

    def predict_image_by_name(self, dataset, image_name):
        self.model.eval()
        transformed_image = dataset.load_image(image_name).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(transformed_image)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
        return predicted_class, probabilities.squeeze().tolist()


class InferenceModelBinary(InferenceModel):
    def __init__(self, model: nn.Module, probs=None, predictions=None):
        super().__init__(model, probs, predictions)
    
    def train_model(self, train_loader, val_loader, lr=0.001, num_epochs=20, criterion_weight=5):
        print(f"Training on device: {self.device}.")

        # Criterion weight specicifies how much more important the correct prediction of melanoma is
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([criterion_weight]))
        optimizer = Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        criterion.to(self.device)
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

        early_stopping_patience = 20
        early_stopping_counter = 0

        best_val_loss = float("inf")
        best_model_weights = deepcopy(self.model.state_dict())
        self.loss_history = []

        for epoch in range(num_epochs):
            self.model.train()  # Set model to training mode
            running_loss = 0.0
            for data in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit='batch'):
                # Get the inputs; data is a dict of [inputs, labels] + adjust label dimensions
                inputs, labels = data['image'].to(self.device), data['target'].to(self.device).float().unsqueeze(1)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward + backward + optimize
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # Print statistics
                running_loss += loss.item()

            # Print loss per epoch
            avg_train_loss = running_loss / len(train_loader)
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_train_loss:.4f}')

            # Validation phase
            avg_val_loss = self.validate_model(val_loader, criterion)

            # Add loss to history for later analysis
            self.loss_history.append((avg_train_loss, avg_val_loss))

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_weights = deepcopy(self.model.state_dict())
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= early_stopping_patience:
                    print(f"Early stopping triggered in epoch {epoch}")
                    break

            # Adjust learning rate
            scheduler.step()

        # Load the best model weights
        self.model.load_state_dict(best_model_weights)

    def validate_model(self, val_loader, criterion):
        self.model.eval()  # Set model to evaluation mode
        val_loss = 0.0
        all_labels = []
        all_predictions = []
        with torch.no_grad():
            for data in val_loader:
                inputs, labels = data['image'].to(self.device), data['target'].to(self.device).float().unsqueeze(1)
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                predicted_probs = torch.sigmoid(outputs).cpu().numpy()
                predicted_labels = (predicted_probs > 0.5).astype(int)
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted_labels)

        avg_loss = val_loss / len(val_loader)
        accuracy = 100 * (np.array(all_predictions) == np.array(all_labels)).mean()
        f1 = f1_score(np.array(all_labels), np.array(all_predictions))

        print(f'Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%, F1 Score: {f1:.2f}')

        return avg_loss
    
    def evaluate_model(self, test_loader, write_to_csv=False):
        self.model.eval()
        if write_to_csv:
            test_loader.dataset.set_eval_mode(True)
        
        all_probs = []  # List to accumulate probability scores
        all_labels = []
        with torch.no_grad():
            for data_index, data in enumerate(test_loader):
                print(f"evaluate {data_index}/{len(test_loader)}")
                images = data['image'].to(self.device)
                labels = data['target'].to(self.device).float().unsqueeze(1)
                outputs = self.model(images)
                probs = torch.sigmoid(outputs)
                if write_to_csv:
                    ages = data['age_approx'].to(self.device).float()
                    names = data['image_name']
                    sex = data['sex']
                    for i in range(len(labels)):
                        write_csv_row([names[i], ages[i].item(), sex[i], labels[i].item(), probs[i].item()])
                all_probs.extend(probs.cpu().numpy())  # Accumulate probability scores
                all_labels.extend(labels.cpu().numpy())

        # Convert lists to numpy arrays for sklearn metrics
        all_probs = np.array(all_probs).flatten()  # Ensure all_probs is flattened
        all_labels = np.array(all_labels).flatten()

        # Calculating metrics
        predicted = (all_probs > 0.5).astype(int)  # Convert probabilities to binary predictions
        tn, fp, fn, tp = confusion_matrix(all_labels, predicted).ravel()
        precision = precision_score(all_labels, predicted)
        recall = recall_score(all_labels, predicted)
        f1 = f1_score(all_labels, predicted)
        auc_roc = roc_auc_score(all_labels, all_probs)  # Use all_probs for AUC-ROC calculation

        # Printing metrics
        print(f'TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}')
        print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}, AUC-ROC: {auc_roc:.4f}')

        accuracy = 100 * (tp + tn) / (tp + tn + fp + fn)
        print(f'Test Accuracy: {accuracy:.2f}%')
        return accuracy

    def compute_predictions(self, input_data_loader: DataLoader, set_attribute=False):
        # Inference and get predictions
        self.model.eval()

        predictions = np.empty(len(input_data_loader.dataset), dtype=int)
        probs = np.empty(len(input_data_loader.dataset), dtype=float)
        index = 0
        with torch.no_grad():
            for data in input_data_loader:
                inputs = data['image'].to(self.device)
                outputs = self.model(inputs)
                predicted_probs = torch.sigmoid(outputs).cpu().numpy()  # Apply sigmoid to convert to probabilities
                predicted_labels = (predicted_probs > 0.5).astype(int)  # Apply threshold to get binary predictions

                batch_size = inputs.size(0)
                probs[index:index + batch_size] = predicted_probs.squeeze()
                predictions[index:index + batch_size] = predicted_labels.squeeze()
                index += batch_size

        if set_attribute:
            self.probs = probs
            self.predictions = predictions

        return predictions, probs

    def predict_image(self, dataset, image):
        self.model.eval()  # Ensure the model is in evaluation mode
        image = image.unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(image)
            probability = float(torch.sigmoid(output).item())  # Convert logit to probability

        predicted_class = int(probability > 0.5)  # Class prediction based on threshold
        return predicted_class, probability

    def predict_image_by_name(self, dataset, image_name):
        # Ensure the model is in evaluation mode
        self.model.eval()
        # Add batch dimension and send to device
        transformed_image = dataset.load_image(image_name).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(transformed_image)
            probability = float(torch.sigmoid(output).item())  # Convert logit to probability

        predicted_class = int(probability > 0.5)  # Class prediction based on threshold
        return predicted_class, probability


class EmbeddingModel(nn.Module):
    def __init__(self, original_model: nn.Module, pca=None, embeddings=None, reduced_embeddings=None, normalized_embeddings=None):
        super(EmbeddingModel, self).__init__()
        # Assuming the last layer is a fully connected layer, we take all but the last layer
        self.model = nn.Sequential(*list(original_model.children())[:-1])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.pca: PCA = pca
        self.embeddings = embeddings
        self.reduced_embeddings = reduced_embeddings
        self.normalized_embeddings = normalized_embeddings

    def forward(self, x):
        # Forward pass through all layers except the last one
        x = self.model(x)
        # Flatten the output
        return torch.flatten(x, 1)

    # TODO: Improve efficieny by directly using a numpy array
    def compute_embeddings(self, data_loader, set_attribute=False):
        print("Compute embeddings")
        self.model.eval()
        embedding_features = []
        with torch.no_grad():
            for data in tqdm(data_loader):
                inputs = data['image'].to(self.device)
                feature = self.model(inputs)
                embedding_features.append(feature.cpu().numpy())
        embeddings = np.concatenate(embedding_features, axis=0).squeeze()

        if set_attribute:
            self.embeddings = embeddings
        return embeddings

    def compute_PCA(self, embeddings=None, n_components=6, set_attribute=False):
        pca = PCA(n_components)

        if embeddings is None:
            embeddings = self.embeddings

        pca.fit(embeddings)

        if set_attribute:
            self.pca = pca
        return pca
    
    def compute_reduced_embeddings(self, embeddings=None, set_attribute=False):
        reduced_embeddings = None
        if embeddings is None:
            embeddings = self.embeddings

        if self.pca is not None:
            reduced_embeddings = self.pca.transform(embeddings)
            if set_attribute:
                self.reduced_embeddings = reduced_embeddings
        return reduced_embeddings

    
    def compute_reduced_embedding(self, image):
        self.model.eval()
        image = image.unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_embedding = self.model(image).cpu().numpy().squeeze()
        return self.pca.transform(image_embedding.reshape(1, -1)).flatten()


    def compute_normalized_embeddings(self, reduced_embeddings=None, set_attribute=False):
        normalized_embeddings = None
        if reduced_embeddings is None:
            reduced_embeddings = self.reduced_embeddings

        normalized_embeddings = reduced_embeddings / np.linalg.norm(reduced_embeddings, axis=1, keepdims=True)
        if set_attribute:
            self.normalized_embeddings = normalized_embeddings
        return normalized_embeddings


    def inverse_reduced_embeddings(self, reduced_embeddings=None, set_attribute=False):
        embeddings = None
        if reduced_embeddings is None:
            reduced_embeddings = self.reduced_embeddings

        if self.pca is not None:
            embeddings = self.pca.inverse_transform(reduced_embeddings)
            if set_attribute:
                self.embeddings = embeddings

        return embeddings

    def find_most_similar_image_index(self, dataset: CustomImageDataset, image_name, image=None, predictions=None):
        from sklearn.metrics.pairwise import cosine_similarity
        if predictions is not None:
            labels = predictions
        else:
            labels = np.array(dataset.meta_data['target'])

        image_index = dataset.get_index_by_image_name(image_name)

        if image is not None:
            image_embedding = self.compute_reduced_embedding(image)
        else:
            image_embedding = self.reduced_embeddings[image_index]

        similarities = cosine_similarity(self.reduced_embeddings, image_embedding.reshape(1, -1)).flatten()

        # Remove image itself as potential similar image
        similarities[image_index] = -1

        # Find most similar with same label
        malig_mask = labels == 1
        similarities_malig = np.where(malig_mask, similarities, -1)
        index_most_similar_malig = np.argmax(similarities_malig)

        benign_mask = np.isin(labels, [DIAGNOSIS_TO_TARGET[i] for i in BENIGN_DIAGNOSIS])
        similarities_benign = np.where(benign_mask, similarities, -1)
        index_most_similar_benign = np.argmax(similarities_benign)

        return index_most_similar_malig, index_most_similar_benign

    # Most similar images to image identified by name. Give modified image as additional argument if image was changed.
    def get_most_similar_by_image_name(self, dataset, image_name, image=None, predictions=None):
        idx_malig, idx_benign = self.find_most_similar_image_index(dataset, image_name, image, predictions)
        name_malig = dataset.meta_data.at[idx_malig, 'image_name']
        name_benign = dataset.meta_data.at[idx_benign, 'image_name']
        return name_malig, name_benign


class OODModel:
    def __init__(self, sorted_distances=None, sorted_normalized_distances=None, embeddings=None, normalized_embeddings=None, probs=None, predictions=None, k=3, metric='mahalanobis') -> None:
        self.k = k
        self.metric = metric
        self.probs = probs
        self.predictions = predictions
        if sorted_distances is None:
            assert sorted_normalized_distances is None
            pairwise_distances = distance.cdist(embeddings, embeddings, metric)
            sorted_distances = np.sort(pairwise_distances, axis=1)[:, :20]  # includes dist to itself

            pairwise_normalized_distances = distance.cdist(normalized_embeddings, normalized_embeddings, metric)
            sorted_normalized_distances = np.sort(pairwise_normalized_distances, axis=1)[:, :20]  # includes dist to itself

        self.sorted_distances = sorted_distances
        if sorted_normalized_distances is not None:
            self.sorted_normalized_distances = sorted_normalized_distances

        self.k_distances = self.sorted_distances[:, k]
        if sorted_normalized_distances is not None:
            self.k_normalized_distances = self.sorted_normalized_distances[:, k]

        self.min_k_distance = np.min(self.k_distances)
        self.max_k_distance = np.max(self.k_distances)

        if sorted_normalized_distances is not None:
            self.min_k_normalized_distance = np.min(self.k_normalized_distances)
            self.max_k_normalized_distance = np.max(self.k_normalized_distances)

        # Compute 5th and 95th percentiles
        self.p5_k_distance = np.percentile(self.k_distances, 5)
        self.p95_k_distance = np.percentile(self.k_distances, 95)

        if sorted_normalized_distances is not None:
            self.p5_k_normalized_distance = np.percentile(self.k_normalized_distances, 5)
            self.p95_k_normalized_distance = np.percentile(self.k_normalized_distances, 95)
        
    def get_ood_score(self, image_embedding, embeddings):
        distances = distance.cdist([image_embedding], embeddings, self.metric).flatten()
        new_k_distance = np.sort(distances)[self.k]

        # Avoid division by zero in case all k_distances are the same
        if self.min_k_distance == self.max_k_distance:
            return 10 if new_k_distance > self.min_k_distance else 0

        # Normalize the new k-distance to a 0-10 scale
        # Handle cases where new_k_distance is outside the percentile range
        if new_k_distance < self.p5_k_distance:
            normalized_k_distance = 0
        elif new_k_distance > self.p95_k_distance:
            normalized_k_distance = 10
        else:
            normalized_k_distance = 10 * (new_k_distance - self.p5_k_distance) / (self.p95_k_distance - self.p5_k_distance)
        
        # Return 10 - k-distance for a value in 0-10 scale corresponding to the unusualness of the embedding
        return int(10 - normalized_k_distance)

    def get_ood_score_normalized_distance(self, normalized_image_embedding, normalized_embeddings):
        distances = distance.cdist([normalized_image_embedding], normalized_embeddings, self.metric).flatten()
        new_k_distance = np.sort(distances)[self.k]

        # Avoid division by zero in case all k_distances are the same
        assert self.min_k_normalized_distance != self.max_k_normalized_distance

        # Normalize the new k-distance to a 0-10 scale
        normalized_k_distance = 10 * (new_k_distance - self.p5_k_normalized_distance) / (self.p95_k_normalized_distance - self.p5_k_normalized_distance)
        
        return -normalized_k_distance


    def get_ood_score_distance(self, image_embedding, embeddings):
        distances = distance.cdist([image_embedding], embeddings, self.metric).flatten()
        new_k_distance = np.sort(distances)[self.k]

        # Avoid division by zero in case all k_distances are the same
        assert self.min_k_distance != self.max_k_distance

        # Normalize the new k-distance to a 0-10 scale
        k_distance = 10 * (new_k_distance - self.p5_k_distance) / (self.p95_k_distance - self.p5_k_distance)
        
        return -k_distance
    
    def get_ood_score_prediction_variance(self, image_embedding, embeddings, train_indices, prob, k_values=[3, 5, 10]):
        # Compute distances from the test embedding to all training embeddings
        distances = distance.cdist([image_embedding], embeddings, self.metric).flatten()
        sorted_neighbor_indices = np.argsort(distances)

        # # Get the probability variances of the k nearest neighbors
        # variances = [np.var(self.probs[sorted_neighbor_indices[:k]]) for k in k_values]

        # return -np.mean(variances)
        probs = [prob]
        probs.extend(self.probs[train_indices][sorted_neighbor_indices[:k_values[-1]]])

        variance = np.var(probs)

        return -variance

def create_efficientnet(multiclass=True):
    print("Create Efficientnet_v2_s\n")
    model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)

    # Adjust the model for multi-class classification with 9 output neurons
    num_features = model.classifier[1].in_features
    
    if multiclass:
        model.classifier[1] = nn.Linear(num_features, 9)
    else:
        model.classifier[1] = nn.Linear(num_features, 1)
    
    # Initialize newly added layer
    nn.init.xavier_uniform_(model.classifier[1].weight) 

    return model
