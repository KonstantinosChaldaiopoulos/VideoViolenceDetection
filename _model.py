import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import GridSearchCV
from torch.nn import Module, Conv2d, ConvTranspose2d, Linear, MaxPool2d, AdaptiveAvgPool2d, ReLU, Sequential
from torchvision.models import alexnet, squeezenet1_0
from sklearn.metrics import precision_recall_fscore_support
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from torch.optim import RMSprop
import torch.nn.functional as F
from _utils import *

class TRClassifier:
    def __init__(self, X, y, fn, num_features, classifier, scaler):
        self.X = X
        self.y = y
        self.fn = fn
        self.num_features = num_features
        self.clf = select_classifier(classifier)["instance"]
        self.scaler = select_scaler(scaler)
        self.pg = select_classifier(classifier)["param_grid"]
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.best_parameters = None
        self.y_val_pred = None
        self.y_test_pred = None
        self.val_confidence = None
        self.test_confidence = None
        self.val_accuracy = None
        self.test_accuracy = None
        
    def preprocess_data(self):
        self.X_train, self.X_val, self.X_test = self.X["training"], self.X["validation"], self.X["testing"]
        self.y_train, self.y_val, self.y_test = self.y["training"], self.y["validation"], self.y["testing"]
        self.scale_features() if self.scaler is not None else None
        self.select_features()
        
    def scale_features(self):
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_val = self.scaler.transform(self.X_val)
        self.X_test = self.scaler.transform(self.X_test)

    def select_features(self):
        selector = SelectKBest(f_classif, k=self.num_features)
        self.X_train = selector.fit_transform(self.X_train, self.y_train)
        self.X_val = selector.transform(self.X_test)
        self.X_test = selector.transform(self.X_test)
        if self.fn is not None: # None-case: text classification (BERT embeddings don't have feature names)
            self.fn = [self.fn[i] for i, selected in enumerate(selector.get_support()) if selected]
  
    def set_parameters(self):
        grid_search = GridSearchCV(self.clf, self.pg, cv=2) # creates a grid search object with 2-fold cross-validation
        grid_search.fit(self.X_train, self.y_train)
        self.best_parameters = grid_search.best_params_
        self.clf.set_params(**self.best_parameters)
        
    def get_feature_names(self):
        return self.fn
    
    def get_best_parameters(self):
        return self.best_parameters
    
    def train(self):
        self.clf.fit(self.X_train, self.y_train)

    def validate(self):
        self.y_val_pred = self.clf.predict(self.X_val)
        self.val_confidence = self.clf.predict_proba(self.X_val)
        self.val_accuracy = accuracy_score(self.y_val, self.y_val_pred)
        print("Validation Accuracy:", self.val_accuracy)
        
    def test(self):
        self.y_test_pred = self.clf.predict(self.X_test)
        self.test_confidence = self.clf.predict_proba(self.X_test)
        self.test_accuracy = accuracy_score(self.y_test, self.y_test_pred)
        print("Test Accuracy:", self.test_accuracy)

    def run(self):
        self.preprocess_data()
        self.set_parameters()
        self.train()
        self.validate()
        self.test()
        return self.val_accuracy, self.test_accuracy


class NNClassifier:
    def __init__(self, train_loader, val_loader, test_loader, epochs, lr, wd, device):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.epochs = epochs
        self.lr = lr
        self.wd = wd
        self.device = device
        self.bva = 0.0
        self.bvp = 0.0
        self.model = HybridNet(num_classes=2).to(self.device)
        self.optimizer = RMSprop(self.model.parameters(), lr=self.lr, weight_decay=self.wd)
        self.criterion = CrossEntropyLoss()
        self.y_test = []
        self.y_pred = []
        self.val_accuracy = None
        self.test_accuracy = None
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []

    def train(self, epoch):
        self.model.train()
        train_loss, train_steps, train_correct, train_total = 0, 0, 0, 0
        pbar = tqdm(self.train_loader, desc="Training", total=len(self.train_loader))
        for i, data in enumerate(pbar):
            inputs, labels = data
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            train_loss += loss.item()
            train_steps += 1
            pbar.set_postfix({'Training Loss': '{0:.3f}'.format(train_loss/train_steps)})
        self.train_losses.append(train_loss/train_steps)
        self.train_accuracies.append(100.0 * train_correct / train_total)
    
    def validate(self, epoch):
        self.model.eval()
        val_loss, val_steps, val_correct, val_total, val_pred, val_true = 0, 0, 0, 0, [], []
        with torch.no_grad():
            for i, data in enumerate(tqdm(self.val_loader, desc="Validating")):
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)  
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                val_pred.extend(predicted.cpu().numpy())
                val_true.extend(labels.cpu().numpy())
                loss = self.criterion(outputs, labels)
                val_loss += loss.item()
                val_steps += 1
        self.val_accuracy = val_correct / val_total
        self.val_losses.append(val_loss/val_steps)
        self.val_accuracies.append(100.0 * self.val_accuracy)
        val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(val_true, val_pred, average='weighted')
        print('Epoch: {}, Validation Accuracy: {:.2f} %, Validation Precision: {:.2f} %, Validation Recall: {:.2f} %, Validation F1-score: {:.2f} %'
                .format(epoch+1, self.val_accuracy*100, val_precision*100, val_recall*100, val_f1*100))
        if self.val_accuracy > self.bva or (self.val_accuracy >= self.bva and val_precision >= self.bvp) :
            self.bva = self.val_accuracy
            self.bvp = val_precision
            torch.save(self.model.state_dict(), 'best_model.pth')
            print('Model saved at epoch {} with Validation Accuracy: {:.2f}%'.format(epoch+1, self.bva))

    def test(self):
        self.model.load_state_dict(torch.load('best_model.pth'))
        self.model.eval()
        test_loss, test_correct, test_total = 0.0, 0, 0
        with torch.no_grad():
            for i, data in enumerate(tqdm(self.test_loader, desc="Testing")):
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
                test_loss += self.criterion(outputs, labels).item()
                self.y_pred.extend(predicted.cpu().numpy())
                self.y_test.extend(labels.cpu().numpy())
        self.test_accuracy = test_correct / test_total
        test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(self.y_test, self.y_pred, average='weighted')
        print('Test Accuracy: {:.2f} %, Test Precision: {:.2f} %, Test Recall: {:.2f} %, Test F1-score: {:.2f} %'
            .format(self.test_accuracy*100, test_precision*100, test_recall*100, test_f1*100))

    def run(self):
        for epoch in range(self.epochs):
            self.train(epoch)
            self.validate(epoch)
        plot_metrics(self.train_losses, self.val_losses, self.train_accuracies, self.val_accuracies)
        self.test()
        return self.bva, self.test_accuracy
    

class MNNClassifier:
    def __init__(self, train_loader, val_loader, test_loader, epochs, lr, wd, device):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.epochs = epochs
        self.lr = lr
        self.wd = wd
        self.device = device
        self.bva = 0.0
        self.model = HybridNet_With_Autoencoders().to(self.device)
        self.optimizer = RMSprop(self.model.parameters(), lr=self.lr, weight_decay=self.wd)
        self.criterion = CrossEntropyLoss()
        self.y_test = []
        self.y_pred = []
        self.test_accuracy = None
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []

    def train(self, epoch):
        self.model.train()
        train_loss, train_steps, train_correct, train_total = 0, 0, 0, 0
        pbar = tqdm(self.train_loader, desc="Training epoch " + str(epoch+1), total=len(self.train_loader))
        for inputs, audio_data, text_data, labels in pbar:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            audio_data, text_data = audio_data.to(self.device), text_data.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs, audio_data, text_data)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            train_steps += 1
        print('Train Loss: {:.6f}, Accuracy: {:.2f}%'.format(train_loss/len(self.train_loader), (100 * train_correct / train_total)))
        self.train_losses.append(train_loss/train_steps)
        self.train_accuracies.append(100.0 * train_correct / train_total)

    def validate(self, epoch):
        self.model.eval()
        val_loss, val_correct, val_total, val_steps = 0, 0, 0, 0
        with torch.no_grad():
            for inputs,audio_data, text_data,labels, in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                audio_data, text_data = audio_data.to(self.device), text_data.to(self.device)
                outputs = self.model(inputs, audio_data, text_data)
                loss = self.criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                val_steps += 1
            val_accuracy = val_correct / val_total
            print('Val Loss: {:.6f}, Accuracy: {:.2f}%'.format(val_loss/len(self.val_loader), (100 * val_accuracy)))
            if val_accuracy >= self.bva:
                self.bva = val_accuracy
                torch.save(self.model.state_dict(), 'best_model.pth')
                print(f'Model saved at epoch {epoch+1} with Accuracy: {self.bva*100:.2f}%')
        self.val_losses.append(val_loss/val_steps)
        self.val_accuracies.append(100.0 * val_accuracy)   

    def test(self):
        self.model.load_state_dict(torch.load('best_model.pth'))
        self.model.eval()
        test_loss, test_correct, test_total = 0.0, 0, 0
        with torch.no_grad():
            for i, data in enumerate(tqdm(self.test_loader, desc="Testing")):
                inputs, audio_data, text_data, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                audio_data, text_data = audio_data.to(self.device), text_data.to(self.device)
                outputs = self.model(inputs, audio_data, text_data)
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
                test_loss += self.criterion(outputs, labels).item()
                self.y_pred.extend(predicted.cpu().numpy())
                self.y_test.extend(labels.cpu().numpy())
        self.test_accuracy = test_correct / test_total
        test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(self.y_test, self.y_pred, average='weighted')
        print('Test Accuracy: {:.2f} %, Test Precision: {:.2f} %, Test Recall: {:.2f} %, Test F1-score: {:.2f} %'
            .format(100.0*self.test_accuracy, 100*test_precision, 100*test_recall, 100*test_f1))

    def run(self):
        for epoch in range(self.epochs):
            self.train(epoch)
            self.validate(epoch)
        plot_metrics(self.train_losses, self.val_losses, self.train_accuracies, self.val_accuracies)
        self.test()
        return self.test_accuracy


class ConvLSTMCell(Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        super(ConvLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        padding = kernel_size // 2
        self.conv = Conv2d(input_dim + hidden_dim, 4 * hidden_dim, kernel_size, 1, padding, bias=bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class ConvLSTM(Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super(ConvLSTM, self).__init__()
        self.cell = ConvLSTMCell(input_dim, hidden_dim, kernel_size, bias)

    def forward(self, input_tensor, hidden_state=None):
        b, t, c, h, w = input_tensor.size()
        if hidden_state is None:
            hidden_state = (torch.zeros(b, self.cell.hidden_dim, h, w).to(input_tensor.device),
                            torch.zeros(b, self.cell.hidden_dim, h, w).to(input_tensor.device))
        outputs = []
        for t in range(t):
            hidden_state = self.cell(input_tensor[:, t, :, :, :], hidden_state)
            outputs.append(hidden_state[0])
        return torch.stack(outputs, dim=1), hidden_state


class HybridNet(Module):
    def __init__(self, num_classes=2):
        super(HybridNet, self).__init__()
        alex = alexnet(pretrained=True)
        squeeze = squeezenet1_0(pretrained=True)
        self.alex_features = alex.features
        self.squeeze_features = squeeze.features
        self.convLSTM_alex = ConvLSTM(256, 256, kernel_size=3)
        self.convLSTM_squeeze = ConvLSTM(512, 256, kernel_size=3)
        self.max_pool = MaxPool2d(kernel_size=2, stride=2)
        self.adaptive_pool = AdaptiveAvgPool2d((6,6))
        self.fc1 = Linear(512 * 3 * 3, 256)
        self.fc2 = Linear(256, num_classes)

    def forward(self, x):
      b, t, c, h, w = x.size()
      alex_out = self.alex_features(x.view(b * t, c, h, w))
      alex_out = alex_out.view(b, t, alex_out.size(1), alex_out.size(2), alex_out.size(3))
      squeeze_out = self.squeeze_features(x.view(b * t, c, h, w))
      squeeze_out = squeeze_out.view(b, t, squeeze_out.size(1), squeeze_out.size(2), squeeze_out.size(3))
      alex_out, alex_state = self.convLSTM_alex(alex_out)
      squeeze_out, squeeze_state = self.convLSTM_squeeze(squeeze_out)
      alex_out = alex_out[:, -1]
      squeeze_out = squeeze_out[:, -1]
      squeeze_out = self.adaptive_pool(squeeze_out)
      combined = torch.cat([alex_out, squeeze_out], dim=1)
      pooled = self.max_pool(combined)
      flat = pooled.view(b, -1)
      fc1_out = self.fc1(flat)
      out1 = self.fc2(fc1_out)
      out = F.softmax(out1, dim=1)
      return out


class Encoder(Module):
    def __init__(self, input_dim, encoding_dim):
        super(Encoder, self).__init__()
        self.fc1 = Linear(input_dim, input_dim//2)
        self.fc2 = Linear(input_dim//2, input_dim//4)
        self.fc3 = Linear(input_dim//4, encoding_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x


class Decoder(Module):
    def __init__(self, encoding_dim, input_dim):
        super(Decoder, self).__init__()
        self.fc1 = Linear(encoding_dim, input_dim//4)
        self.fc2 = Linear(input_dim//4, input_dim//2)
        self.fc3 = Linear(input_dim//2, input_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x


class AutoEncoder(Module):
    def __init__(self, input_dim, encoding_dim):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(input_dim, encoding_dim)
        self.decoder = Decoder(encoding_dim, input_dim)

    def forward(self, x):
        encoded = self.encoder((x.float()))
        decoded = self.decoder(encoded)
        return decoded, encoded


class SmallEncoder(Module):
    def __init__(self, input_dim, encoding_dim):
        super(SmallEncoder, self).__init__()
        self.fc1 = Linear(input_dim, input_dim//2000)
        self.fc2 = Linear(input_dim//2000 ,encoding_dim)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


class SmallDecoder(Module):
    def __init__(self, encoding_dim, input_dim):
        super(SmallDecoder, self).__init__()
        self.fc1 = Linear(encoding_dim, input_dim//2000)
        self.fc2 = Linear(input_dim//2000, input_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


class SmallAutoEncoder(Module):
    def __init__(self, input_dim, encoding_dim):
        super(SmallAutoEncoder, self).__init__()
        self.encoder = SmallEncoder(input_dim, encoding_dim)
        self.decoder = SmallDecoder(encoding_dim, input_dim)

    def forward(self, x):
        encoded = self.encoder((x.float()))
        decoded = self.decoder(encoded)
        return decoded, encoded


class HybridNet_With_Autoencoders(Module):
    def __init__(self):
        super(HybridNet_With_Autoencoders, self).__init__()
        alex = alexnet(pretrained=True)
        squeeze = squeezenet1_0(pretrained=True)
        self.alex_features = alex.features
        self.squeeze_features = squeeze.features
        self.convLSTM_alex = ConvLSTM(256, 256, kernel_size=3)
        self.convLSTM_squeeze = ConvLSTM(512, 256, kernel_size=3)
        self.max_pool = MaxPool2d(kernel_size=2, stride=2)
        self.adaptive_pool = AdaptiveAvgPool2d((6,6))
        self.reduce_dim = Linear(512, 256)
        self.dropout = nn.Dropout(0.5)
        self.gap = Linear(512 * 3 * 3, 256)
        self.audio_autoencoder = AutoEncoder(136, encoding_dim=256)
        self.text_autoencoder = SmallAutoEncoder(53760, encoding_dim=1)
        self.feature_autoencoder = AutoEncoder(513 ,256)
        self.fc1 = Linear(256, 2)

    def forward(self, x, audio_data, text_data):
        _, audio_encoded = self.audio_autoencoder(audio_data)
        _, text_encoded = self.text_autoencoder(text_data)
        
        b, t, c, h, w = x.size()
        
        alex_out = self.alex_features(x.view(b * t, c, h, w))
        alex_out = alex_out.view(b, t, alex_out.size(1), alex_out.size(2), alex_out.size(3))
        
        squeeze_out = self.squeeze_features(x.view(b * t, c, h, w))
        squeeze_out = squeeze_out.view(b, t, squeeze_out.size(1), squeeze_out.size(2), squeeze_out.size(3))
        
        alex_out, alex_state = self.convLSTM_alex(alex_out)
        squeeze_out, squeeze_state = self.convLSTM_squeeze(squeeze_out)
        
        alex_out = alex_out[:, -1]
        squeeze_out = squeeze_out[:, -1]
        squeeze_out = self.adaptive_pool(squeeze_out)
        
        combined = torch.cat([alex_out, squeeze_out], dim=1)
        gap = AdaptiveAvgPool2d(1)
        pooled = gap(combined)
        pooled2 = pooled.view(pooled.size(0), -1)
        reduced_pooled2 = self.reduce_dim(pooled2)
        combined2 = torch.cat([reduced_pooled2, audio_encoded, text_encoded], dim=1)
        _, reduced = self.feature_autoencoder(combined2)
        
        dropped_out = self.dropout(reduced)  
        out1 = self.fc1(dropped_out)  
        out = F.softmax(out1, dim=1)
        
        return out
